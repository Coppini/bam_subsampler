#!/usr/bin/env python3

import argparse
import datetime
import logging
import multiprocessing
import random
import tempfile
import tracemalloc
from pathlib import Path
from shutil import move

import pysam
import tqdm

from subsampler_worker import (
    ContigJobInput,
    ContigJobOutput,
    format_duration,
    format_memory,
    process_contig,
)

DEFAULT_COVERAGE = 50
DEFAULT_SEED = 42
DEFAULT_THREADS = 1
DEFAULT_LOW_COV_BASES_TO_PRIORITIZE = 10

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_read_count(bamfile: Path, contig: str) -> int:
    """
    Count the number of reads mapped to a specific contig (contig) in a BAM file.

    Args:
        bamfile (Path): Path to the input BAM file.
        contig (str): Contig (contig) name.

    Returns:
        int: Number of reads aligned to the specified contig.
    """
    bamfile.seek(0)
    read_count = bamfile.count(contig=contig)
    bamfile.seek(0)
    return read_count


def subsample_bam_parallel(
    input_bam: Path,
    output_bam: Path,
    desired_coverage: int = DEFAULT_COVERAGE,
    coverage_cap: int = 0,
    large_coverage_matrix: bool = True,
    low_coverage_bases_to_prioritize: int = DEFAULT_LOW_COV_BASES_TO_PRIORITIZE,
    ignore_n_bases_on_edges: int = 0,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    contigs_to_parallelize_on: int = 1,
    verbose: bool = False,
) -> Path:
    """
    Subsample a BAM file to reach a target per-base coverage, using multi-threaded or multi-contig parallelization.

    Args:
        input_bam (Path): Path to the input BAM file.
        output_bam (Path): Path where the output BAM file will be written.
        desired_coverage (int): Target per-base coverage.
        coverage_cap (int): Maximum coverage value to consider when finding low coverage positions. Anything equal or higher than this value is considered the same when sorting by coverage.
        low_coverage_bases_to_prioritize (int): Number of low-coverage positions to prioritize during sorting.
        ignore_n_bases_on_edges (int): Number of bases to ignore from start/end of reads for coverage counting.
        seed (int): Random seed for reproducibility.
        threads (int): Number of threads to use.
        contigs_to_parallelize_on (int): Number of contigs/references to process in parallel.
        verbose (bool): Whether to show progress bars even in multi-contig mode.

    Returns:
        Path: Path to the final subsampled BAM file.
    """
    tracemalloc.start()
    start = datetime.datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S") + f"{start.microsecond / 1e6:.4f}"[1:]
    if threads == 0:
        threads = max(1, (multiprocessing.cpu_count() - 1))
    if contigs_to_parallelize_on == 0 or contigs_to_parallelize_on > threads:
        contigs_to_parallelize_on = threads
    track_with_tqdm = True if (contigs_to_parallelize_on == 1 or verbose) else False
    contig_to_tmp_file_paths = dict()
    contig_job_outputs = list()
    try:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as bamfile:
            contigs = bamfile.references
            logging.debug(f"{len(contigs)} contigs found in the BAM file.\n")
            contig_to_lengths: dict[str, int] = {
                contig: bamfile.get_reference_length(contig)
                for contig in contigs
            }
            contig_to_read_counts: dict[str, int] = (
                # {contig: None for contig in contigs}
                {
                    contig: get_read_count(bamfile, contig=contig)
                    for contig, _ in tqdm.tqdm(
                        sorted(
                            contig_to_lengths.items(),
                            key=lambda ref_to_len: ref_to_len[1],
                            reverse=True,
                        ),
                        desc="Calculating read count per contig",
                        unit=" contigs",
                    )
                }
            )
            contig_to_tmp_file_paths = {
                contig: Path(
                    tempfile.NamedTemporaryFile(
                        delete=False, prefix=f"{timestamp}.{contig}.", suffix=f".bam"
                    ).name
                )
                for contig in contigs
            }
            args_list = sorted(
                (
                    ContigJobInput(
                        contig=contig,
                        contig_length=contig_to_lengths[contig],
                        read_count=contig_to_read_counts[contig],
                        input_bam_path=input_bam,
                        desired_coverage=desired_coverage,
                        large_coverage_matrix=large_coverage_matrix,
                        coverage_cap=coverage_cap,
                        low_coverage_bases_to_prioritize=low_coverage_bases_to_prioritize,
                        ignore_n_bases_on_edges=ignore_n_bases_on_edges,
                        seed=(seed + i),
                        tmp_file_path=contig_to_tmp_file_paths[contig],
                        profile=True,
                        threads=max(
                            1,
                            threads // min(len(contigs), contigs_to_parallelize_on),
                        ),
                        track_with_tqdm=track_with_tqdm,
                        tqdm_position=(
                            i
                            if track_with_tqdm and contigs_to_parallelize_on > 1
                            else None
                        ),
                    )
                    for i, contig in enumerate(
                        sorted(
                            contigs,
                            key=lambda ref: (
                                contig_to_read_counts[ref],
                                contig_to_lengths[ref],
                                ref,
                            ),
                            reverse=True,
                        )
                    )
                ),
                key=lambda x: (x.read_count, x.contig_length, x.contig),
                reverse=True,
            )

        logging.info(
            f"Starting processes in {threads} {'parallel threads' if threads > 1 else 'single thread'}\n"
        )

        if contigs_to_parallelize_on == 1:
            for job in args_list:
                contig_job_outputs.append(process_contig(job))
        else:
            with multiprocessing.Pool(
                processes=min(threads, contigs_to_parallelize_on, len(contigs))
            ) as pool:
                for contig_job_output in tqdm.tqdm(
                    pool.imap_unordered(process_contig, args_list),
                    total=len(args_list),
                    desc="Processing contigs",
                ):
                    contig_job_outputs.append(contig_job_output)

        logging.debug(f"Writing subsample reads to {output_bam}\n")
        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            header = bamfile.header

        if len(contig_job_outputs) == 1:
            move(contig_job_outputs[0].temp_bam, output_bam)
        else:
            with pysam.AlignmentFile(output_bam, "wb", header=header) as final_bam:
                for contig_job_output in contig_job_outputs:
                    with pysam.AlignmentFile(
                        contig_job_output.temp_bam, "rb"
                    ) as tmp_fh:
                        for read in tmp_fh:
                            final_bam.write(read)
                    Path(contig_job_output.temp_bam).unlink()
        current, main_peak = tracemalloc.get_traced_memory()
        peak = main_peak + sum(sorted((process.peak_memory for process in contig_job_outputs), reverse=True)[:min(threads, contigs_to_parallelize_on)])
        logging.info(
            "Done subsampling.\n"
            f"It took {format_duration(datetime.datetime.now() - start)} time to run.\n"
            f"Possible peak memory usage: {format_memory(peak)}\n"
        )
        return output_bam
    except:
        logging.warning("Interrupted! Cleaning up temporary files...")
        current, main_peak = tracemalloc.get_traced_memory()
        peak = main_peak + sum(process.peak_memory for process in contig_job_outputs)
        logging.warning(
            f"It took {format_duration(datetime.datetime.now() - start)} time to run before the exception."
        )
        logging.warning(
            f"Peak memory usage in a single contig: {format_memory(peak)}"
        )
        raise
    finally:
        for tmp_file_path in contig_to_tmp_file_paths.values():
            if tmp_file_path.exists():
                tmp_file_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a BAM file to a target per-base coverage."
    )
    parser.add_argument("input_bam", type=Path, help="Path to the input BAM file.")
    parser.add_argument("output_bam", type=Path, help="Path to the output BAM file.")
    parser.add_argument(
        "-c",
        "--coverage",
        type=int,
        default=DEFAULT_COVERAGE,
        help=f"Desired per-base coverage (default: {DEFAULT_COVERAGE}).",
    )
    parser.add_argument(
        "--coverage-cap",
        type=int,
        default=0,
        help=(
            "Maximum coverage value to consider when finding low coverage positions."
            " Anything equal or higher than this value is considered the same when sorting by coverage."
            " (default: 0 for automatically using 2x desired coverage.)"
        ),
    )
    # parser.add_argument(
    #     "--large-coverage-matrix",
    #     action="store_true",
    #     default=False,
    #     help="Forces the subsampler to use a large coverage matrix, which uses more RAM, but can be faster for high depth input BAMs.",
    # )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of threads to use. 0 to use all threads. Safe to increase without increasing memory usage. (default: {DEFAULT_THREADS}).",
    )
    parser.add_argument(
        "--contigs-to-parallelize-on",
        type=int,
        default=1,
        help="[⚠️ Experimental - increases RAM usage a lot] How many contigs to parallelize on. 0 means one contig per thread. Higher numbers mean more usage. (default: 1 - do not parallelize over contigs)",
    )
    parser.add_argument(
        "--low-coverage-bases-to-prioritize",
        type=int,
        default=DEFAULT_LOW_COV_BASES_TO_PRIORITIZE,
        help=(
            "Prioritize reads with the N positions with lowest coverages"
            f" (default: {DEFAULT_LOW_COV_BASES_TO_PRIORITIZE})."
            " Higher values will increase memory usage but make the overall coverage distribution more uniform (less positions that end up above the desired coverage)."
            " Lower values will take less memory and increase randomness, but might allow for more coverage spikes in the resulting BAM."
            " Currently testing with values between 3 and 15."
        ),
    )
    parser.add_argument(
        "--ignore-n-bases-on-edges",
        type=int,
        default=0,
        help=(
            f"[⚠️ Experimental] Ignore N bases from start/end of a read when calculating coverage of each position"
            " (with the exception of reads starting/ending the contig). Might be useful if reads have lower quality on the start/end. (default: 0)"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Whether it should print tqdm bars even when running parallel over multiple contigs or not. By default, bars are shown when running in single-contig mode, but not when over several.",
    )

    args = parser.parse_args()

    multiprocessing.set_start_method("fork")

    random.seed(args.seed)

    subsample_bam_parallel(
        input_bam=args.input_bam,
        output_bam=args.output_bam,
        desired_coverage=args.coverage,
        coverage_cap=args.coverage_cap,
        large_coverage_matrix=True,  # args.large_coverage_matrix,
        seed=args.seed,
        threads=args.threads,
        contigs_to_parallelize_on=(
            args.contigs_to_parallelize_on
            if args.contigs_to_parallelize_on
            else args.threads
        ),
        verbose=args.verbose,
        ignore_n_bases_on_edges=args.ignore_n_bases_on_edges,
        low_coverage_bases_to_prioritize=args.low_coverage_bases_to_prioritize,
    )
