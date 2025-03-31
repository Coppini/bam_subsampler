#!/usr/bin/env python3

import argparse
import datetime
import logging
import multiprocessing
import random
import tempfile
import tracemalloc
from pathlib import Path

import pysam
import tqdm

from subsampler_worker import (
    ReferenceJobInput,
    ReferenceJobOutput,
    format_duration,
    format_memory,
    process_reference,
)

DEFAULT_COVERAGE = 50
DEFAULT_SEED = 42
DEFAULT_THREADS = 1

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def subsample_bam_parallel(
    input_bam: Path,
    output_bam: Path,
    desired_coverage: int = DEFAULT_COVERAGE,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
) -> Path:
    tracemalloc.start()
    start = datetime.datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S") + f"{start.microsecond / 1e6:.4f}"[1:]
    if threads == 0:
        threads = max(1, (multiprocessing.cpu_count() - 1))
    track_with_tqdm = True if threads == 1 else False
    try:
        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            reference_to_tmp_file_paths = {
                reference: Path(
                    tempfile.NamedTemporaryFile(
                        delete=False, prefix=f"{timestamp}.{reference}.", suffix=f".bam"
                    ).name
                )
                for reference in bamfile.references
            }
            logging.debug(
                f"{len(reference_to_tmp_file_paths)} references found in the BAM file."
            )
            args_list = sorted(
                (
                    ReferenceJobInput(
                        reference=reference,
                        reference_length=bamfile.get_reference_length(reference),
                        input_bam_path=input_bam,
                        desired_coverage=desired_coverage,
                        seed=(seed + i),
                        tmp_file_path=reference_to_tmp_file_paths[reference],
                        track_with_tqdm=track_with_tqdm,
                    )
                    for i, reference in tqdm.tqdm(
                        enumerate(list(reference_to_tmp_file_paths.keys())),
                        desc="Calculating read count and reference lengths",
                        unit=" references",
                    )
                ),
                key=lambda x: x.reference_length,
            )

        logging.info(
            f"Starting processes in {threads} {'parallele threads' if threads > 1 else 'single thread'}"
        )
        with multiprocessing.Pool(processes=threads) as pool:
            reference_job_outputs: list[ReferenceJobOutput] = list(
                tqdm.tqdm(
                    pool.imap_unordered(process_reference, args_list),
                    total=len(args_list),
                    desc="Processing references",
                )
            )

        logging.debug(f"Writing subsample reads to {output_bam}")
        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            header = bamfile.header

        with pysam.AlignmentFile(output_bam, "wb", header=header) as final_bam:
            for reference_job_output in reference_job_outputs:
                with pysam.AlignmentFile(reference_job_output.temp_bam, "rb") as tmp_fh:
                    for read in tmp_fh:
                        final_bam.write(read)
                if reference_job_output.temp_bam.exists:
                    Path(reference_job_output.temp_bam).unlink()
        current, peak = tracemalloc.get_traced_memory()
        logging.info(
            "Done subsampling.\n"
            f"It took {format_duration(datetime.datetime.now() - start)} time to run.\n"
            f"Peak memory usage: {format_memory(peak)}"
        )
        return output_bam
    except:
        logging.warning("Interrupted! Cleaning up temporary files...")
        for tmp_file_path in reference_to_tmp_file_paths.values():
            if tmp_file_path.exists():
                tmp_file_path.unlink()
        raise


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
        help=f"Number of parallel processes to use (default {DEFAULT_THREADS}). 0 to use all threads (might require a lot of memory for large genomes)",
    )

    args = parser.parse_args()

    multiprocessing.set_start_method("fork")

    random.seed(args.seed)

    subsample_bam_parallel(
        input_bam=args.input_bam,
        output_bam=args.output_bam,
        desired_coverage=args.coverage,
        seed=args.seed,
        threads=args.threads,
    )
