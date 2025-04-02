#!/usr/bin/env python3

import argparse
import datetime
import gc
import logging
import multiprocessing
import random
import re
import tempfile
import time
import tracemalloc
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Iterable, NamedTuple

import psutil
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
DEFAULT_LOW_COV_BASES_TO_PRIORITIZE = 10


class AsyncJob(NamedTuple):
    job_input: ReferenceJobInput
    async_result: AsyncResult


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_read_count(bamfile: Path, reference: str) -> int:
    bamfile.seek(0)
    read_count = bamfile.count(reference=reference)
    bamfile.seek(0)
    return read_count


def parse_memory_limit(mem_str: str) -> float:
    mem_str = mem_str.strip().upper()
    match = re.match(r"^([\d.]+)\s*(TB|GB|MB|KB|B)?$", mem_str)
    if not match:
        raise ValueError(f"Invalid memory limit format: {mem_str}")
    value, unit = match.groups()
    value = float(value)
    unit = unit or "GB"
    unit_multipliers = {
        "TB": 1024,
        "GB": 1,
        "MB": 1 / 1024,
        "KB": 1 / (1024**2),
        "B": 1 / (1024**3),
    }
    return value * unit_multipliers[unit]


def get_used_memory_gb() -> float:
    return psutil.virtual_memory().used / 1e9


def adaptive_parallel_run(
    pool,
    jobs: list[ReferenceJobInput],
    threads: int,
    contigs_to_parallelize_on: int,
    max_memory_gb: float | None = None,
    check_interval: datetime.timedelta = datetime.timedelta(seconds=1),
) -> Iterable[ReferenceJobOutput]:
    active: list[AsyncJob] = list()
    pending: list[ReferenceJobInput] = jobs.copy()
    completed_bar = tqdm.tqdm(
        total=len(jobs), desc="Processing references", unit=" references"
    )
    try:
        while pending or active:
            threads_in_use = max(
                len(active), sum(async_job.job_input.threads for async_job in active)
            )
            available_threads = threads - threads_in_use
            available_slots = min(len(pending), contigs_to_parallelize_on - len(active))
            done = [result for result in active if result.async_result.ready()]
            for result in done:
                active.remove(result)
                completed_bar.update(1)
                gc.collect()
                yield result.async_result.get()
            # Launch a new job if memory allows
            if (
                pending
                and available_threads
                and available_slots
                and (
                    len(active) == 0
                    or max_memory_gb is None
                    or (0 < get_used_memory_gb() < max_memory_gb)
                )
            ):
                job: ReferenceJobInput = pending.pop(0)
                if available_threads > available_slots:
                    job.threads = max(
                        2, job.threads, available_threads // available_slots
                    )
                async_result = AsyncJob(
                    job_input=job,
                    async_result=pool.apply_async(process_reference, (job,)),
                )
                active.append(async_result)
            else:
                time.sleep(check_interval.total_seconds())
    finally:
        completed_bar.close()


def subsample_bam_parallel(
    input_bam: Path,
    output_bam: Path,
    desired_coverage: int = DEFAULT_COVERAGE,
    low_coverage_bases_to_prioritize: int = DEFAULT_LOW_COV_BASES_TO_PRIORITIZE,
    ignore_n_bases_on_edges: int = 0,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    contigs_to_parallelize_on: int = 1,
    max_memory: str | None = None,
    check_interval: datetime.timedelta = datetime.timedelta(seconds=1),
    verbose: bool = False,
) -> Path:
    tracemalloc.start()
    start = datetime.datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S") + f"{start.microsecond / 1e6:.4f}"[1:]
    if threads == 0:
        threads = max(1, (multiprocessing.cpu_count() - 1))
    elif threads < 0:
        raise ValueError(f"{threads=} (should be greater or equal to 0)")
    if contigs_to_parallelize_on == 0 or contigs_to_parallelize_on > threads:
        contigs_to_parallelize_on = threads
    elif contigs_to_parallelize_on < 0:
        raise ValueError(
            f"{contigs_to_parallelize_on=} (should be greater or equal to 0)"
        )
    track_with_tqdm = True if (contigs_to_parallelize_on == 1 or verbose) else False
    reference_job_outputs = list()
    reference_to_tmp_file_paths = dict()
    try:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as bamfile:
            references = bamfile.references
            logging.debug(f"{len(references)} references found in the BAM file.\n")
            reference_to_lengths: dict[str, int] = {
                reference: bamfile.get_reference_length(reference)
                for reference in references
            }
            reference_to_read_counts: dict[str, int] = (
                {reference: None for reference in references}
                # {
                #     reference: get_read_count(bamfile, reference=reference)
                #     for reference, _ in tqdm.tqdm(
                #         sorted(reference_to_lengths.items(), key=lambda ref_to_len: ref_to_len[1], reverse=True),
                #         desc="Calculating read count per reference",
                #         unit=" references"
                #     )
                # }
            )
            reference_to_tmp_file_paths = {
                reference: Path(
                    tempfile.NamedTemporaryFile(
                        delete=False, prefix=f"{timestamp}.{reference}.", suffix=f".bam"
                    ).name
                )
                for reference in references
            }
            args_list = [
                ReferenceJobInput(
                    reference=reference,
                    reference_length=reference_to_lengths[reference],
                    read_count=reference_to_read_counts[reference],
                    input_bam_path=input_bam,
                    desired_coverage=desired_coverage,
                    low_coverage_bases_to_prioritize=low_coverage_bases_to_prioritize,
                    ignore_n_bases_on_edges=ignore_n_bases_on_edges,
                    seed=(seed + i),
                    tmp_file_path=reference_to_tmp_file_paths[reference],
                    profile=True,
                    threads=max(
                        1,
                        threads // min(len(references), contigs_to_parallelize_on),
                    ),
                    track_with_tqdm=track_with_tqdm,
                    tqdm_position=(
                        i
                        if (track_with_tqdm and contigs_to_parallelize_on > 1)
                        else None
                    ),
                )
                for i, reference in enumerate(
                    sorted(
                        references,
                        key=lambda ref: (
                            reference_to_read_counts[ref],
                            reference_to_lengths[ref],
                            ref,
                        ),
                    )
                )
            ]

        if contigs_to_parallelize_on == 1:
            for job in args_list:
                reference_job_outputs.append(process_reference(job))
        else:
            with multiprocessing.Pool(
                processes=min(threads, contigs_to_parallelize_on)
            ) as pool:
                for result in adaptive_parallel_run(
                    pool,
                    jobs=args_list,
                    threads=threads,
                    contigs_to_parallelize_on=contigs_to_parallelize_on,
                    max_memory_gb=(
                        None
                        if max_memory is None
                        else (parse_memory_limit(max_memory) / threads)
                    ),
                    check_interval=check_interval,
                ):
                    reference_job_outputs.append(result)

        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            header = bamfile.header

        with pysam.AlignmentFile(output_bam, "wb", header=header) as final_bam:
            for reference_job_output in reference_job_outputs:
                with pysam.AlignmentFile(reference_job_output.temp_bam, "rb") as tmp_fh:
                    for read in tmp_fh:
                        final_bam.write(read)
                if reference_job_output.temp_bam.exists:
                    Path(reference_job_output.temp_bam).unlink()
        current, main_peak = tracemalloc.get_traced_memory()
        peak = main_peak + max(process.peak_memory for process in reference_job_outputs)
        logging.info("Done subsampling.")
        logging.info(
            f"It took {format_duration(datetime.datetime.now() - start)} time to run."
        )
        logging.debug(f"Peak memory usage in a SINGLE reference: {format_memory(peak)}")
        return output_bam
    except:
        logging.warning("Interrupted! Cleaning up temporary files...")
        current, main_peak = tracemalloc.get_traced_memory()
        peak = main_peak + sum(process.peak_memory for process in reference_job_outputs)
        logging.warning(
            f"It took {format_duration(datetime.datetime.now() - start)} time to run before the exception."
        )
        logging.warning(
            f"Peak memory usage in a single reference: {format_memory(peak)}"
        )
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
        help=f"Random seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of parallel processes to use (default {DEFAULT_THREADS}). 0 to use all threads (might require a lot of memory for large genomes)",
    )
    parser.add_argument(
        "--contigs-to-parallelize-on",
        type=int,
        default=1,
        help="How many contigs to parallelize on (default: 0 means one contig per thread)",
    )
    parser.add_argument(
        "--max-memory",
        type=str,
        default=None,
        help="Maximum memory usage (e.g. 20GB, 20000MB, or 20).",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=1,
        help="Interval in seconds to wait for before spawning a new job when parallelizing over contigs/references",
    )
    parser.add_argument(
        "--low-coverage-bases-to-prioritize",
        type=int,
        default=DEFAULT_LOW_COV_BASES_TO_PRIORITIZE,
        help=f"Prioritize reads with the N positions with lowest coverages (default: {DEFAULT_LOW_COV_BASES_TO_PRIORITIZE})",
    )
    parser.add_argument(
        "--ignore-n-bases-on-edges",
        type=int,
        default=0,
        help=f"Ignore N bases from start/end of a read when calculating coverage of each position (with the exception of reads starting/ending the contig) (default: 0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Whether it should print tqdm bars even when running multithreaded or not",
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
        contigs_to_parallelize_on=(
            args.contigs_to_parallelize_on
            if args.contigs_to_parallelize_on
            else args.threads
        ),
        max_memory=args.max_memory,
        check_interval=datetime.timedelta(seconds=args.check_interval),
        verbose=args.verbose,
        ignore_n_bases_on_edges=args.ignore_n_bases_on_edges,
        low_coverage_bases_to_prioritize=args.low_coverage_bases_to_prioritize,
    )
