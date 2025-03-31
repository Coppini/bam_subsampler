#!/usr/bin/env python3

import argparse
import datetime
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


class AsyncJob(NamedTuple):
    job_input: ReferenceJobInput
    async_result: AsyncResult


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


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
    jobs: Iterable[ReferenceJobInput],
    threads: int,
    max_memory_gb: float | None = None,
    check_interval: datetime.timedelta = datetime.timedelta(seconds=1),
) -> Iterable[ReferenceJobOutput]:
    active: list[AsyncJob] = list()
    pending = jobs.copy()
    completed_bar = tqdm.tqdm(
        total=len(jobs), desc="Processing references", unit=" references"
    )
    try:
        while pending or active:
            done = [result for result in active if result.async_result.ready()]
            for result in done:
                active.remove(result)
                completed_bar.update(1)
                yield result.async_result.get()
            # Launch a new job if memory allows
            if (
                pending
                and len(active) < threads
                and (
                    (len(active) == 0)
                    or (max_memory_gb is None)
                    or (0 < get_used_memory_gb() < max_memory_gb)
                )
            ):
                job = pending.pop(0)
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
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    max_memory: str | None = None,
) -> Path:
    tracemalloc.start()
    start = datetime.datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S") + f"{start.microsecond / 1e6:.4f}"[1:]
    if threads == 0:
        threads = max(1, (multiprocessing.cpu_count() - 1))
    track_with_tqdm = True if threads == 1 else False
    reference_job_outputs = list()
    try:
        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            reference_to_tmp_file_paths = {
                reference: Path(
                    tempfile.NamedTemporaryFile(
                        delete=False, prefix=f"{timestamp}.{reference}.", suffix=".bam"
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

        with multiprocessing.Pool(processes=threads) as pool:
            for result in adaptive_parallel_run(
                pool,
                jobs=args_list,
                threads=threads,
                max_memory_gb=(
                    None
                    if max_memory is None
                    else (parse_memory_limit(max_memory) / threads)
                ),
                check_interval=datetime.timedelta(seconds=1),
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
        logging.debug(f"Peak memory usage in a single reference: {format_memory(peak)}")
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
        "--max-memory",
        type=str,
        default=None,
        help="Maximum memory usage (e.g. 20GB, 20000MB, or 20).",
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
        max_memory=args.max_memory,
    )
