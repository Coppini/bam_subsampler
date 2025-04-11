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
from collections import defaultdict
from enum import Enum
from pathlib import Path
from shutil import move
from typing import Iterable

import psutil
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


class AsyncProcessJob:
    def __init__(self, job_input: ContigJobInput, process: multiprocessing.Process):
        self.job_input = job_input
        self.process = process
        self.start = datetime.datetime.now()

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def get_memory_usage(self) -> int:
        try:
            return psutil.Process(self.process.pid).memory_info().rss or 0
        except:
            return 0


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_read_count(bamfile: Path, contig: str) -> int:
    bamfile.seek(0)
    read_count = bamfile.count(contig=contig)
    bamfile.seek(0)
    return read_count


class MemoryUnit(Enum):
    byte = "B"
    kilobyte = "KB"
    megabyte = "MB"
    gigabyte = "GB"
    terabyte = "TB"


def parse_memory(
    memory: str | int,
    mem_unit: MemoryUnit | str = MemoryUnit.byte,
    output_unit: MemoryUnit | str = MemoryUnit.gigabyte,
) -> float:
    mem_str = str(memory).strip().upper()
    match = re.match(r"^([\d.]+)\s*(TB|GB|MB|KB|B)?$", mem_str)
    if not match:
        raise ValueError(f"Invalid memory limit format: {mem_str}")
    value, unit = match.groups()
    value = float(value)
    unit = MemoryUnit(unit or mem_unit)
    unit_power = {
        MemoryUnit.terabyte: 4,
        MemoryUnit.gigabyte: 3,
        MemoryUnit.megabyte: 2,
        MemoryUnit.kilobyte: 1,
        MemoryUnit.byte: 0,
    }
    return value * (1024 ** (unit_power[unit] - unit_power[MemoryUnit(output_unit)]))

def get_system_available_memory_bytes() -> int:
    return psutil.virtual_memory().available

def get_system_total_memory_bytes() -> int:
    return psutil.virtual_memory().total

def process_wrapper(job: ContigJobInput, queue: multiprocessing.Queue) -> None:
    process = process_contig(job)
    queue.put((job, process))


def adaptive_parallel_run(
    jobs: list[ContigJobInput],
    threads: int,
    contigs_to_parallelize_on: int,
    max_memory_bytes: int,
    check_interval: datetime.timedelta = datetime.timedelta(seconds=1),
) -> Iterable[ContigJobOutput]:
    queue = multiprocessing.Queue()
    completed: list[ContigJobOutput] = list()
    active: list[AsyncProcessJob] = list()
    pending: list[ContigJobInput] = jobs.copy()
    peak_process_memory_usage: dict[str, int] = defaultdict(int)
    peak_memory_usage = 0
    job_input: ContigJobInput
    job_output: ContigJobOutput
    active_job: AsyncProcessJob | None
    system_total_memory = psutil.virtual_memory().total
    try:
        with tqdm.tqdm(
            total=len(jobs), desc="Processing contigs", unit=" contigs"
        ) as completed_bar:
            while pending or active:
                current, peak = tracemalloc.get_traced_memory()
                threads_in_use = max(
                    len(active),
                    sum(async_job.job_input.threads for async_job in active),
                )
                available_threads = threads - threads_in_use
                available_slots = min(
                    len(pending), contigs_to_parallelize_on - len(active)
                )
                contig_to_memory_usage = {
                    process.job_input.contig: process.get_memory_usage()
                    for process in active
                }
                for contig, memory_usage in contig_to_memory_usage.items():
                    peak_process_memory_usage[contig] = max(
                        peak_process_memory_usage[contig], memory_usage
                    )
                current_max_memory = (
                    max(contig_to_memory_usage.values())
                    if contig_to_memory_usage
                    else 0
                )
                current_memory_used = current + sum(contig_to_memory_usage.values())
                peak_memory_usage = max(peak_memory_usage, current_memory_used)
                available_memory = max_memory_bytes - current_memory_used
                system_available_memory = get_system_available_memory_bytes()
                # Check and clean up finished jobs
                for job in active:
                    if not job.is_alive():
                        job.process.join()
                        active.remove(job)
                while not queue.empty():
                    job_input, job_output = queue.get()
                    active_job = None
                    for active_job in active:
                        if active_job.job_input == job_input:
                            active.remove(active_job)
                            break
                    duration = job_output.start - job_output.end
                    logging.info(
                        f"Job finished for contig {job_input.contig} after {format_duration(duration)},"
                        f" with peak memory usage of {format_memory(max(peak_process_memory_usage[job_input.contig], job_output.peak_memory))}"
                    )
                    completed_bar.update(1)
                    yield job_output
                    completed.append(job_output)
                    gc.collect()
                if (
                    pending
                    and available_threads
                    and available_slots
                    and (
                        len(active) == 0
                        or (
                            available_memory > current_max_memory
                            and system_available_memory > current_max_memory
                        )
                    )
                ):
                    job: ContigJobInput = pending.pop(0)
                    job.start = datetime.datetime.now()
                    if available_threads > available_slots:
                        job.threads = max(
                            2, job.threads, available_threads // available_slots
                        )
                    logging.info(
                        f"Starting job for contig {job.contig}"
                        f" (contig length: {job.contig_length}; total reads: {job.read_count})"
                        f" [Current memory usage: {format_memory(current_memory_used)}]"
                    )
                    proc = multiprocessing.Process(
                        target=process_wrapper, args=(job, queue)
                    )
                    proc.start()
                    active.append(AsyncProcessJob(job_input=job, process=proc))
                elif current_memory_used > max_memory_bytes and len(active) > 1:
                    last_job = sorted(active, key=lambda x: x.start)[-1]
                    logging.warning(
                        f"Current memory is above {format_memory(max_memory_bytes)}."
                        f" Killing the most recently started process for contig {last_job.job_input.contig} (contig length: {last_job.job_input.contig_length}; total reads: {last_job.job_input.read_count})."
                        f" This should free {format_memory(last_job.get_memory_usage())} of memory."
                    )
                    pending.insert(0, last_job.job_input)
                    last_job.process.terminate()
                    last_job.process.join()
                    active.remove(last_job)
                    gc.collect()
                    logging.warning(
                        f"Current memory usage: {format_memory(current_memory_used)}"
                    )
                    time.sleep(check_interval.total_seconds())
                else:
                    time.sleep(check_interval.total_seconds())
                gc.collect()
                completed_bar.set_description(
                    f"Processing contigs (completed: {len(completed)}; active: {len(active)}; pending: {len(pending)}"
                )
    except Exception as exc:
        logging.error(f"Error: {exc}")
        raise
    finally:
        memory_usages = {
            contig: format_memory(memory)
            for contig, memory in sorted(
                contig_to_memory_usage.items(), key=lambda x: x[1], reverse=True
            )[:3]
        }
        logging.warning(f"Peak memory usage: {peak_memory_usage} ({memory_usages=})")
        for process_job in active:
            process_job.process.terminate()
        for process_job in active:
            process_job.process.join()
        queue.close()


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
    max_memory: str | None = None,
    check_interval: datetime.timedelta = datetime.timedelta(seconds=1),
    verbose: bool = False,
) -> Path:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
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
    contig_job_outputs = list()
    contig_to_tmp_file_paths = dict()
    try:
        with pysam.AlignmentFile(input_bam, "rb", threads=threads) as bamfile:
            contigs = bamfile.references
            logging.debug(f"{len(contigs)} contigs found in the BAM file.")
            contig_to_lengths: dict[str, int] = {
                contig: bamfile.get_reference_length(contig) for contig in contigs
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
            args_list = [
                ContigJobInput(
                    contig=contig,
                    contig_length=contig_to_lengths[contig],
                    read_count=contig_to_read_counts[contig],
                    input_bam_path=input_bam,
                    desired_coverage=desired_coverage,
                    coverage_cap=coverage_cap,
                    large_coverage_matrix=large_coverage_matrix,
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
                        if (track_with_tqdm and contigs_to_parallelize_on > 1)
                        else None
                    ),
                    start=None,
                )
                for i, contig in enumerate(
                    sorted(
                        contigs,
                        key=lambda ref: (
                            contig_to_read_counts[ref],
                            contig_to_lengths[ref],
                            ref,
                        ),
                        reverse=False,
                    )
                )
            ]

        if contigs_to_parallelize_on == 1:
            for job in args_list:
                contig_job_outputs.append(process_contig(job))
        else:
            for result in adaptive_parallel_run(
                jobs=args_list,
                threads=threads,
                contigs_to_parallelize_on=contigs_to_parallelize_on,
                max_memory_bytes=(
                    psutil.virtual_memory().total
                    if max_memory is None
                    else (
                        parse_memory(
                            max_memory,
                            mem_unit=MemoryUnit.gigabyte,
                            output_unit=MemoryUnit.byte,
                        )
                    )
                ),
                check_interval=check_interval,
            ):
                contig_job_outputs.append(result)

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
        peak = main_peak + max(process.peak_memory for process in contig_job_outputs)
        logging.info("Done subsampling.")
        logging.info(
            f"It took {format_duration(datetime.datetime.now() - start)} time to run."
        )
        logging.debug(f"Peak memory usage in a SINGLE contig: {format_memory(peak)}")
        return output_bam
    except:
        logging.warning("Interrupted! Cleaning up temporary files...")
        current, main_peak = tracemalloc.get_traced_memory()
        peak = main_peak + sum(process.peak_memory for process in contig_job_outputs)
        logging.warning(
            f"It took {format_duration(datetime.datetime.now() - start)} time to run before the exception."
        )
        logging.warning(f"Peak memory usage in a single contig: {format_memory(peak)}")
        for tmp_file_path in contig_to_tmp_file_paths.values():
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
        "-l",
        "--coverage-cap",
        type=int,
        default=0,
        help=(
            "Maximum coverage value to consider when finding low coverage positions."
            " Anything equal or higher than this value is considered the same when sorting by coverage."
            " (default: 0 for automatic detection based on desired coverage.)"
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
        coverage_cap=args.coverage_cap,
        large_coverage_matrix=True,  # args.large_coverage_matrix,
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
