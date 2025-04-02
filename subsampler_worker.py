#!/usr/bin/env python3

import datetime
import gc
import heapq
import logging
import random
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import pysam
import tqdm
import xxhash

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class ReferenceJobInput:
    reference: str
    reference_length: int
    read_count: int | None
    input_bam_path: str
    desired_coverage: int
    low_coverage_bases_to_prioritize: int
    ignore_n_bases_on_edges: int
    seed: int
    tmp_file_path: Path
    profile: bool
    threads: int
    tqdm_position: int
    track_with_tqdm: bool


@dataclass
class ReferenceJobOutput:
    reference: str
    temp_bam: Path
    peak_memory: int


# class ReadSpan(NamedTuple):
#     start: int
#     end: int


def format_memory(bytes_amount: int) -> str:
    """
    Format a memory size in bytes into a human-readable string (GB, MB, KB).

    Args:
        bytes_amount (int): Number of bytes.

    Returns:
        str: Human-readable memory size string.
    """
    gb = bytes_amount / 10**9
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = bytes_amount / 10**6
    if mb >= 1:
        return f"{mb:.2f} MB"
    kb = bytes_amount / 10**3
    if kb >= 1:
        return f"{kb:.2f} KB"
    return f"{bytes_amount} B"


def format_duration(delta: datetime.timedelta) -> str:
    """
    Format a time duration into a human-readable string with hours, minutes, and seconds.

    Args:
        delta (datetime.timedelta): Time duration.

    Returns:
        str: Formatted duration string.
    """
    total_seconds = delta.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    seconds = total_seconds - (hours * 3600 + minutes * 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds:.2f}s")

    return " ".join(parts)


def get_unique_positions(
    spans: list[tuple[int, int]],
    trim_n_bases: int = 0,
    reference_length: int | None = None,
) -> np.ndarray:
    """
    Compute the unique genomic positions covered by a list of read spans.

    Args:
        spans (list[tuple[int, int]]): List of (start, end) read spans.
        trim_n_bases (int): Number of bases to trim from each end of a span.
        reference_length (int | None): Length of the reference sequence.

    Returns:
        np.ndarray: Sorted array of unique positions.
    """
    return np.unique(
        np.concatenate(
            [
                np.arange(
                    (start + trim_n_bases) if start != 0 else start,
                    (end - trim_n_bases) if end != reference_length else end,
                )
                for start, end in spans
            ]
        )
    )


def get_read_hashes_to_spans_sorted_by_lower_to_higher_coverage(
    bamfile: pysam.AlignmentFile,
    reference: str,
    reference_length: int,
    coverage_cap: int,
    low_coverage_bases_to_prioritize: int,
    read_count: int | None = None,
    disable_tqdm: bool = False,
    tqdm_position: int = 0,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """
    Iterates through reads of a given reference in the BAM file, collects their spans,
    builds a coverage array across the reference, and yields read spans sorted by
    increasing coverage (prioritizing under-covered regions).

    Args:
        bamfile (pysam.AlignmentFile): Opened BAM file object.
        reference (str): The name of the reference (contig) to process.
        reference_length (int): Length of the reference sequence.
        coverage_cap (int): Maximum coverage value to consider per base. Anything equal or higher than this value is considered the same when sorting by coverage.
        read_count (int | None): Number of reads in the reference. If None, it will be counted.
        disable_tqdm (bool): If True, disables the tqdm progress bars.
        low_coverage_bases_to_prioritize (int): Number of lowest-coverage positions to use as sort priority per read.
        tqdm_position (int): Vertical position of the tqdm bar in the terminal (for multi-bar display).

    Returns:
        list: List of (read_hash, spans) sorted by coverage priority.
    """
    bamfile.seek(0)
    if read_count is None:
        read_count = bamfile.count(reference=reference)
        bamfile.seek(0)
    read_hash_to_spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    coverage_array = np.zeros(
        reference_length,
        dtype=(np.uint8 if coverage_cap < 255 else np.uint16),
    )
    coverage_limiter = max(coverage_cap, (np.iinfo(coverage_array.dtype).max - 10))
    for read in tqdm.tqdm(
        bamfile.fetch(
            reference=reference,
            multiple_iterators=(True if bamfile.threads > 1 else False),
        ),
        total=read_count,
        desc=f"Indexing {reference}{f' with {bamfile.threads} threads' if bamfile.threads > 1 else ''}",
        unit=" reads",
        position=tqdm_position,
        disable=disable_tqdm,
        leave=True,
    ):
        if not read.is_proper_pair:
            continue
        try:
            start = min(read.reference_start, read.reference_end)
            end = max(read.reference_start, read.reference_end)
        except TypeError:
            continue
        length = end - start
        if length:
            read_hash_to_spans[xxhash.xxh64(read.query_name).intdigest()].append(
                (start, end)
            )
            positions = np.arange(start, end)
            coverage_array[positions] += 1
            if coverage_array[positions].max() >= coverage_limiter:
                coverage_array = np.minimum(
                    coverage_array, coverage_cap
                )  # to avoid overflow

    coverage_array = np.minimum(coverage_array, coverage_cap)
    total_hashes = len(read_hash_to_spans)
    random_ceil = coverage_cap + (total_hashes * 10)

    def coverage_vector(spans: list[tuple[int, int]]) -> tuple[int, ...]:
        coverages = coverage_array[get_unique_positions(spans)]
        return tuple(
            heapq.nsmallest(low_coverage_bases_to_prioritize, coverages)
            + [random.randint(coverage_cap, random_ceil)]
        )

    return [
        (read_hash, spans)
        for (read_hash, spans), _ in sorted(
            (
                (item, coverage_vector(item[1]))
                for item in tqdm.tqdm(
                    read_hash_to_spans.items(),
                    total=total_hashes,
                    desc=f"Sorting read pairs by the {low_coverage_bases_to_prioritize} positions with lowest coverages for {reference}",
                    disable=disable_tqdm,
                    unit=" read pairs",
                    position=tqdm_position,
                    leave=True,
                )
            ),
            key=lambda x: x[1],
        )
    ]


def process_reference(args: ReferenceJobInput) -> ReferenceJobOutput:
    """
    Process a single reference (contig) from a BAM file:
    - Selects reads to match a target per-base coverage.
    - Prioritizes reads that cover underrepresented regions.
    - Writes selected reads to a temporary BAM file.

    Args:
        args (ReferenceJobInput): Job configuration and metadata for processing the contig.
            - reference (str): Name of the reference/contig.
            - reference_length (int): Length of the reference sequence.
            - read_count (int | None): Number of reads for this reference. If None, it will be counted.
            - input_bam_path (str): Path to the original input BAM file.
            - desired_coverage (int): Target per-base coverage to reach.
            - low_coverage_bases_to_prioritize (int): Number of lowest-coverage positions to use as sort priority per read.
            - ignore_n_bases_on_edges (int): Number of bases to ignore at the start and end of reads for coverage.
            - seed (int): Seed for reproducibility (used for shuffling reads with the same lowest coverages).
            - tmp_file_path (Path): Path to the temporary output BAM file.
            - profile (bool): Whether to collect memory profiling statistics with tracemalloc.
            - threads (int): Number of threads to use for BAM I/O operations.
            - tqdm_position (int): Position of the progress bar (used in multi-bar tqdm).
            - track_with_tqdm (bool): Whether to display tqdm progress bars.

    Returns:
        ReferenceJobOutput: Metadata for the result, including the reference name,
        path to the temporary output BAM, and peak memory used during the process.
    """
    start_time = datetime.datetime.now()
    if args.profile:
        tracemalloc.start()
    else:
        peak = 0
    logging.debug(f"Starting to process reference {args.reference}")
    disable_tqdm = not args.track_with_tqdm
    random.seed(args.seed)
    with (
        pysam.AlignmentFile(args.input_bam_path, "rb", threads=args.threads) as bamfile,
        pysam.AlignmentFile(
            args.tmp_file_path, "wb", header=bamfile.header, threads=args.threads
        ) as output_fh,
    ):
        read_count = (
            args.read_count
            if args.read_count is not None
            else bamfile.count(reference=args.reference)
        )
        if read_count == 0:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return ReferenceJobOutput(args.reference, args.tmp_file_path, peak)
        logging.info(
            f"Starting to subsample reference {args.reference} ({read_count} reads over {args.reference_length} positions)\n"
        )
        coverage_cap = (
            min(254, int(args.desired_coverage * 2))
            if args.desired_coverage <= 200
            else (args.desired_coverage * 1.5)
        )
        coverage_array = np.zeros(
            args.reference_length, dtype=(np.uint8 if coverage_cap < 255 else np.uint16)
        )
        coverage_limiter = np.iinfo(coverage_array.dtype).max - 10
        selected_read_hashes = set()
        sorted_read_hash_and_spans = (
            get_read_hashes_to_spans_sorted_by_lower_to_higher_coverage(
                bamfile=bamfile,
                reference=args.reference,
                reference_length=args.reference_length,
                coverage_cap=coverage_cap,
                read_count=read_count,
                disable_tqdm=disable_tqdm,
                low_coverage_bases_to_prioritize=args.low_coverage_bases_to_prioritize,
                tqdm_position=args.tqdm_position,
            )
        )
        for read_hash, spans in tqdm.tqdm(
            sorted_read_hash_and_spans,
            desc=(f"Selecting reads for {args.reference}"),
            unit=" reads",
            total=read_count / 2,
            disable=disable_tqdm,
            leave=True,
        ):
            positions = get_unique_positions(
                spans,
                trim_n_bases=args.ignore_n_bases_on_edges,
                reference_length=args.reference_length,
            )
            if np.any(coverage_array[positions] < args.desired_coverage):
                selected_read_hashes.add(read_hash)
                coverage_array[positions] += 1
                if coverage_array[positions].max() >= coverage_limiter:
                    coverage_array = np.minimum(coverage_array, args.desired_coverage)
        del sorted_read_hash_and_spans
        gc.collect()
        if not selected_read_hashes:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return ReferenceJobOutput(args.reference, args.tmp_file_path, peak)
        bamfile.seek(0)
        total_output_reads = 0
        for read in tqdm.tqdm(
            bamfile.fetch(
                reference=args.reference,
                multiple_iterators=(True if bamfile.threads > 1 else False),
            ),
            total=read_count,
            desc=f"Writing {args.reference}{f' with {bamfile.threads} threads' if bamfile.threads > 1 else ''}",
            disable=disable_tqdm,
            leave=True,
            unit=" reads",
            position=args.tqdm_position,
        ):
            if xxhash.xxh64(read.query_name).intdigest() in selected_read_hashes:
                output_fh.write(read)
                total_output_reads += 1
    del selected_read_hashes
    gc.collect()
    if args.profile:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logging.info(
            f"Done subsampling reference {args.reference} ({total_output_reads}/{read_count} = {total_output_reads / read_count * 100:.2f}%)"
            f" [Time: {format_duration(datetime.datetime.now() - start_time)}] "
            f" [Peak memory usage: {format_memory(peak)}]\n"
        )
    else:
        logging.info(
            f"Done subsampling reference {args.reference} ({total_output_reads}/{read_count} = {total_output_reads / read_count * 100:.2f}%)\n"
        )
    return ReferenceJobOutput(args.reference, args.tmp_file_path, peak)
