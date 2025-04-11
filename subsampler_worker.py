#!/usr/bin/env python3

import datetime
import gc
import logging
import random
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, TypeVar

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
class ContigJobInput:
    contig: str
    contig_length: int
    read_count: int | None
    input_bam_path: str
    desired_coverage: int
    coverage_cap: int
    large_coverage_matrix: bool
    low_coverage_bases_to_prioritize: int
    ignore_n_bases_on_edges: int
    seed: int
    tmp_file_path: Path
    profile: bool
    threads: int
    tqdm_position: int
    track_with_tqdm: bool
    start: datetime.datetime


@dataclass
class ContigJobOutput:
    contig: str
    start: datetime.datetime
    end: datetime.datetime
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
    tb = bytes_amount / (1024**4)
    if tb >= 1:
        return f"{tb:.2f} TB"
    gb = bytes_amount / (1024**3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = bytes_amount / (1024**2)
    if mb >= 1:
        return f"{mb:.2f} MB"
    kb = bytes_amount / (1024)
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


def get_positions(
    spans: np.ndarray,
    trim_n_bases: int = 0,
    contig_length: int | None = None,
) -> np.ndarray:
    """
    Compute the unique genomic positions covered by a list of read slices.

    Args:
        slices (list[slice]): List of read slices (start, end).
        trim_n_bases (int): Number of bases to trim from each end of a slice.
        contig_length (int | None): Length of the contig sequence.

    Returns:
        np.ndarray: Sorted array of unique positions.
    """
    if trim_n_bases:
        return np.concatenate(
            [
                np.arange(
                    ((start + trim_n_bases) if start != 0 else start),
                    ((stop - trim_n_bases) if stop != contig_length else stop),
                )
                for (start, stop) in spans
            ]
        )
    return np.concatenate(
        [np.arange(spans[0][0], spans[0][1]), np.arange(spans[1][0], spans[1][1])]
    )


def smallest_possible_uint_dtype(min_value: int) -> np.dtype:
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if np.iinfo(dtype).max >= min_value:
            return dtype
    raise ValueError(f"No suitable unsigned integer dtype for value {min_value}")


K = TypeVar("K")
V = TypeVar("V")


def pop_items_from_dictionary(
    d: dict[K, V], random_order: bool = False
) -> Generator[tuple[K, V], None, None]:
    for key in (
        random.sample(list(d.keys()), len(d)) if random_order else list(d.keys())
    ):
        yield key, d.pop(key)


def get_read_hashes_to_spans_by_lower_to_higher_coverage(
    bamfile: pysam.AlignmentFile,
    contig: str,
    contig_length: int,
    desired_coverage: int,
    coverage_cap: int = 0,
    low_coverage_bases_to_prioritize: int = True,
    read_count: int = 0,
    disable_tqdm: bool = False,
    tqdm_position: int = 0,
) -> dict[bytes, np.ndarray]:
    """
    Iterates through reads of a given contig in the BAM file, collects their slices,
    builds a coverage array across the contig, and yields read slices sorted by
    increasing coverage (prioritizing under-covered regions).

    Args:
        bamfile (pysam.AlignmentFile): Opened BAM file object.
        contig (str): The name of the contig/reference to process.
        contig_length (int): Length of the contig sequence.
        coverage_cap (int): Maximum coverage value to consider per base. Anything equal or higher than this value is considered the same when sorting by coverage.
        read_count (int): Number of reads in the contig. If 0, it will be counted.
        disable_tqdm (bool): If True, disables the tqdm progress bars.
        low_coverage_bases_to_prioritize (int): Number of lowest-coverage positions to use as sort priority per read.
        tqdm_position (int): Vertical position of the tqdm bar in the terminal (for multi-bar display).

    Returns:
        hashes_to_spans (dict[bytes, np.ndarray]): A dictionary of {read_hash: spans} sorted by coverage priority, from lowest coverage to highest.
                              spans (np.ndarray): has shape=(2,2), as [[start_read1, stop_read1], [start_read2, stop_read2]]
    """
    bamfile.seek(0)
    if not read_count:
        read_count = bamfile.count(contig=contig)
        bamfile.seek(0)
    if not coverage_cap:
        coverage_cap = desired_coverage + 1
    position_array_dtype = smallest_possible_uint_dtype(contig_length)
    coverage_array_dtype = smallest_possible_uint_dtype(
        max(np.iinfo(np.uint16).max, coverage_cap + 10, coverage_cap * 2)
    )
    hash_to_spans: dict[bytes, np.ndarray] = dict()
    coverage_array_dtype_max = np.iinfo(coverage_array_dtype).max
    coverage_limiter = coverage_array_dtype_max - 10
    coverage_array = np.zeros(
        contig_length,
        dtype=coverage_array_dtype,
    )
    for read in tqdm.tqdm(
        bamfile.fetch(
            contig=contig,
            multiple_iterators=(True if bamfile.threads > 1 else False),
        ),
        total=read_count,
        desc=f"Indexing {contig}{f' with {bamfile.threads} threads' if bamfile.threads > 1 else ''}",
        unit=" reads",
        position=tqdm_position,
        disable=disable_tqdm,
        leave=True,
    ):
        if not read.is_proper_pair:
            continue
        start = read.reference_start
        stop = read.reference_end
        if stop < start:
            start, stop = stop, start
        elif stop == start:
            continue
        read_hash = xxhash.xxh64(read.query_name).digest()
        span = hash_to_spans.get(read_hash)
        if span is None:
            hash_to_spans[read_hash] = np.array(
                [[start, stop], [0, 0]], dtype=position_array_dtype
            )
        else:
            span[1] = [start, stop]
        span_slice = slice(start, stop)
        coverage_array[span_slice] += 1
        if coverage_array[span_slice].max() >= coverage_limiter:
            logging.debug(
                f"Coverage above max value of {coverage_array_dtype_max} will be capped at read spanning {contig}:{span_slice.start}-{span_slice.stop}",
                extra={"region": contig},
            )
            coverage_array = np.minimum(coverage_array, coverage_cap)
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 3 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    gc.collect()
    coverage_array = np.minimum(coverage_array, coverage_cap)
    minimum_dtype_required = smallest_possible_uint_dtype(coverage_array.max())
    if (
        coverage_array_dtype_max
        > np.iinfo(minimum_dtype_required).max
        >= coverage_array.max()
    ):
        coverage_array = coverage_array.astype(minimum_dtype_required)
        coverage_array_dtype = minimum_dtype_required
        coverage_array_dtype_max = np.iinfo(minimum_dtype_required).max
    logging.info(
        f"Using {coverage_array_dtype} with max {coverage_array_dtype_max}",
        extra={"region": contig},
    )
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 4 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    hash_to_spans = dict(
        tqdm.tqdm(
            pop_items_from_dictionary(hash_to_spans, random_order=True),
            total=len(hash_to_spans),
            desc="Shuffling read hashes to remove position biases",
            disable=disable_tqdm,
            unit=" hashes",
            position=tqdm_position,
            leave=True,
        )
    )
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 5 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    gc.collect()

    random_part_length = int(
        np.ceil(
            np.log(coverage_cap + len(hash_to_spans)) / np.log(coverage_array_dtype_max)
        )
    )
    current, peak = tracemalloc.get_traced_memory()
    logging.info(
        f" 6 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    with tqdm.tqdm(
        total=len(hash_to_spans),
        desc=f"Sorting read pairs by the {low_coverage_bases_to_prioritize} positions with lowest coverages for {contig}",
        disable=disable_tqdm,
        unit=" read pairs",
        position=tqdm_position,
        leave=True,
    ) as sorting_bar:

        def coverage_vector(spans: np.ndarray) -> bytes:
            coverages = np.concatenate(
                [
                    coverage_array[spans[0][0] : spans[0][1]],
                    coverage_array[spans[1][0] : spans[1][1]],
                ]
            )
            try:
                coverages = np.partition(coverages, low_coverage_bases_to_prioritize)[
                    :low_coverage_bases_to_prioritize
                ]
            except ValueError:
                coverages = np.pad(
                    coverages,
                    (0, (low_coverage_bases_to_prioritize - len(coverages))),
                    mode="constant",
                    constant_values=coverage_array_dtype_max,
                )
            coverages.sort()

            sorting_bar.update(1)

            if coverages[0] <= desired_coverage:
                return coverages.tobytes()
            return np.concatenate(
                [
                    coverages,
                    np.array(
                        [
                            random.randint(0, coverage_array_dtype_max)
                            for _ in range(random_part_length)
                        ],
                        dtype=coverage_array_dtype,
                    ),
                ]
            ).tobytes()

        current, peak = tracemalloc.get_traced_memory()
        logging.debug(
            f" 7 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
            extra={"region": contig},
        )
        return dict(
            sorted(
                pop_items_from_dictionary(hash_to_spans),
                key=lambda rh2s: coverage_vector(rh2s[1]),
            )
        )


def select_reads(
    sorted_read_hash_and_spans: dict[bytes, np.ndarray],
    contig: str,
    contig_length: int,
    desired_coverage: int,
    coverage_cap: int,
    ignore_n_bases_on_edges: int = 0,
    large_coverage_matrix: bool = True,
    disable_tqdm: bool = False,
    tqdm_position: int = 0,
) -> Generator[bytes, None, None]:
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 10 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    coverage_array = np.zeros(
        contig_length,
        dtype=(
            np.uint16
            if (large_coverage_matrix or coverage_cap >= (np.iinfo(np.uint8).max - 10))
            else np.uint8
        ),
    )
    coverage_limiter = np.iinfo(coverage_array.dtype).max - 10
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 11 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    for read_hash, spans in tqdm.tqdm(
        pop_items_from_dictionary(sorted_read_hash_and_spans, random_order=False),
        total=len(sorted_read_hash_and_spans),
        desc=(f"Selecting reads for {contig}"),
        unit=" reads",
        disable=disable_tqdm,
        leave=True,
        position=tqdm_position,
    ):
        if np.any(
            coverage_array[spans[0][0] : spans[0][1]] <= desired_coverage
        ) or np.any(coverage_array[spans[1][0] : spans[1][1]] <= desired_coverage):
            yield read_hash
            positions = get_positions(
                spans,
                trim_n_bases=ignore_n_bases_on_edges,
                contig_length=contig_length,
            )
            unique_positions = np.unique(positions)
            coverage_array[unique_positions] += 1
            if coverage_array[unique_positions].max() >= coverage_limiter:
                coverage_array = np.minimum(coverage_array, coverage_cap)
        current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 12 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 13 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": contig},
    )


def process_contig(args: ContigJobInput) -> ContigJobOutput:
    """
    Process a single contig/reference from a BAM file:
    - Selects reads to match a target per-base coverage.
    - Prioritizes reads that cover underrepresented regions.
    - Writes selected reads to a temporary BAM file.

    Args:
        args (ContigJobInput): Job configuration and metadata for processing the contig.
            - contig (str): Name of the contig/reference.
            - contig_length (int): Length of the contig sequence.
            - read_count (int | None): Number of reads for this contig. If None, it will be counted.
            - input_bam_path (str): Path to the original input BAM file.
            - desired_coverage (int): Target per-base coverage to reach.
            - coverage_cap (int): Maximum coverage value to consider per base. Anything equal or higher than this value is considered the same when sorting by coverage. 0 for automatic.
            - large_coverage_matrix (bool): If True, forces the use of a large coverage matrix, which uses more RAM, but can be faster. If False, selection depends on coverage cap.
            - low_coverage_bases_to_prioritize (int): Number of lowest-coverage positions to use as sort priority per read.
            - ignore_n_bases_on_edges (int): Number of bases to ignore at the start and end of reads for coverage.
            - seed (int): Seed for reproducibility (used for shuffling reads with the same lowest coverages).
            - tmp_file_path (Path): Path to the temporary output BAM file.
            - profile (bool): Whether to collect memory profiling statistics with tracemalloc.
            - threads (int): Number of threads to use for BAM I/O operations.
            - tqdm_position (int): Position of the progress bar (used in multi-bar tqdm).
            - track_with_tqdm (bool): Whether to display tqdm progress bars.

    Returns:
        ContigJobOutput: Metadata for the result, including the contig name,
        path to the temporary output BAM, and peak memory used during the process.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] [%(region)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.start = datetime.datetime.now()
    if args.profile:
        tracemalloc.start()
    else:
        peak = 0
    logging.debug(
        f"Starting to process contig {args.contig}", extra={"region": args.contig}
    )
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
            else bamfile.count(contig=args.contig)
        )
        if read_count == 0:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return ContigJobOutput(
                contig=args.contig,
                start=args.start,
                end=datetime.datetime.now(),
                temp_bam=args.tmp_file_path,
                peak_memory=peak,
            )
        logging.info(
            f"Starting to subsample contig {args.contig} ({read_count} reads over {args.contig_length} positions)",
            extra={"region": args.contig},
        )
        if args.coverage_cap == 0:
            args.coverage_cap = args.desired_coverage * 2
        sorted_read_hash_and_spans = (
            get_read_hashes_to_spans_by_lower_to_higher_coverage(
                bamfile=bamfile,
                contig=args.contig,
                contig_length=args.contig_length,
                desired_coverage=args.desired_coverage,
                coverage_cap=args.coverage_cap,
                low_coverage_bases_to_prioritize=args.low_coverage_bases_to_prioritize,
                read_count=read_count,
                disable_tqdm=disable_tqdm,
                tqdm_position=args.tqdm_position,
            )
        )
        current, peak = tracemalloc.get_traced_memory()
        logging.debug(
            f" 8 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
            extra={"region": args.contig},
        )
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        logging.debug(
            f" 9 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
            extra={"region": args.contig},
        )
        selected_read_hashes = set(
            select_reads(
                sorted_read_hash_and_spans,
                contig=args.contig,
                contig_length=args.contig_length,
                desired_coverage=args.desired_coverage,
                coverage_cap=args.coverage_cap,
                ignore_n_bases_on_edges=args.ignore_n_bases_on_edges,
                large_coverage_matrix=args.large_coverage_matrix,
                disable_tqdm=disable_tqdm,
                tqdm_position=args.tqdm_position,
            )
        )
        current, peak = tracemalloc.get_traced_memory()
        logging.debug(
            f" 14 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
            extra={"region": args.contig},
        )
        del sorted_read_hash_and_spans
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        logging.debug(
            f" 15 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
            extra={"region": args.contig},
        )
        if not selected_read_hashes:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return ContigJobOutput(
                contig=args.contig,
                start=args.start,
                end=datetime.datetime.now(),
                temp_bam=args.tmp_file_path,
                peak_memory=peak,
            )
        bamfile.seek(0)
        total_output_reads = 0
        for read in tqdm.tqdm(
            bamfile.fetch(
                contig=args.contig,
                multiple_iterators=(True if bamfile.threads > 1 else False),
            ),
            total=read_count,
            desc=f"Writing {args.contig}{f' with {bamfile.threads} threads' if bamfile.threads > 1 else ''}",
            disable=disable_tqdm,
            leave=True,
            unit=" reads",
            position=args.tqdm_position,
        ):
            if xxhash.xxh64(read.query_name).digest() in selected_read_hashes:
                output_fh.write(read)
                total_output_reads += 1
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 16 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": args.contig},
    )
    del selected_read_hashes
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    logging.debug(
        f" 17 - [Memory usage: peak={format_memory(peak)}; current={format_memory(current)}]",
        extra={"region": args.contig},
    )
    if args.profile:
        current, peak = tracemalloc.get_traced_memory()
        logging.info(
            f"Done subsampling contig {args.contig} ({total_output_reads}/{read_count} = {total_output_reads / read_count * 100:.2f}%)"
            f" [Time: {format_duration(datetime.datetime.now() - args.start)}] "
            f" [Peak memory usage: {format_memory(peak)}]\n",
            extra={"region": args.contig},
        )
        tracemalloc.stop()
    else:
        logging.info(
            f"Done subsampling contig {args.contig} ({total_output_reads}/{read_count} = {total_output_reads / read_count * 100:.2f}%)\n",
            extra={"region": args.contig},
        )
    return ContigJobOutput(
        contig=args.contig,
        start=args.start,
        end=datetime.datetime.now(),
        temp_bam=args.tmp_file_path,
        peak_memory=peak,
    )
