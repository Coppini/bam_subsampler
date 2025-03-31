#!/usr/bin/env python3

import datetime
import gc
import logging
import random
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

import pysam
import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


class ReferenceJobInput(NamedTuple):
    reference: str
    reference_length: int
    input_bam_path: str
    desired_coverage: int
    seed: int
    tmp_file_path: Path
    track_with_tqdm: bool


class ReferenceJobOutput(NamedTuple):
    reference: str
    temp_bam: Path
    peak_memory: int


def format_memory(bytes_amount: int) -> str:
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


def get_position_to_reads(
    bamfile: pysam.AlignmentFile,
    reference: str,
    read_count: int | None = None,
    disable_tqdm: bool = False,
) -> dict[int, set]:
    bamfile.seek(0)
    if read_count is None:
        read_count = bamfile.count(reference=reference)
        bamfile.seek(0)
    position_to_reads = defaultdict(set)
    for read in tqdm.tqdm(
        bamfile.fetch(reference=reference),
        total=read_count,
        desc=f"Indexing {reference}",
        disable=disable_tqdm,
        unit=" reads checked",
    ):
        if read.is_proper_pair:
            hash_read = hash(read.query_name)
            for position in range(read.reference_start, read.reference_end):
                position_to_reads[position].add(hash_read)
    return position_to_reads


def get_selected_reads(
    sorted_position_reads: Iterable,
    desired_coverage: int,
    reference: str | None = None,
    disable_tqdm: bool = False,
) -> set:
    selected_reads = set()
    for position_reads in tqdm.tqdm(
        sorted_position_reads,
        desc=(
            "Selecting reads" if not reference else f"Selecting reads for {reference}"
        ),
        unit=" positions analyzed",
        disable=disable_tqdm,
    ):
        already_selected = position_reads & selected_reads
        non_selected_reads = position_reads - already_selected
        to_select = min(
            len(non_selected_reads), max(0, desired_coverage - len(already_selected))
        )
        if to_select:
            selected_reads |= set(random.sample(list(non_selected_reads), to_select))
    del position_reads
    return selected_reads


def process_reference(args: ReferenceJobInput) -> ReferenceJobOutput:
    start = datetime.datetime.now()
    tracemalloc.start()
    disable_tqdm = not args.track_with_tqdm
    random.seed(args.seed)
    with (
        pysam.AlignmentFile(args.input_bam_path, "rb") as bamfile,
        pysam.AlignmentFile(
            args.tmp_file_path, "wb", header=bamfile.header
        ) as output_fh,
    ):
        read_count = bamfile.count(reference=args.reference)
        if read_count == 0:
            return args.tmp_file_path
        logging.info(
            f"Starting to subsample reference {args.reference} ({read_count} reads over {args.reference_length} positions)"
        )
        selected_reads = get_selected_reads(
            sorted_position_reads=sorted(
                get_position_to_reads(
                    bamfile=bamfile,
                    reference=args.reference,
                    read_count=read_count,
                    disable_tqdm=disable_tqdm,
                ).values(),
                key=lambda x: len(x),
            ),
            desired_coverage=args.desired_coverage,
            reference=args.reference,
            disable_tqdm=disable_tqdm,
        )
        if not selected_reads:
            return args.tmp_file_path
        bamfile.seek(0)
        total_output_reads = 0
        for read in tqdm.tqdm(
            bamfile.fetch(reference=args.reference),
            total=read_count,
            desc=f"Writing {args.reference}",
            disable=disable_tqdm,
            unit=" reads written",
        ):
            if hash(read.query_name) in selected_reads:
                output_fh.write(read)
                total_output_reads += 1
    del selected_reads
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logging.info(
        f"Done subsampling reference {args.reference} ({total_output_reads}/{read_count} = {total_output_reads / read_count * 100:.2f}%) "
        f"[Time: {format_duration(datetime.datetime.now() - start)}] "
        f"[Peak memory usage: {format_memory(peak)}]"
    )
    return ReferenceJobOutput(args.reference, args.tmp_file_path, peak)
