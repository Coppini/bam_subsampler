#!/usr/bin/env python3

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

import pysam
import tqdm

class ReferenceJobInput(NamedTuple):
    reference: str
    input_bam_path: str
    desired_coverage: int
    seed: int
    tmp_file_path: Path

class ReferenceJobOutput(NamedTuple):
    reference: str
    temp_bam: Path

def get_position_to_reads(bamfile: pysam.AlignmentFile, reference: str, read_count: int | None = None) -> dict[int, set]:
    bamfile.seek(0)
    if read_count is None:
        read_count = bamfile.count(reference=reference)
        bamfile.seek(0)
    position_to_reads = defaultdict(set)
    for read in tqdm.tqdm(
        bamfile.fetch(reference=reference),
        total=read_count,
        desc=f"Indexing {reference}"
    ):
        if read.reference_end is not None:
            hash_read = hash(read.query_name)
            for position in range(read.reference_start, read.reference_end):
                position_to_reads[position].add(hash_read)
    return position_to_reads


def get_selected_reads(position_to_reads: dict[int, Iterable], desired_coverage: int = 50, reference: str | None = None) -> set:
    selected_reads = set()
    for position_reads in tqdm.tqdm(
        sorted(position_to_reads.values(), key=lambda x: len(x)),
        desc=("Selecting reads" if not reference else f"Selecting reads for {reference}")
    ):
        already_selected = position_reads & selected_reads
        non_selected_reads = position_reads - already_selected
        to_select = min(
            len(non_selected_reads),
            max(0, desired_coverage - len(already_selected))
        )
        if to_select:
            selected_reads |= set(random.sample(list(non_selected_reads), to_select))
    del position_reads
    return selected_reads


def process_reference(args: ReferenceJobInput) -> ReferenceJobOutput:
    reference, input_bam_path, desired_coverage, seed, tmp_file_path = args
    random.seed(seed)
    with pysam.AlignmentFile(input_bam_path, "rb") as bamfile, \
         pysam.AlignmentFile(tmp_file_path, "wb", header=bamfile.header) as output_fh:
        read_count = bamfile.count(reference=reference)
        if read_count == 0:
            return tmp_file_path
        selected_reads = get_selected_reads(
            position_to_reads=get_position_to_reads(
                bamfile=bamfile,
                reference=reference,
                read_count=read_count,
            ),
            desired_coverage=desired_coverage,
            reference=reference,
        )
        if not selected_reads:
            return tmp_file_path
        bamfile.seek(0)
        for read in tqdm.tqdm(bamfile.fetch(reference=reference), total=read_count, desc=f"Writing {reference}"):
            if hash(read.query_name) in selected_reads:
                output_fh.write(read)
        del selected_reads
    return ReferenceJobOutput(reference, tmp_file_path)
