#!/usr/bin/env python3

import argparse
import datetime
import random
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pysam
import tqdm


def get_position_to_reads(bamfile: pysam.AlignmentFile, reference: str, read_count: int | None = None) -> dict[int, set]:
    bamfile.seek(0)
    if read_count is None:
        read_count = bamfile.count(reference=reference)
        bamfile.seek(0)
    position_to_reads = defaultdict(set)
    position_to_reads = defaultdict(set)
    for read in tqdm.tqdm(bamfile.fetch(reference=reference), total=read_count, desc=f"Getting all reads for {reference}"):
        if read.reference_end is not None:
            hash_read = hash(read.query_name)
            for position in range(read.reference_start, read.reference_end):
                position_to_reads[position].add(hash_read)
    return position_to_reads


def get_selected_reads(position_to_reads: dict[int, Iterable], desired_coverage: int = 50, reference: str | None = None) -> set:
    selected_reads = set()
    for position_reads in tqdm.tqdm(
        sorted(position_to_reads.values(), key=lambda x: len(x)),
        desc=f"Subsampling coverage to {desired_coverage} for each position {'in reference ' + reference if reference else ''}"
    ):
        already_selected = (position_reads & selected_reads)
        non_selected_reads = (position_reads - already_selected)
        to_select = min(
            len(non_selected_reads),
            max(0, desired_coverage - len(already_selected))
        )
        if to_select:
            selected_reads |= set(random.sample(list(non_selected_reads), to_select))
    return selected_reads


def subsample_bam(input_bam: Path, output_bam: Path, desired_coverage: int = 50, seed: int = 42) -> Path:
    tracemalloc.start()
    start = datetime.datetime.now()
    random.seed(seed)
    with pysam.AlignmentFile(input_bam, "rb") as bamfile, pysam.AlignmentFile(output_bam, "wb", header=bamfile.header) as output_fh:
        for reference in bamfile.references:
            bamfile.seek(0)
            read_count = bamfile.count(reference=reference)
            if read_count == 0:
                continue
            selected_reads = get_selected_reads(get_position_to_reads(bamfile, reference, read_count), desired_coverage, reference)
            if not selected_reads:
                continue
            bamfile.seek(0)
            for read in tqdm.tqdm(bamfile.fetch(reference=reference), total=read_count):
                if hash(read.query_name) in selected_reads:
                    output_fh.write(read)
            del selected_reads
    current, peak = tracemalloc.get_traced_memory()
    print(
        "Done subsampling.\n"
        f"It took {datetime.datetime.now() - start} time to run.\n"
        f"Current memory usage: {current / 10**6}MB;\n"
        f"Peak memory usage: {peak / 10**6}MB"
    )
    return output_bam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a BAM file to a target per-base coverage.")
    parser.add_argument("input_bam", type=Path, help="Path to the input BAM file.")
    parser.add_argument("output_bam", type=Path, help="Path to the output BAM file.")
    parser.add_argument("-c", "--coverage", type=int, default=50, help="Desired per-base coverage (default: 50).")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42).")

    args = parser.parse_args()

    subsample_bam(
        input_bam=args.input_bam,
        output_bam=args.output_bam,
        desired_coverage=args.coverage,
        seed=args.seed
    )
