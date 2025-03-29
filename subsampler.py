#!/usr/bin/env python3

import argparse
import datetime
import random
import tracemalloc
import tempfile
import multiprocessing
from pathlib import Path

import pysam
import tqdm

from subsampler_worker import ReferenceJobInput, ReferenceJobOutput, process_reference, format_memory, format_duration

def subsample_bam_parallel(input_bam: Path, output_bam: Path, desired_coverage: int = 50, seed: int = 42, threads: int = 1) -> Path:
    tracemalloc.start()
    start = datetime.datetime.now()
    if threads == 0:
        threads = max(1, (multiprocessing.cpu_count() - 1))
    track_with_tqdm = True if threads == 1 else False
    try:
        with pysam.AlignmentFile(input_bam, "rb") as bamfile:
            args_list = [
                ReferenceJobInput(
                    reference=reference,
                    input_bam_path=input_bam,
                    desired_coverage=desired_coverage,
                    seed=(seed + i),
                    tmp_file_path=Path(tempfile.NamedTemporaryFile(delete=False, suffix=f".{reference}.bam").name),
                    track_with_tqdm=track_with_tqdm,
                )
                for i, reference in enumerate(sorted(bamfile.references))
            ]
        tmp_file_paths = [reference_job_input.tmp_file_path for reference_job_input in args_list]

        with multiprocessing.Pool(processes=threads) as pool:
            reference_job_outputs: list[ReferenceJobOutput] = list(tqdm.tqdm(pool.imap_unordered(process_reference, args_list), total=len(args_list), desc="Processing references"))
            
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
        print(
            "Done subsampling.\n"
            f"It took {format_duration(datetime.datetime.now() - start)} time to run.\n"
            f"Peak memory usage: {format_memory(peak)}"
        )
        return output_bam
    except:
        print("\nInterrupted! Cleaning up temporary files...")
        for tmp_file_path in tmp_file_paths:
            if tmp_file_path.exists():
                tmp_file_path.unlink()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a BAM file to a target per-base coverage.")
    parser.add_argument("input_bam", type=Path, help="Path to the input BAM file.")
    parser.add_argument("output_bam", type=Path, help="Path to the output BAM file.")
    parser.add_argument("-c", "--coverage", type=int, default=50, help="Desired per-base coverage (default: 50).")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of parallel processes to use. 0 to use all threads (might require a lot of memory for large genomes)")

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
