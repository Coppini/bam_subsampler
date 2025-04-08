# bam_subsampler

A Python tool for subsampling BAM files to a target per-base coverage using a coverage-aware strategy. 
It aims to retain sufficient read depth at every position while minimizing data loss.

---

## Features
- **Per-base coverage control**: Attempts to ensure every position has the desired depth.
- **Reference-aware parallelism**: Automatically splits the work by contig/reference.
- **Memory usage tracking**: Reports peak memory usage per reference.
- **Optional single-thread mode**: Shows progress per reference when using a single thread.

---

## Installation

Install dependencies with `pip`:
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python subsampler.py [options] <input.bam> <output.bam>
```

### Positional arguments:
- `<input.bam>`: Path to the input BAM file
- `<output.bam>`: Path to write the subsampled BAM file

### Optional arguments:
- `-c`, `--coverage <int>`: Desired per-base coverage (default: `50`)
- `-s`, `--seed <int>`: Random seed for reproducibility (default: `42`)
- `-t`, `--threads <int>`: Number of parallel processes (default: `1`; use `0` to use all available cores)

### Example
```bash
python subsampler.py -c 30 -t 4 input.bam output.bam
```

---

## How It Works
- Reads are indexed by their positions.
- For each position, the script selects reads to reach the desired depth.
- References (contigs) are processed in parallel.
- Positions with low-coverage initally are prioritized, so that reads are selected for them first.
- Some positions might still end up with a higher coverage due to the nature of paired-end data (positions with lower coverage might require more reads to span them, increasing coverage of neighboring positions).

---

## Output
- A BAM file with reduced size and coverage close to the specified target.
- Summary printed at the end, including total time and peak memory usage.
- Peak memory usage per reference (when available).

---

## Notes
- The script uses Python's multiprocessing module and may use a significant amount of memory on large genomes.
- Progress bars are suppressed in parallel mode to avoid clutter.
- Memory is tracked per-reference when running with multiple processes.

---

## License
MIT License