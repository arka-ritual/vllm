"""
Benchmark script for comparing baseline vs lookahead performance.

This script measures the timing overhead of the lookahead feature when it's
configured to never trigger (threshold=0), providing a worst-case estimate
of the performance impact.

Usage:
    python tests/lookahead/benchmark_lookahead.py
    python tests/lookahead/benchmark_lookahead.py --num-warmup 3 --num-runs 10
    python tests/lookahead/benchmark_lookahead.py --prompts-file custom_prompts.json
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    num_prompts: int
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    time_per_prompt_ms: float
    individual_times: list[float]


def get_default_prompts() -> list[str]:
    """Return a diverse set of prompts for benchmarking."""
    return [
        # Math/logic prompts
        "The square root of 144 is",
        "Calculate the sum of 1 + 2 + 3 + ... + 100. The answer is",
        "What is 15 factorial? The result is",

        # Code prompts
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    return",
        "class LinkedList:\n    def __init__(self):\n        self.head = None\n    \n    def append(self, value):",

        # Natural language prompts
        "The capital of France is",
        "In the year 1969, humans first",
        "The theory of relativity was developed by",
        "The largest planet in our solar system is",

        # Longer prompts
        "Write a Python function that takes a list of integers and returns the second largest element. Here is the implementation:",
        "Explain the difference between a stack and a queue data structure. A stack is",
        "The quick brown fox jumps over the lazy dog. This sentence is famous because it",
    ]


def run_baseline_benchmark(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> BenchmarkResult:
    """Run baseline benchmark without lookahead."""

    print(f"\n{'='*60}")
    print("BASELINE BENCHMARK (no lookahead)")
    print(f"{'='*60}")

    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Timed runs
    print(f"Running {num_runs} timed iterations...")
    times = []
    total_tokens = 0

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        # Count tokens on first run
        if i == 0:
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    tokens_per_second = (total_tokens * num_runs) / sum(times)
    time_per_prompt = (avg_time / len(prompts)) * 1000  # ms

    return BenchmarkResult(
        name="baseline",
        num_prompts=len(prompts),
        total_tokens_generated=total_tokens,
        total_time_seconds=avg_time,
        tokens_per_second=tokens_per_second,
        time_per_prompt_ms=time_per_prompt,
        individual_times=times,
    )


def run_lookahead_benchmark(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    lookahead_tokens: str = " the",  # Common token that won't match
    lookahead_threshold: float = 0.0,  # Never triggers
    num_warmup: int = 2,
    num_runs: int = 5,
) -> BenchmarkResult:
    """Run lookahead benchmark with threshold=0 (never triggers)."""

    print(f"\n{'='*60}")
    print(f"LOOKAHEAD BENCHMARK (threshold={lookahead_threshold})")
    print(f"{'='*60}")

    # Convert prompts to LookaheadPrompts
    lookahead_prompts = [
        LookaheadPrompt(
            prompt=p,
            lookahead_tokens=lookahead_tokens,
            lookahead_threshold=lookahead_threshold,
        )
        for p in prompts
    ]

    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        _ = llm.generate(lookahead_prompts, sampling_params)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Timed runs
    print(f"Running {num_runs} timed iterations...")
    times = []
    total_tokens = 0

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate(lookahead_prompts, sampling_params)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        # Count tokens on first run
        if i == 0:
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    tokens_per_second = (total_tokens * num_runs) / sum(times)
    time_per_prompt = (avg_time / len(prompts)) * 1000  # ms

    return BenchmarkResult(
        name="lookahead",
        num_prompts=len(prompts),
        total_tokens_generated=total_tokens,
        total_time_seconds=avg_time,
        tokens_per_second=tokens_per_second,
        time_per_prompt_ms=time_per_prompt,
        individual_times=times,
    )


def run_lookahead_varying_tokens_benchmark(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    lookahead_lengths: list[int] = [1, 5, 10, 20],
    num_warmup: int = 1,
    num_runs: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark with varying lookahead token lengths."""

    print(f"\n{'='*60}")
    print("LOOKAHEAD BENCHMARK - VARYING TOKEN LENGTHS")
    print(f"{'='*60}")

    results = []

    for length in lookahead_lengths:
        # Create lookahead tokens of specified length
        lookahead_tokens = " token" * length

        print(f"\n--- Lookahead length: {length} tokens ---")

        lookahead_prompts = [
            LookaheadPrompt(
                prompt=p,
                lookahead_tokens=lookahead_tokens,
                lookahead_threshold=0.0,  # Never triggers
            )
            for p in prompts
        ]

        # Warmup
        for _ in range(num_warmup):
            _ = llm.generate(lookahead_prompts, sampling_params)

        # Timed runs
        times = []
        total_tokens = 0

        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = llm.generate(lookahead_prompts, sampling_params)

            torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)

            if i == 0:
                total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

            print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")

        avg_time = sum(times) / len(times)
        tokens_per_second = (total_tokens * num_runs) / sum(times)
        time_per_prompt = (avg_time / len(prompts)) * 1000

        results.append(BenchmarkResult(
            name=f"lookahead_{length}_tokens",
            num_prompts=len(prompts),
            total_tokens_generated=total_tokens,
            total_time_seconds=avg_time,
            tokens_per_second=tokens_per_second,
            time_per_prompt_ms=time_per_prompt,
            individual_times=times,
        ))

    return results


def print_comparison(baseline: BenchmarkResult, lookahead: BenchmarkResult):
    """Print comparison between baseline and lookahead results."""

    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    overhead_pct = ((lookahead.total_time_seconds - baseline.total_time_seconds)
                    / baseline.total_time_seconds) * 100

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Lookahead':>15} {'Overhead':>12}")
    print("-" * 72)

    print(f"{'Avg total time (s)':<30} {baseline.total_time_seconds:>15.3f} "
          f"{lookahead.total_time_seconds:>15.3f} {overhead_pct:>+11.1f}%")

    print(f"{'Tokens/second':<30} {baseline.tokens_per_second:>15.1f} "
          f"{lookahead.tokens_per_second:>15.1f}")

    print(f"{'Time per prompt (ms)':<30} {baseline.time_per_prompt_ms:>15.2f} "
          f"{lookahead.time_per_prompt_ms:>15.2f}")

    print(f"{'Total tokens generated':<30} {baseline.total_tokens_generated:>15} "
          f"{lookahead.total_tokens_generated:>15}")

    # Variance analysis
    import statistics
    baseline_std = statistics.stdev(baseline.individual_times) if len(baseline.individual_times) > 1 else 0
    lookahead_std = statistics.stdev(lookahead.individual_times) if len(lookahead.individual_times) > 1 else 0

    print(f"\n{'Timing variance (std dev)':<30} {baseline_std:>15.4f} {lookahead_std:>15.4f}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: Lookahead overhead = {overhead_pct:+.2f}%")
    if abs(overhead_pct) < 5:
        print("Result: Negligible overhead (<5%)")
    elif overhead_pct > 0:
        print(f"Result: {overhead_pct:.1f}% slowdown with lookahead enabled")
    else:
        print(f"Result: {-overhead_pct:.1f}% speedup (likely within noise)")
    print(f"{'='*60}")


def print_varying_tokens_comparison(baseline: BenchmarkResult, results: list[BenchmarkResult]):
    """Print comparison for varying lookahead token lengths."""

    print(f"\n{'='*60}")
    print("LOOKAHEAD TOKEN LENGTH SCALING")
    print(f"{'='*60}")

    print(f"\n{'Configuration':<25} {'Avg Time (s)':>12} {'Tokens/s':>12} {'Overhead':>10}")
    print("-" * 60)

    print(f"{'Baseline':<25} {baseline.total_time_seconds:>12.3f} "
          f"{baseline.tokens_per_second:>12.1f} {'---':>10}")

    for result in results:
        overhead = ((result.total_time_seconds - baseline.total_time_seconds)
                    / baseline.total_time_seconds) * 100
        print(f"{result.name:<25} {result.total_time_seconds:>12.3f} "
              f"{result.tokens_per_second:>12.1f} {overhead:>+9.1f}%")


def save_results(
    baseline: BenchmarkResult,
    lookahead: BenchmarkResult,
    varying_results: Optional[list[BenchmarkResult]],
    output_path: Path,
):
    """Save benchmark results to JSON file."""

    data = {
        "baseline": {
            "name": baseline.name,
            "num_prompts": baseline.num_prompts,
            "total_tokens_generated": baseline.total_tokens_generated,
            "total_time_seconds": baseline.total_time_seconds,
            "tokens_per_second": baseline.tokens_per_second,
            "time_per_prompt_ms": baseline.time_per_prompt_ms,
            "individual_times": baseline.individual_times,
        },
        "lookahead": {
            "name": lookahead.name,
            "num_prompts": lookahead.num_prompts,
            "total_tokens_generated": lookahead.total_tokens_generated,
            "total_time_seconds": lookahead.total_time_seconds,
            "tokens_per_second": lookahead.tokens_per_second,
            "time_per_prompt_ms": lookahead.time_per_prompt_ms,
            "individual_times": lookahead.individual_times,
        },
        "overhead_percent": ((lookahead.total_time_seconds - baseline.total_time_seconds)
                             / baseline.total_time_seconds) * 100,
    }

    if varying_results:
        data["varying_token_lengths"] = [
            {
                "name": r.name,
                "total_time_seconds": r.total_time_seconds,
                "tokens_per_second": r.tokens_per_second,
                "overhead_percent": ((r.total_time_seconds - baseline.total_time_seconds)
                                     / baseline.total_time_seconds) * 100,
            }
            for r in varying_results
        ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark lookahead vs baseline performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/paperspace/Qwen3-0.6B",
        help="Path to the model",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with custom prompts (list of strings)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--skip-varying-lengths",
        action="store_true",
        help="Skip the varying lookahead token length benchmark",
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = get_default_prompts()
        print(f"Using {len(prompts)} default prompts")

    # Initialize model
    print(f"\nLoading model from {args.model}...")
    llm = LLM(model=args.model)
    print("Model loaded successfully!")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    print(f"\nBenchmark configuration:")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Warmup runs: {args.num_warmup}")
    print(f"  Timed runs: {args.num_runs}")

    # Run benchmarks
    baseline_result = run_baseline_benchmark(
        llm, prompts, sampling_params,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    lookahead_result = run_lookahead_benchmark(
        llm, prompts, sampling_params,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    # Print comparison
    print_comparison(baseline_result, lookahead_result)

    # Varying token lengths benchmark
    varying_results = None
    if not args.skip_varying_lengths:
        varying_results = run_lookahead_varying_tokens_benchmark(
            llm, prompts, sampling_params,
            lookahead_lengths=[1, 5, 10, 20],
            num_warmup=1,
            num_runs=3,
        )
        print_varying_tokens_comparison(baseline_result, varying_results)

    # Save results
    if args.output:
        save_results(
            baseline_result,
            lookahead_result,
            varying_results,
            Path(args.output),
        )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
