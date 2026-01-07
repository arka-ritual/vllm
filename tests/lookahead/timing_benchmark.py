"""
Timing benchmark comparing baseline vs lookahead generation.

This script measures the overhead of lookahead token processing by comparing:
1. Baseline: Regular generation without lookahead
2. Lookahead: Generation with lookahead tokens but threshold=0 (never triggers)

Since threshold=0 means lookahead never triggers, both should produce identical
outputs. The difference in time represents the overhead of processing lookahead tokens.

Usage:
    python tests/lookahead/timing_benchmark.py
"""

import time
from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


MODEL_PATH = "/home/paperspace/Qwen3-0.6B"
BATCH_SIZE = 8
MAX_TOKENS = 1024
LOOKAHEAD_TOKENS = "abc"
NUM_WARMUP_RUNS = 1
NUM_BENCHMARK_RUNS = 3


def create_prompts():
    """Create a batch of prompts for benchmarking."""
    base_prompts = [
        "The capital of France is",
        "In the year 2050, technology will",
        "def fibonacci(n):",
        "The quick brown fox jumps over",
        "Machine learning is a field of",
        "To make a delicious pasta, you need",
        "The history of the Roman Empire",
        "Climate change affects our planet by",
    ]
    # Ensure we have exactly BATCH_SIZE prompts
    prompts = (base_prompts * ((BATCH_SIZE // len(base_prompts)) + 1))[:BATCH_SIZE]
    return prompts


def benchmark_baseline(llm, prompts, sampling_params, num_runs):
    """Benchmark baseline generation (no lookahead)."""
    times = []
    last_outputs = None

    for run in range(num_runs):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        last_outputs = outputs

        # Verify output
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        print(f"  Run {run + 1}: {elapsed:.3f}s, {total_tokens} tokens generated")

    return times, last_outputs


def benchmark_lookahead(llm, prompts, sampling_params, num_runs):
    """Benchmark lookahead generation (threshold=0, never triggers)."""
    # Convert prompts to LookaheadPrompts
    lookahead_prompts = [
        LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            lookahead_threshold=0.0,  # Never triggers (logprobs are always negative)
        )
        for prompt in prompts
    ]

    times = []
    last_outputs = None

    for run in range(num_runs):
        start = time.perf_counter()
        outputs = llm.generate(lookahead_prompts, sampling_params)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        last_outputs = outputs

        # Verify output and that lookahead didn't trigger
        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        triggered = sum(1 for out in outputs if out.outputs[0].finish_reason == "lookahead")
        print(f"  Run {run + 1}: {elapsed:.3f}s, {total_tokens} tokens, {triggered} triggered")

    return times, last_outputs


def main():
    print("=" * 70)
    print("LOOKAHEAD TIMING BENCHMARK")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Lookahead tokens: '{LOOKAHEAD_TOKENS}'")
    print(f"Warmup runs: {NUM_WARMUP_RUNS}")
    print(f"Benchmark runs: {NUM_BENCHMARK_RUNS}")
    print()

    # Create LLM
    print("Loading model...")
    llm = LLM(model=MODEL_PATH)

    # Create prompts and sampling params
    prompts = create_prompts()
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=True,
        ignore_eos=True,  # Don't stop at EOS, generate full max_tokens
    )

    # Warmup runs
    print("\n" + "=" * 70)
    print("WARMUP (baseline)")
    print("=" * 70)
    benchmark_baseline(llm, prompts, sampling_params, NUM_WARMUP_RUNS)

    print("\n" + "=" * 70)
    print("WARMUP (lookahead)")
    print("=" * 70)
    benchmark_lookahead(llm, prompts, sampling_params, NUM_WARMUP_RUNS)

    # Benchmark runs
    print("\n" + "=" * 70)
    print("BENCHMARK: Baseline (no lookahead)")
    print("=" * 70)
    baseline_times, baseline_outputs = benchmark_baseline(llm, prompts, sampling_params, NUM_BENCHMARK_RUNS)

    print("\n" + "=" * 70)
    print("BENCHMARK: Lookahead (threshold=0, never triggers)")
    print("=" * 70)
    lookahead_times, lookahead_outputs = benchmark_lookahead(llm, prompts, sampling_params, NUM_BENCHMARK_RUNS)

    # Verify outputs match using SAME-BATCH comparison
    # (separate generate() calls have async scheduler non-determinism)
    print("\n" + "=" * 70)
    print("OUTPUT VERIFICATION (same-batch comparison)")
    print("=" * 70)
    print("  Running baseline + lookahead in SAME batch for accurate comparison...")

    # Build mixed batch: [regular0, lookahead0, regular1, lookahead1, ...]
    mixed_batch = []
    for prompt in prompts:
        mixed_batch.append(prompt)  # Regular
        mixed_batch.append(LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            lookahead_threshold=0.0,
        ))

    mixed_outputs = llm.generate(mixed_batch, sampling_params)

    all_match = True
    for i in range(len(prompts)):
        base_ids = list(mixed_outputs[i * 2].outputs[0].token_ids)
        la_ids = list(mixed_outputs[i * 2 + 1].outputs[0].token_ids)
        if base_ids == la_ids:
            print(f"  Prompt {i}: MATCH ({len(base_ids)} tokens)")
        else:
            print(f"  Prompt {i}: MISMATCH!")
            all_match = False
            # Find first difference
            for j, (b, l) in enumerate(zip(base_ids, la_ids)):
                if b != l:
                    print(f"    First diff at token {j}: baseline={b}, lookahead={l}")
                    break
            if len(base_ids) != len(la_ids):
                print(f"    Length: baseline={len(base_ids)}, lookahead={len(la_ids)}")

    if all_match:
        print("\n  ALL OUTPUTS MATCH - Lookahead produces identical results!")
    else:
        print("\n  WARNING: Some outputs differ!")

    # Calculate statistics
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_min = min(baseline_times)
    baseline_max = max(baseline_times)

    lookahead_avg = sum(lookahead_times) / len(lookahead_times)
    lookahead_min = min(lookahead_times)
    lookahead_max = max(lookahead_times)

    overhead_avg = lookahead_avg - baseline_avg
    overhead_pct = (overhead_avg / baseline_avg) * 100 if baseline_avg > 0 else 0

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBaseline (no lookahead):")
    print(f"  Average: {baseline_avg:.3f}s")
    print(f"  Min:     {baseline_min:.3f}s")
    print(f"  Max:     {baseline_max:.3f}s")

    print(f"\nLookahead (threshold=0):")
    print(f"  Average: {lookahead_avg:.3f}s")
    print(f"  Min:     {lookahead_min:.3f}s")
    print(f"  Max:     {lookahead_max:.3f}s")

    print(f"\nOverhead:")
    print(f"  Absolute: {overhead_avg:.3f}s")
    print(f"  Relative: {overhead_pct:.1f}%")

    # Per-token statistics
    tokens_per_batch = BATCH_SIZE * MAX_TOKENS
    baseline_tps = tokens_per_batch / baseline_avg
    lookahead_tps = tokens_per_batch / lookahead_avg

    print(f"\nThroughput:")
    print(f"  Baseline:  {baseline_tps:.1f} tokens/sec")
    print(f"  Lookahead: {lookahead_tps:.1f} tokens/sec")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
