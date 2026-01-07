"""
Profile lookahead overhead by comparing timing at each stage.

This script instruments key sections of the generation pipeline and
compares baseline vs lookahead (threshold=0) timings side-by-side.
"""

import time
import os

# Enable profiling via environment variable
os.environ["PROFILE_LOOKAHEAD"] = "1"

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


MODEL_PATH = "/home/paperspace/Qwen3-0.6B"
BATCH_SIZE = 8
MAX_TOKENS = 256  # Shorter for profiling
LOOKAHEAD_TOKENS = "abc"
NUM_RUNS = 3


def create_prompts():
    """Create batch of prompts."""
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
    return (base_prompts * ((BATCH_SIZE // len(base_prompts)) + 1))[:BATCH_SIZE]


def run_baseline(llm, prompts, sampling_params):
    """Run baseline generation."""
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    return elapsed, total_tokens


def run_lookahead(llm, prompts, sampling_params):
    """Run lookahead generation (threshold=0, never triggers)."""
    lookahead_prompts = [
        LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            lookahead_threshold=0.0,
        )
        for prompt in prompts
    ]
    start = time.perf_counter()
    outputs = llm.generate(lookahead_prompts, sampling_params)
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    triggered = sum(1 for out in outputs if out.outputs[0].finish_reason == "lookahead")
    return elapsed, total_tokens, triggered


def main():
    print("=" * 70)
    print("LOOKAHEAD OVERHEAD PROFILING")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Lookahead tokens: '{LOOKAHEAD_TOKENS}'")
    print(f"Runs: {NUM_RUNS}")
    print()

    # Create LLM
    print("Loading model...")
    llm = LLM(model=MODEL_PATH)

    prompts = create_prompts()
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=True,
        ignore_eos=True,
    )

    # Warmup
    print("\nWarmup runs...")
    run_baseline(llm, prompts, sampling_params)
    run_lookahead(llm, prompts, sampling_params)

    # Collect timings
    baseline_times = []
    lookahead_times = []

    print("\n" + "=" * 70)
    print("PROFILING RUNS")
    print("=" * 70)

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1} ---")

        # Baseline
        elapsed, tokens = run_baseline(llm, prompts, sampling_params)
        baseline_times.append(elapsed)
        print(f"Baseline:  {elapsed:.3f}s, {tokens} tokens")

        # Lookahead
        elapsed, tokens, triggered = run_lookahead(llm, prompts, sampling_params)
        lookahead_times.append(elapsed)
        print(f"Lookahead: {elapsed:.3f}s, {tokens} tokens, {triggered} triggered")

    # Results
    baseline_avg = sum(baseline_times) / len(baseline_times)
    lookahead_avg = sum(lookahead_times) / len(lookahead_times)
    overhead = lookahead_avg - baseline_avg
    overhead_pct = (overhead / baseline_avg) * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline average:  {baseline_avg:.3f}s")
    print(f"Lookahead average: {lookahead_avg:.3f}s")
    print(f"Overhead: {overhead:.3f}s ({overhead_pct:.1f}%)")

    # Per-iteration overhead
    num_iterations = MAX_TOKENS  # Approximately, ignoring prefill
    per_iter_overhead_ms = (overhead / num_iterations) * 1000
    print(f"\nPer-iteration overhead: ~{per_iter_overhead_ms:.2f}ms")

    print("\n" + "=" * 70)
    print("DETAILED PROFILING")
    print("=" * 70)
    print("\nTo see detailed per-function timing, we need to add instrumentation")
    print("to gpu_model_runner.py. Running instrumented comparison now...\n")

    # Run a single iteration with detailed timing
    profile_single_batch(llm, prompts, sampling_params)


def profile_single_batch(llm, prompts, sampling_params):
    """Profile a single batch with detailed timing."""
    import torch

    # We'll use CUDA events for accurate GPU timing
    print("Profiling with torch.cuda.Event timing...")
    print("(This measures actual GPU time, not just CPU dispatch time)\n")

    # For accurate comparison, we need to look at what happens inside
    # the model runner. Let's at least measure the high-level generate() call
    # with proper synchronization.

    torch.cuda.synchronize()

    # Baseline with sync
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs_base = llm.generate(prompts, sampling_params)
    end.record()
    torch.cuda.synchronize()
    baseline_gpu_ms = start.elapsed_time(end)

    # Lookahead with sync
    lookahead_prompts = [
        LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            lookahead_threshold=0.0,
        )
        for prompt in prompts
    ]

    start.record()
    outputs_la = llm.generate(lookahead_prompts, sampling_params)
    end.record()
    torch.cuda.synchronize()
    lookahead_gpu_ms = start.elapsed_time(end)

    print(f"Baseline GPU time:  {baseline_gpu_ms:.1f}ms")
    print(f"Lookahead GPU time: {lookahead_gpu_ms:.1f}ms")
    print(f"GPU overhead: {lookahead_gpu_ms - baseline_gpu_ms:.1f}ms ({(lookahead_gpu_ms/baseline_gpu_ms - 1)*100:.1f}%)")

    # Verify outputs match
    all_match = True
    for i, (b, l) in enumerate(zip(outputs_base, outputs_la)):
        if list(b.outputs[0].token_ids) != list(l.outputs[0].token_ids):
            all_match = False
            print(f"\nWARNING: Output mismatch at prompt {i}")
            break

    if all_match:
        print(f"\nOutputs verified: All {len(prompts)} prompts match")


if __name__ == "__main__":
    main()
