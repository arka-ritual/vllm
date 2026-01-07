"""
Profile baseline vs lookahead using cProfile to identify slow functions.
"""

import cProfile
import pstats
import io
from pstats import SortKey

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


MODEL_PATH = "/home/paperspace/Qwen3-0.6B"
BATCH_SIZE = 8
MAX_TOKENS = 64  # Shorter for clearer profiling
LOOKAHEAD_TOKENS = "abc"


def create_prompts():
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
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def run_lookahead(llm, prompts, sampling_params):
    lookahead_prompts = [
        LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=LOOKAHEAD_TOKENS,
            lookahead_threshold=0.0,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(lookahead_prompts, sampling_params)
    return outputs


def profile_and_report(func, name, *args):
    """Profile a function and return stats."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args)
    pr.disable()

    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions

    return result, s.getvalue(), pr


def main():
    print("=" * 70)
    print("CPROFILE COMPARISON: Baseline vs Lookahead")
    print("=" * 70)
    print(f"Batch size: {BATCH_SIZE}, Max tokens: {MAX_TOKENS}")
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
    print("Warmup...")
    run_baseline(llm, prompts, sampling_params)
    run_lookahead(llm, prompts, sampling_params)

    # Profile baseline
    print("\n" + "=" * 70)
    print("PROFILING: Baseline (no lookahead)")
    print("=" * 70)
    _, baseline_stats, baseline_pr = profile_and_report(
        run_baseline, "baseline", llm, prompts, sampling_params
    )
    print(baseline_stats)

    # Profile lookahead
    print("\n" + "=" * 70)
    print("PROFILING: Lookahead (threshold=0)")
    print("=" * 70)
    _, lookahead_stats, lookahead_pr = profile_and_report(
        run_lookahead, "lookahead", llm, prompts, sampling_params
    )
    print(lookahead_stats)

    # Extract key functions for comparison
    print("\n" + "=" * 70)
    print("KEY FUNCTION COMPARISON")
    print("=" * 70)

    key_functions = [
        "_prepare_inputs",
        "execute_model",
        "sample_tokens",
        "_check_lookahead_termination",
        "compute_logits",
        "forward",
    ]

    baseline_times = {}
    lookahead_times = {}

    for stat_name, stats in baseline_pr.stats.items():
        func_name = stat_name[2]  # (filename, lineno, funcname)
        for kf in key_functions:
            if kf in func_name:
                baseline_times[kf] = baseline_times.get(kf, 0) + stats[3]  # cumtime

    for stat_name, stats in lookahead_pr.stats.items():
        func_name = stat_name[2]
        for kf in key_functions:
            if kf in func_name:
                lookahead_times[kf] = lookahead_times.get(kf, 0) + stats[3]

    print(f"{'Function':<30} {'Baseline':>12} {'Lookahead':>12} {'Overhead':>12}")
    print("-" * 70)
    for kf in key_functions:
        bt = baseline_times.get(kf, 0)
        lt = lookahead_times.get(kf, 0)
        overhead = lt - bt
        print(f"{kf:<30} {bt*1000:>10.1f}ms {lt*1000:>10.1f}ms {overhead*1000:>+10.1f}ms")


if __name__ == "__main__":
    main()
