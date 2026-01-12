"""
Simple isolated test for longer lookahead tokens (> 1 token).

Tests that:
1. Thresholds BELOW exact logprobs -> SHOULD trigger (lookahead terminates)
2. Thresholds ABOVE exact logprobs -> should NOT trigger (normal generation)

Exact logprob values were measured from the base model.
"""

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


MODEL_PATH = "/home/paperspace/Qwen3-0.6B"


def test_longer_lookahead():
    """
    Test lookahead threshold behavior with exact logprob boundaries.

    For each test case, we have the exact sum of logprobs from the model.
    We test that:
    - threshold slightly BELOW exact -> triggers (sum > threshold)
    - threshold slightly ABOVE exact -> does NOT trigger (sum < threshold)
    """
    print("=" * 70)
    print("TEST: Lookahead Threshold Boundary Testing")
    print("=" * 70)

    llm = LLM(model=MODEL_PATH)
    tokenizer = llm.get_tokenizer()

    # Test cases with exact logprob values from model measurements
    # Format: (name, prompt, lookahead, exact_logprob, threshold_pass, threshold_fail)
    test_cases = [
        {
            "name": "Paris/Madrid",
            "prompt": "The capitals of France and Spain respectively are",
            "lookahead": " Paris and Madrid.",
            "exact_logprob": -6.67987,  # Measured from model
            "threshold_pass": -7.0,     # Below exact -> should trigger
            "threshold_fail": -6.0,     # Above exact -> should NOT trigger
        },
        {
            "name": "Pacific Ocean",
            "prompt": "The largest ocean in the world is called the",
            "lookahead": " Pacific Ocean.",
            "exact_logprob": -3.13156,  # Measured from model
            "threshold_pass": -4.0,     # Below exact -> should trigger
            "threshold_fail": -2.9,     # Above exact -> should NOT trigger
        },
        {
            "name": "Baseline for the next test",
            "prompt": "The capital of France is",
            "lookahead": " Paris which is a beautiful city.",
            "exact_logprob": -18.125,  # Measured from model
            "threshold_pass": -19.0,     # Below exact -> should trigger
            "threshold_fail": -17.5,     # Above exact -> should NOT trigger
        },
        {
            "name": "Sample first, then lookahead",
            "prompt": "The capital of France",  # expecting to sample 'is'
            "lookahead": " Paris which is a beautiful city.",
            "exact_logprob": -18.125,  # Measured from model
            "threshold_pass": -19.0,     # Below exact -> should trigger
            "threshold_fail": -17.5,     # Above exact -> should NOT trigger
        },
        {
            "name": "Sample many tokens, then lookahead",
            "prompt": "The capital of France",  # expecting to sample 'is Paris.'
            "lookahead": " The capital of France is also the capital of the Republic of France.",
            "exact_logprob": -9.4375,  # Measured from model
            "threshold_pass": -10.0,     # Below exact -> should trigger
            "threshold_fail": -9.0,     # Above exact -> should NOT trigger
        },
    ]

    # Show token counts for each lookahead
    print("\nLookahead token counts:")
    for tc in test_cases:
        token_ids = tokenizer.encode(tc["lookahead"], add_special_tokens=False)
        print(f"  {tc['name']}: '{tc['lookahead']}' -> {len(token_ids)} tokens: {token_ids}")
        print(f"    Exact logprob sum: {tc['exact_logprob']}")
        print(f"    Pass threshold: {tc['threshold_pass']} (below exact, should trigger)")
        print(f"    Fail threshold: {tc['threshold_fail']} (above exact, should NOT trigger)")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        skip_special_tokens=False,
    )

    all_passed = True

    # =========================================================================
    # Test 1: Thresholds BELOW exact logprobs -> SHOULD trigger
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Thresholds BELOW exact (should trigger)")
    print("=" * 70)

    prompts_pass = [
        LookaheadPrompt(
            prompt=tc["prompt"],
            lookahead_tokens=tc["lookahead"],
            lookahead_threshold=tc["threshold_pass"],
        )
        for tc in test_cases
    ]

    print(f"\nRunning {len(prompts_pass)} prompts (expecting ALL to trigger)...")
    outputs_pass = llm.generate(prompts_pass, sampling_params)

    for i, (output, tc) in enumerate(zip(outputs_pass, test_cases)):
        finish_reason = output.outputs[0].finish_reason
        output_text = output.outputs[0].text
        output_tokens = list(output.outputs[0].token_ids)
        triggered = (finish_reason == "lookahead")

        print(f"\n--- {tc['name']} (threshold={tc['threshold_pass']}, exact={tc['exact_logprob']}) ---")
        print(f"  finish_reason: {finish_reason}")
        print(f"  Output: '{output_text}'")

        if triggered:
            print(f"  PASS: Triggered as expected (threshold below exact)")
        else:
            print(f"  FAIL: Did NOT trigger! Threshold {tc['threshold_pass']} < exact {tc['exact_logprob']}")
            print(f"Output tokens: '{tokenizer.encode(output_text, add_special_tokens=False)}")
            all_passed = False

    # =========================================================================
    # Test 2: Thresholds ABOVE exact logprobs -> should NOT trigger
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Thresholds ABOVE exact (should NOT trigger)")
    print("=" * 70)

    prompts_fail = [
        LookaheadPrompt(
            prompt=tc["prompt"],
            lookahead_tokens=tc["lookahead"],
            lookahead_threshold=tc["threshold_fail"],
        )
        for tc in test_cases
    ]

    print(f"\nRunning {len(prompts_fail)} prompts (expecting NONE to trigger)...")
    outputs_fail = llm.generate(prompts_fail, sampling_params)

    for i, (output, tc) in enumerate(zip(outputs_fail, test_cases)):
        finish_reason = output.outputs[0].finish_reason
        output_text = output.outputs[0].text
        output_tokens = list(output.outputs[0].token_ids)
        triggered = (finish_reason == "lookahead")

        print(f"\n--- {tc['name']} (threshold={tc['threshold_fail']}, exact={tc['exact_logprob']}) ---")
        print(f"  finish_reason: {finish_reason}")
        print(f"  Output: '{output_text}' ({len(output_tokens)} tokens)")

        if not triggered:
            print(f"  PASS: Did NOT trigger as expected (threshold above exact)")
        else:
            print(f"  FAIL: Triggered unexpectedly! Threshold {tc['threshold_fail']} > exact {tc['exact_logprob']}")
            print(f"Output tokens: '{tokenizer.encode(output_text, add_special_tokens=False)}")
            all_passed = False

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
        print("  - Thresholds below exact logprobs correctly trigger")
        print("  - Thresholds above exact logprobs correctly do NOT trigger")
    else:
        print("RESULT: SOME TESTS FAILED")
        print("  Lookahead threshold logic may be broken!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import sys
    success = test_longer_lookahead()
    sys.exit(0 if success else 1)
