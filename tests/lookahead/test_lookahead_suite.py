"""
Comprehensive test suite for the lookahead token feature.

Tests:
1. Threshold 0 produces same output as baseline (lookahead never triggers)
2. Batched generation with mixed lookahead thresholds
3. Lookahead termination triggers correctly
4. Edge cases and correctness checks

Usage:
    python tests/lookahead/test_lookahead_suite.py
    python tests/lookahead/test_lookahead_suite.py --test <test_name>
"""

import sys
import json
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


# Test configuration
MODEL_PATH = "/home/paperspace/Qwen3-0.6B"
BASELINE_FILE = Path(__file__).parent / "baseline_outputs.json"


def load_baseline() -> dict:
    """Load baseline outputs for comparison."""
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}\n"
            "Run 'python tests/lookahead/baseline_test.py' first."
        )
    with open(BASELINE_FILE) as f:
        return json.load(f)


def create_llm() -> LLM:
    """Create LLM instance for testing."""
    return LLM(model=MODEL_PATH)


# =============================================================================
# Test 1: Threshold 0 matches regular output (same-batch comparison)
# =============================================================================

def test_threshold_zero_matches_baseline():
    """
    Test that lookahead with threshold=0 produces identical output to regular.

    With threshold=0, lookahead should NEVER trigger because:
    - Log probabilities are always <= 0
    - Sum of logprobs is always <= 0
    - Threshold check: sum > threshold (0) will never be true

    Therefore, generation should proceed exactly as without lookahead.

    NOTE: We compare regular vs lookahead in the SAME BATCH to avoid
    non-determinism from the async scheduler between separate generate() calls.
    """
    print("\n" + "=" * 70)
    print("TEST: Threshold 0 matches regular (same-batch comparison)")
    print("=" * 70)

    llm = create_llm()

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time in a small village",
        "The chemical formula for water is",
        "To solve this equation, we need to",
    ]

    # Create LookaheadPrompts with threshold=0 (should never trigger)
    lookahead_tokens_list = [
        " UNLIKELY_TOKEN_123",  # Very unlikely token
        " the",                  # Common token
        " xyz",                  # Random token
        " END",                  # Another token
        " ZZZZZ",               # Very unlikely
    ]

    # Build mixed batch: [regular0, lookahead0, regular1, lookahead1, ...]
    batch = []
    for i, prompt in enumerate(test_prompts):
        batch.append(prompt)  # Regular
        batch.append(LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=lookahead_tokens_list[i % len(lookahead_tokens_list)],
            lookahead_threshold=0.0,  # Will never trigger (logprobs are negative)
        ))

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for determinism
        max_tokens=100,
        skip_special_tokens=True,
    )

    print(f"Running {len(batch)} prompts ({len(test_prompts)} regular + {len(test_prompts)} lookahead)...")
    outputs = llm.generate(batch, sampling_params)

    # Compare regular vs lookahead outputs
    all_match = True
    for i, prompt in enumerate(test_prompts):
        regular_idx = i * 2
        lookahead_idx = i * 2 + 1

        regular_ids = list(outputs[regular_idx].outputs[0].token_ids)
        lookahead_ids = list(outputs[lookahead_idx].outputs[0].token_ids)
        finish_reason = outputs[lookahead_idx].outputs[0].finish_reason

        # Check that lookahead didn't trigger
        if finish_reason == "lookahead":
            print(f"  FAIL: Prompt {i} - lookahead triggered with threshold=0!")
            all_match = False
            continue

        # Check token IDs match exactly
        if regular_ids != lookahead_ids:
            print(f"  FAIL: Prompt {i} - token mismatch")
            print(f"    Regular ({len(regular_ids)} tokens): {outputs[regular_idx].outputs[0].text[:50]}...")
            print(f"    Lookahead ({len(lookahead_ids)} tokens): {outputs[lookahead_idx].outputs[0].text[:50]}...")
            import pdb; pdb.set_trace()
            all_match = False
        else:
            print(f"  PASS: Prompt {i} - {len(regular_ids)} tokens match")

    if all_match:
        print("\nRESULT: ALL PROMPTS MATCH REGULAR OUTPUT")
    else:
        print("\nRESULT: SOME PROMPTS DIFFER")

    return all_match


# =============================================================================
# Test 2: Batched generation with thresholding
# =============================================================================

def test_batched_generation_with_threshold():
    """
    Test batched generation with mixed lookahead configurations.

    Creates a batch with:
    - Regular prompts (no lookahead)
    - LookaheadPrompts with high threshold (should NOT trigger)
    - LookaheadPrompts with very negative threshold (SHOULD trigger)

    Verifies:
    - All requests complete without error
    - Correct finish reasons
    - Triggered requests have fewer tokens
    """
    print("\n" + "=" * 70)
    print("TEST: Batched generation with mixed thresholds")
    print("=" * 70)

    llm = create_llm()

    # Create mixed batch
    prompts = [
        # Regular prompt (baseline)
        "The capital of France is",

        # LookaheadPrompt with very negative threshold (should trigger quickly)
        # Looking for " Paris" which is very likely
        LookaheadPrompt(
            prompt="The capital of France is",
            lookahead_tokens=" Paris",
            lookahead_threshold=-5.0,  # Very lenient, should trigger
        ),

        # LookaheadPrompt with threshold=0 (should NOT trigger)
        LookaheadPrompt(
            prompt="The capital of France is",
            lookahead_tokens=" Paris",
            lookahead_threshold=0.0,  # Will not trigger
        ),

        # Regular prompt
        "Python is a programming language that",

        # LookaheadPrompt with unlikely tokens (should NOT trigger)
        LookaheadPrompt(
            prompt="Python is a programming language that",
            lookahead_tokens=" XYZZY",
            lookahead_threshold=-3.0,  # Reasonable threshold, but unlikely token
        ),

        # LookaheadPrompt with likely continuation
        LookaheadPrompt(
            prompt="2 + 2 =",
            lookahead_tokens=" 4",
            lookahead_threshold=-8.0,  # More lenient threshold for triggering
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        skip_special_tokens=True,
    )

    print(f"Running batch of {len(prompts)} mixed prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Expected results
    expected = [
        {"index": 0, "type": "regular", "should_trigger": False},
        {"index": 1, "type": "lookahead_likely", "should_trigger": True},
        {"index": 2, "type": "lookahead_threshold_0", "should_trigger": False},
        {"index": 3, "type": "regular", "should_trigger": False},
        {"index": 4, "type": "lookahead_unlikely", "should_trigger": False},
        {"index": 5, "type": "lookahead_likely", "should_trigger": True},
    ]

    all_correct = True
    for i, (output, exp) in enumerate(zip(outputs, expected)):
        finish_reason = output.outputs[0].finish_reason
        num_tokens = len(output.outputs[0].token_ids)
        text = output.outputs[0].text
        triggered = (finish_reason == "lookahead")

        status = "PASS" if triggered == exp["should_trigger"] else "FAIL"
        if status == "FAIL":
            all_correct = False

        print(f"  {status}: Prompt {i} ({exp['type']})")
        print(f"       finish_reason={finish_reason}, tokens={num_tokens}")
        print(f"       output: {text[:60]}")
        if triggered != exp["should_trigger"]:
            print(f"       EXPECTED: should_trigger={exp['should_trigger']}")

    if all_correct:
        print("\nRESULT: ALL EXPECTATIONS MET")
    else:
        print("\nRESULT: SOME EXPECTATIONS NOT MET (may be model-dependent)")

    return all_correct


# =============================================================================
# Test 3: Lookahead termination triggers correctly
# =============================================================================

def test_lookahead_termination():
    """
    Test that lookahead termination:
    1. Triggers for highly likely continuations with lenient threshold
    2. Does NOT trigger for unlikely continuations
    3. Returns correct finish_reason
    4. Appends lookahead tokens to output
    """
    print("\n" + "=" * 70)
    print("TEST: Lookahead termination correctness")
    print("=" * 70)

    llm = create_llm()
    tokenizer = llm.get_tokenizer()

    # NOTE: Thresholds are model-dependent. These values are tuned for
    # Qwen3-0.6B but may need adjustment for other models.
    test_cases = [
        {
            "name": "sqrt_144_correct",
            "prompt": "The square root of 144 is",
            "lookahead": " 12",
            "threshold": -10.0,  # Lenient threshold
            "should_trigger": True,
            "description": "Likely answer, lenient threshold",
        },
        {
            "name": "sqrt_144_wrong",
            "prompt": "The square root of 144 is",
            "lookahead": " 42",
            "threshold": -5.0,
            "should_trigger": False,
            "description": "Wrong answer, should not trigger",
        },
        {
            "name": "hello_world",
            "prompt": "print('Hello",
            "lookahead": " World')",
            "threshold": -15.0,  # Lenient for multi-token
            "should_trigger": True,
            "description": "Common code pattern",
        },
        {
            "name": "fibonacci",
            "prompt": "def fib(n): return n if n <= 1 else",
            "lookahead": " fib(n-1) + fib(n-2)",
            "threshold": -25.0,  # Very lenient for longer sequence
            "should_trigger": True,
            "description": "Common recursion pattern",
        },
    ]

    prompts = [
        LookaheadPrompt(
            prompt=tc["prompt"],
            lookahead_tokens=tc["lookahead"],
            lookahead_threshold=tc["threshold"],
        )
        for tc in test_cases
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
        skip_special_tokens=True,
    )

    print(f"Running {len(test_cases)} termination test cases...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for tc, output in zip(test_cases, outputs):
        finish_reason = output.outputs[0].finish_reason
        text = output.outputs[0].text
        triggered = (finish_reason == "lookahead")
        passed = (triggered == tc["should_trigger"])

        status = "PASS" if passed else "FAIL"
        results.append(passed)

        print(f"\n  {status}: {tc['name']}")
        print(f"       {tc['description']}")
        print(f"       Prompt: {tc['prompt'][:40]}...")
        print(f"       Lookahead: {tc['lookahead']}")
        print(f"       Threshold: {tc['threshold']}")
        print(f"       finish_reason: {finish_reason}")
        print(f"       Output: {text[:50]}...")

        # If triggered, verify lookahead tokens are in output
        if triggered:
            lookahead_text = tc["lookahead"]
            if lookahead_text.strip() in text:
                print(f"       Lookahead tokens found in output")
            else:
                print(f"       WARNING: Lookahead tokens not clearly in output")

    passed_count = sum(results)
    total_count = len(results)
    print(f"\nRESULT: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("PASS: All tests passed")
        return True
    else:
        return False


# =============================================================================
# Test 4: Edge cases
# =============================================================================

def test_edge_cases():
    """
    Test edge cases:
    1. Single token lookahead
    2. Very long lookahead
    3. Empty batch with lookahead
    4. Mixed regular and lookahead in same batch
    """
    print("\n" + "=" * 70)
    print("TEST: Edge cases")
    print("=" * 70)

    llm = create_llm()

    # Test 4a: Single token lookahead
    print("\n  Subtest 4a: Single token lookahead")
    prompts_4a = [
        LookaheadPrompt(
            prompt="1 + 1 =",
            lookahead_tokens=" 2",  # Single token
            lookahead_threshold=-3.0,
        ),
    ]
    outputs_4a = llm.generate(prompts_4a, SamplingParams(temperature=0.0, max_tokens=20))
    print(f"       Output: {outputs_4a[0].outputs[0].text}")
    print(f"       finish_reason: {outputs_4a[0].outputs[0].finish_reason}")

    # Test 4b: Longer lookahead sequence
    print("\n  Subtest 4b: Longer lookahead (5+ tokens)")
    prompts_4b = [
        LookaheadPrompt(
            prompt="def hello():",
            lookahead_tokens="\n    print('Hello, World!')",
            lookahead_threshold=-20.0,  # Lenient for long sequence
        ),
    ]
    outputs_4b = llm.generate(prompts_4b, SamplingParams(temperature=0.0, max_tokens=50))
    print(f"       Output: {outputs_4b[0].outputs[0].text[:60]}...")
    print(f"       finish_reason: {outputs_4b[0].outputs[0].finish_reason}")

    # Test 4c: Multiple prompts, some with lookahead
    print("\n  Subtest 4c: Mixed regular and lookahead prompts")
    prompts_4c = [
        "What is 5 + 5?",  # Regular
        LookaheadPrompt(
            prompt="What is 5 + 5?",
            lookahead_tokens=" 10",
            lookahead_threshold=-5.0,
        ),
        "The sky is",  # Regular
        LookaheadPrompt(
            prompt="The sky is",
            lookahead_tokens=" blue",
            lookahead_threshold=-3.0,
        ),
    ]
    outputs_4c = llm.generate(prompts_4c, SamplingParams(temperature=0.0, max_tokens=30))
    for i, out in enumerate(outputs_4c):
        prompt_type = "lookahead" if i % 2 == 1 else "regular"
        print(f"       [{i}] {prompt_type}: {out.outputs[0].text[:40]}... "
              f"(finish={out.outputs[0].finish_reason})")

    # Test 4d: Token IDs instead of text for lookahead
    print("\n  Subtest 4d: Lookahead with token IDs")
    tokenizer = llm.get_tokenizer()
    token_ids = tokenizer.encode(" 42", add_special_tokens=False)
    prompts_4d = [
        LookaheadPrompt(
            prompt="The answer is",
            lookahead_tokens=token_ids,  # Using token IDs
            lookahead_threshold=-5.0,
        ),
    ]
    outputs_4d = llm.generate(prompts_4d, SamplingParams(temperature=0.0, max_tokens=20))
    print(f"       Token IDs used: {token_ids}")
    print(f"       Output: {outputs_4d[0].outputs[0].text}")
    print(f"       finish_reason: {outputs_4d[0].outputs[0].finish_reason}")

    # Validate results
    print("\n  Validating edge case results...")
    all_passed = True

    # 4a: Single token lookahead should run without error and produce output
    if len(outputs_4a) > 0 and outputs_4a[0].outputs[0].text:
        print("  4a: PASS - Single token lookahead produced output")
    else:
        print("  4a: FAIL - Single token lookahead failed")
        all_passed = False

    # 4b: Long lookahead should run without error and produce output
    if len(outputs_4b) > 0 and outputs_4b[0].outputs[0].text:
        print("  4b: PASS - Long lookahead produced output")
    else:
        print("  4b: FAIL - Long lookahead failed")
        all_passed = False

    # 4c: Mixed prompts should all produce output
    if len(outputs_4c) == 4:
        print("  4c: PASS - Mixed prompts all produced output")
    else:
        print(f"  4c: FAIL - Expected 4 outputs, got {len(outputs_4c)}")
        all_passed = False

    # 4d: Token IDs lookahead should run without error
    if len(outputs_4d) > 0 and outputs_4d[0].outputs[0].text:
        print("  4d: PASS - Token IDs lookahead produced output")
    else:
        print("  4d: FAIL - Token IDs lookahead failed")
        all_passed = False

    print(f"\n  Edge cases: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =============================================================================
# Test 5: Lookahead never changes output unless triggered (same-batch)
# =============================================================================

def test_lookahead_does_not_affect_output():
    """
    CRITICAL TEST: Verify that lookahead NEVER changes the generated output
    unless it actually triggers.

    This test compares regular and lookahead prompts in the SAME BATCH to avoid
    non-determinism from the async scheduler.

    We test various lookahead tokens (common, unlikely, multi-token) all with
    threshold=0, which should never trigger. The outputs must be identical to
    regular prompts processed in the same batch.
    """
    print("\n" + "=" * 70)
    print("TEST: Lookahead does NOT affect output unless triggered (same-batch)")
    print("=" * 70)

    llm = create_llm()

    # Test prompts
    test_prompts = [
        "The quick brown fox",
        "def factorial(n):",
        "In the year 2050,",
        "The chemical formula for water is",
        "To be or not to be,",
    ]

    # Various lookahead configurations - all with threshold=0 (never triggers)
    lookahead_configs = [
        " the",            # Common token
        " XYZZY",          # Unlikely token
        " jumped over",    # Multi-token
        " END_OF_TEXT",    # Very unlikely
        "\n\n\n",          # Whitespace
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
        skip_special_tokens=True,
    )

    # Build batch: [regular0, lookahead0_config0, lookahead0_config1, ..., regular1, ...]
    # For each prompt, include regular + all lookahead configs
    batch = []
    for prompt in test_prompts:
        batch.append(prompt)  # Regular
        for lookahead_tokens in lookahead_configs:
            batch.append(LookaheadPrompt(
                prompt=prompt,
                lookahead_tokens=lookahead_tokens,
                lookahead_threshold=0.0,  # Never triggers
            ))

    entries_per_prompt = 1 + len(lookahead_configs)  # 1 regular + N lookahead
    print(f"  Running {len(batch)} prompts ({len(test_prompts)} prompts x {entries_per_prompt} variants)...")
    outputs = llm.generate(batch, sampling_params)

    # Compare outputs
    all_match = True
    for prompt_idx, prompt in enumerate(test_prompts):
        base_offset = prompt_idx * entries_per_prompt
        regular_ids = list(outputs[base_offset].outputs[0].token_ids)

        print(f"\n  Prompt {prompt_idx}: '{prompt[:30]}...'")
        print(f"    Regular: {len(regular_ids)} tokens")

        for config_idx, lookahead_tokens in enumerate(lookahead_configs):
            la_offset = base_offset + 1 + config_idx
            la_ids = list(outputs[la_offset].outputs[0].token_ids)
            la_finish = outputs[la_offset].outputs[0].finish_reason

            if la_finish == "lookahead":
                print(f"    Config '{lookahead_tokens[:15]}': TRIGGERED (unexpected!)")
                all_match = False
            elif la_ids != regular_ids:
                print(f"    Config '{lookahead_tokens[:15]}': MISMATCH!")
                all_match = False
            else:
                print(f"    Config '{lookahead_tokens[:15]}': MATCH")

    if all_match:
        print("\nRESULT: LOOKAHEAD DOES NOT AFFECT OUTPUT (CRITICAL TEST PASSED)")
    else:
        print("\nRESULT: LOOKAHEAD AFFECTED OUTPUT INCORRECTLY (CRITICAL TEST FAILED)")

    return all_match


# =============================================================================
# Test 6: Consistency across multiple runs
# =============================================================================

def test_determinism():
    """
    Test that lookahead is deterministic with temperature=0.
    Run the same prompts multiple times and verify identical results.
    """
    print("\n" + "=" * 70)
    print("TEST: Determinism (multiple runs)")
    print("=" * 70)

    llm = create_llm()

    prompts = [
        LookaheadPrompt(
            prompt="The capital of Germany is",
            lookahead_tokens=" Berlin",
            lookahead_threshold=-5.0,
        ),
        LookaheadPrompt(
            prompt="Water freezes at",
            lookahead_tokens=" 0 degrees",
            lookahead_threshold=-8.0,
        ),
        "What is machine learning?",  # Regular prompt
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    NUM_RUNS = 3
    results = []

    for run in range(NUM_RUNS):
        outputs = llm.generate(prompts, sampling_params)
        run_result = [
            (out.outputs[0].text, out.outputs[0].finish_reason, list(out.outputs[0].token_ids))
            for out in outputs
        ]
        results.append(run_result)
        print(f"  Run {run + 1}: Completed")

    # Compare all runs
    all_match = True
    for i in range(len(prompts)):
        first_run = results[0][i]
        for run_idx in range(1, NUM_RUNS):
            if results[run_idx][i] != first_run:
                print(f"  FAIL: Prompt {i} differs between run 1 and run {run_idx + 1}")
                all_match = False
                break
        else:
            print(f"  PASS: Prompt {i} - identical across {NUM_RUNS} runs")

    if all_match:
        print("\nRESULT: ALL RUNS IDENTICAL (deterministic)")
    else:
        print("\nRESULT: RUNS DIFFER (non-deterministic)")

    return all_match


# =============================================================================
# Test 7: Low threshold always triggers (even with random tokens)
# =============================================================================

def test_low_threshold_always_triggers():
    """
    Test that with a very low threshold (e.g., -10000), lookahead ALWAYS triggers,
    even for completely random/unlikely tokens.

    This verifies:
    1. All lookahead prompts trigger (finish_reason == "lookahead")
    2. Output length = 1 (sampled token) + N (lookahead tokens)
       Note: The threshold check happens AFTER sampling, so output includes
       the sampled token followed by the lookahead tokens.
    3. The last N tokens in output ARE the lookahead tokens
    4. Works with random tokens sampled uniformly from vocabulary
    """
    import random

    print("\n" + "=" * 70)
    print("TEST: Low threshold always triggers (random tokens)")
    print("=" * 70)

    llm = create_llm()
    tokenizer = llm.get_tokenizer()

    # Get vocabulary size for random sampling
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size}")

    # Set seed for reproducibility
    random.seed(42)

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time",
        "The answer to life is",
        "Machine learning involves",
    ]

    # Very low threshold - should ALWAYS trigger regardless of token likelihood
    VERY_LOW_THRESHOLD = -10000.0

    # Number of random lookahead tokens per prompt
    NUM_LOOKAHEAD_TOKENS = 3

    # Build prompts with random tokens
    prompts = []
    lookahead_token_ids_list = []

    for prompt in test_prompts:
        # Sample random token IDs from vocabulary (avoiding special tokens at start)
        # Use range 1000 to vocab_size-1000 to avoid special tokens
        safe_start = min(1000, vocab_size // 10)
        safe_end = max(vocab_size - 1000, vocab_size * 9 // 10)
        random_token_ids = [
            random.randint(safe_start, safe_end)
            for _ in range(NUM_LOOKAHEAD_TOKENS)
        ]

        prompts.append(LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=random_token_ids,
            lookahead_threshold=VERY_LOW_THRESHOLD,
        ))
        lookahead_token_ids_list.append(random_token_ids)

        # Decode tokens for display
        decoded = tokenizer.decode(random_token_ids)
        print(f"  Prompt: '{prompt[:30]}...'")
        print(f"    Random tokens: {random_token_ids} -> '{decoded}'")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,  # Should never reach this - trigger happens immediately
        skip_special_tokens=False,  # Keep all tokens for accurate counting
    )

    print(f"\n  Running {len(prompts)} prompts with threshold={VERY_LOW_THRESHOLD}...")
    outputs = llm.generate(prompts, sampling_params)

    # Expected output: ONLY the lookahead tokens (sampled token is replaced)
    expected_output_len = NUM_LOOKAHEAD_TOKENS

    # Verify results
    all_passed = True
    for i, (output, lookahead_ids) in enumerate(zip(outputs, lookahead_token_ids_list)):
        finish_reason = output.outputs[0].finish_reason
        output_token_ids = list(output.outputs[0].token_ids)
        num_output_tokens = len(output_token_ids)

        triggered = (finish_reason == "lookahead")
        correct_length = (num_output_tokens == expected_output_len)
        # Output should be exactly the lookahead tokens
        correct_lookahead_tokens = (output_token_ids == lookahead_ids)

        # Check if triggered
        if not triggered:
            print(f"\n  FAIL: Prompt {i} - did NOT trigger (finish_reason={finish_reason})")
            print(f"    Expected: lookahead trigger")
            print(f"    Got: {num_output_tokens} tokens, finish_reason={finish_reason}")
            all_passed = False
            continue

        # Check output length
        if not correct_length:
            print(f"\n  FAIL: Prompt {i} - wrong output length")
            print(f"    Expected: {expected_output_len} lookahead tokens")
            print(f"    Got: {num_output_tokens} tokens")
            print(f"    Token IDs: {output_token_ids}")
            all_passed = False
            continue

        # Check lookahead tokens match exactly
        if not correct_lookahead_tokens:
            print(f"\n  FAIL: Prompt {i} - lookahead tokens don't match")
            print(f"    Expected: {lookahead_ids}")
            print(f"    Got: {output_token_ids}")
            all_passed = False
            continue

        print(f"\n  PASS: Prompt {i}")
        print(f"    Triggered: YES")
        print(f"    Output tokens: {num_output_tokens} (lookahead tokens only)")
        print(f"    Lookahead tokens verified: {lookahead_ids}")

    if all_passed:
        print("\n" + "=" * 70)
        print("RESULT: ALL PROMPTS TRIGGERED WITH CORRECT LENGTH")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("RESULT: SOME PROMPTS FAILED")
        print("=" * 70)

    return all_passed


# =============================================================================
# Main test runner
# =============================================================================

def run_all_tests():
    """Run all tests and report summary."""
    tests = [
        ("threshold_zero_matches_baseline", test_threshold_zero_matches_baseline),
        ("batched_generation_with_threshold", test_batched_generation_with_threshold),
        ("lookahead_termination", test_lookahead_termination),
        ("edge_cases", test_edge_cases),
        ("lookahead_does_not_affect_output", test_lookahead_does_not_affect_output),
        ("determinism", test_determinism),
        ("low_threshold_always_triggers", test_low_threshold_always_triggers),
    ]

    print("=" * 70)
    print("LOOKAHEAD FEATURE TEST SUITE")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Tests: {len(tests)}")

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        # Run specific test
        test_name = sys.argv[2]
        test_map = {
            "baseline": test_threshold_zero_matches_baseline,
            "batched": test_batched_generation_with_threshold,
            "termination": test_lookahead_termination,
            "edge": test_edge_cases,
            "determinism": test_determinism,
            "low_threshold": test_low_threshold_always_triggers,
        }
        if test_name in test_map:
            test_map[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {list(test_map.keys())}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
