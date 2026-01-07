"""
Test script for the lookahead token feature.

This script tests the LookaheadPrompt functionality that allows early termination
when the model predicts a specific sequence of tokens with high confidence.

Usage:
    python tests/lookahead/lookahead_test.py
"""

import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


def test_lookahead_basic():
    """Test basic lookahead functionality."""

    model_path = "/home/paperspace/Qwen3-0.6B"

    print(f"Loading model from {model_path}...")
    llm = LLM(model=model_path)

    # Test case 1: Lookahead with likely continuation
    # The model should predict "12" after "The square root of 144 is "
    prompts = [
        # Regular prompt (baseline)
        "The square root of 144 is",

        # LookaheadPrompt - looking for " 12" which is the expected answer
        LookaheadPrompt(
            prompt="The square root of 144 is",
            lookahead_tokens=" 12",  # String will be tokenized
            lookahead_threshold=-3.0,  # Threshold for acceptance
        ),

        # LookaheadPrompt with unlikely continuation
        LookaheadPrompt(
            prompt="The square root of 144 is",
            lookahead_tokens=" 42",  # Wrong answer - should NOT terminate early
            lookahead_threshold=-3.0,
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=50,
        skip_special_tokens=True,
    )

    print(f"\nRunning lookahead test with {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    for i, output in enumerate(outputs):
        prompt_text = output.prompt if isinstance(output.prompt, str) else output.prompt
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        num_tokens = len(output.outputs[0].token_ids)

        print(f"\nPrompt {i}:")
        if i == 0:
            print("  Type: Regular prompt (baseline)")
        elif i == 1:
            print("  Type: LookaheadPrompt with likely tokens (' 12')")
        else:
            print("  Type: LookaheadPrompt with unlikely tokens (' 42')")

        print(f"  Input: {prompt_text}")
        print(f"  Generated ({num_tokens} tokens): {generated_text[:100]}...")
        print(f"  Finish reason: {finish_reason}")

        # Check if lookahead termination occurred
        if finish_reason == "lookahead":
            print("  -> LOOKAHEAD TERMINATION TRIGGERED!")

    return outputs


def test_lookahead_code_completion():
    """Test lookahead with code completion scenario."""

    model_path = "/home/paperspace/Qwen3-0.6B"

    print(f"\nLoading model from {model_path}...")
    llm = LLM(model=model_path)

    # Code completion with expected continuation
    prompts = [
        # Fibonacci function completion - expecting recursion
        LookaheadPrompt(
            prompt="def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
            lookahead_tokens=" fibonacci(n-1) + fibonacci(n-2)",
            lookahead_threshold=-5.0,  # More lenient threshold for longer sequence
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
        skip_special_tokens=True,
    )

    print(f"\nRunning code completion lookahead test...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 60)
    print("CODE COMPLETION RESULTS:")
    print("=" * 60)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        print(f"\nPrompt: def fibonacci(n):...")
        print(f"Generated: {generated_text}")
        print(f"Finish reason: {finish_reason}")

    return outputs


def test_lookahead_with_token_ids():
    """Test lookahead using explicit token IDs."""

    model_path = "/home/paperspace/Qwen3-0.6B"

    print(f"\nLoading model from {model_path}...")
    llm = LLM(model=model_path)

    # Get tokenizer to convert tokens to IDs
    tokenizer = llm.get_tokenizer()

    # Tokenize " 12" to get token IDs
    token_ids = tokenizer.encode(" 12", add_special_tokens=False)
    print(f"Token IDs for ' 12': {token_ids}")

    prompts = [
        LookaheadPrompt(
            prompt="The square root of 144 is",
            lookahead_tokens=token_ids,  # Using token IDs directly
            lookahead_threshold=-3.0,
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
        skip_special_tokens=True,
    )

    print(f"\nRunning token ID lookahead test...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 60)
    print("TOKEN ID LOOKAHEAD RESULTS:")
    print("=" * 60)

    for output in outputs:
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        print(f"Generated: {generated_text}")
        print(f"Finish reason: {finish_reason}")

    return outputs


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "basic":
            test_lookahead_basic()
        elif test_name == "code":
            test_lookahead_code_completion()
        elif test_name == "token_ids":
            test_lookahead_with_token_ids()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: basic, code, token_ids")
    else:
        # Run all tests
        print("\n" + "=" * 60)
        print("RUNNING ALL LOOKAHEAD TESTS")
        print("=" * 60)

        test_lookahead_basic()
        test_lookahead_code_completion()
        test_lookahead_with_token_ids()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
