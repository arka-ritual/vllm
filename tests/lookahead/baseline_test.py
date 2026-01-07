"""
Baseline test for vLLM generation before lookahead token feature.

This script generates outputs for a set of varied prompts using greedy decoding
(temperature=0) with the Qwen3-0.6B model. The outputs are saved to a JSON file
for future verification after implementing the lookahead token feature.

Usage:
    python tests/lookahead/baseline_test.py

The outputs will be saved to tests/lookahead/baseline_outputs.json
"""

import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams


# Varied prompts to test different generation scenarios
TEST_PROMPTS = [
    # Mathematical/logical reasoning
    "The square root of 144 is",

    # Code completion
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",

    # Creative/narrative
    "Once upon a time in a small village nestled between two mountains, there lived a young",

    # Factual/encyclopedic
    "The capital city of Japan is Tokyo, which has a population of approximately",

    # Conversational/instruction-following
    "Please explain the difference between a list and a tuple in Python. A list is",
]

# Prompt metadata for documentation
PROMPT_METADATA = {
    0: "mathematical_reasoning",
    1: "code_completion",
    2: "creative_narrative",
    3: "factual_encyclopedic",
    4: "instructional_explanation",
}


def run_baseline_generation():
    """Run baseline generation and save outputs."""

    model_path = "/home/paperspace/Qwen3-0.6B"

    print(f"Loading model from {model_path}...")
    llm = LLM(model=model_path)

    # Greedy decoding parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=256,
        skip_special_tokens=True,
    )

    print(f"Running generation for {len(TEST_PROMPTS)} prompts...")
    outputs = llm.generate(TEST_PROMPTS, sampling_params)

    # Collect results
    results = {
        "model": model_path,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "generations": []
    }

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)

        generation_result = {
            "index": i,
            "category": PROMPT_METADATA[i],
            "prompt": prompt,
            "generated_text": generated_text,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
        }
        results["generations"].append(generation_result)

        print(f"\n{'='*60}")
        print(f"Prompt {i} ({PROMPT_METADATA[i]}):")
        print(f"  Input: {prompt[:50]}...")
        print(f"  Output ({len(token_ids)} tokens): {generated_text[:100]}...")

    # Save results
    output_dir = Path(__file__).parent
    output_file = output_dir / "baseline_outputs.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Baseline outputs saved to: {output_file}")
    print(f"Total generations: {len(results['generations'])}")

    return results


def verify_against_baseline(new_outputs_file: str = None):
    """Verify new outputs match baseline (to be used after implementation)."""

    baseline_file = Path(__file__).parent / "baseline_outputs.json"

    if not baseline_file.exists():
        print("ERROR: Baseline file not found. Run baseline generation first.")
        return False

    with open(baseline_file) as f:
        baseline = json.load(f)

    if new_outputs_file:
        with open(new_outputs_file) as f:
            new_outputs = json.load(f)
    else:
        # Re-run generation and compare
        print("Re-running generation to verify...")
        new_outputs = run_baseline_generation()

    # Compare
    all_match = True
    for i, (base_gen, new_gen) in enumerate(zip(
        baseline["generations"], new_outputs["generations"]
    )):
        if base_gen["token_ids"] != new_gen["token_ids"]:
            print(f"MISMATCH at prompt {i} ({base_gen['category']}):")
            print(f"  Baseline: {base_gen['generated_text'][:50]}...")
            print(f"  New:      {new_gen['generated_text'][:50]}...")
            all_match = False
        else:
            print(f"MATCH: prompt {i} ({base_gen['category']})")

    if all_match:
        print("\nAll outputs match baseline!")
    else:
        print("\nWARNING: Some outputs differ from baseline!")

    return all_match


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_against_baseline()
    else:
        run_baseline_generation()
