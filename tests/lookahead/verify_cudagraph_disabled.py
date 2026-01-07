"""
Quick verification that CUDA graphs are disabled with lookahead.
"""

import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


def main():
    print("=" * 70)
    print("Checking if CUDA graphs are used with/without lookahead")
    print("Look for 'cudagraph_mode' in the debug output")
    print("=" * 70)

    llm = LLM(model="/home/paperspace/Qwen3-0.6B")

    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    print("\n" + "=" * 70)
    print("BASELINE (no lookahead) - should use CUDA graphs")
    print("=" * 70)
    llm.generate(prompts, sampling_params)

    print("\n" + "=" * 70)
    print("LOOKAHEAD - CUDA graphs should be DISABLED")
    print("=" * 70)
    lookahead_prompts = [
        LookaheadPrompt(prompt=p, lookahead_tokens="abc", lookahead_threshold=0.0)
        for p in prompts
    ]
    llm.generate(lookahead_prompts, sampling_params)

    print("\n" + "=" * 70)
    print("Check the debug output above for 'cudagraph_mode: NONE' vs 'FULL'")
    print("=" * 70)


if __name__ == "__main__":
    main()
