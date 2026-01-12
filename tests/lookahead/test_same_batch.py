"""Test that lookahead with threshold=0 matches regular output when processed in the same batch."""

from vllm import LLM, SamplingParams
from vllm.inputs import LookaheadPrompt


def main():
    llm = LLM(model='/home/paperspace/Qwen3-0.6B')
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=100, skip_special_tokens=True)

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Once upon a time in a small village",
        "def fibonacci(n):",
        "The quick brown fox",
    ]

    print("="*70)
    print("TEST: Regular vs Lookahead (threshold=0) in SAME BATCH")
    print("="*70)
    print("If lookahead with threshold=0 is correct, it should produce")
    print("identical output to regular prompts when processed together.")
    print()

    # Create mixed batch: [regular, lookahead, regular, lookahead, ...]
    batch = []
    for prompt in test_prompts:
        # Regular prompt
        batch.append(prompt)
        # Lookahead with threshold=0 (should never trigger)
        batch.append(LookaheadPrompt(
            prompt=prompt,
            lookahead_tokens=" UNLIKELY_TOKEN",
            lookahead_threshold=0.0,
        ))

    print(f"Batch size: {len(batch)} ({len(test_prompts)} regular + {len(test_prompts)} lookahead)")
    print()

    outputs = llm.generate(batch, sp)

    all_match = True
    for i, prompt in enumerate(test_prompts):
        regular_idx = i * 2
        lookahead_idx = i * 2 + 1

        regular_output = outputs[regular_idx]
        lookahead_output = outputs[lookahead_idx]

        regular_ids = list(regular_output.outputs[0].token_ids)
        lookahead_ids = list(lookahead_output.outputs[0].token_ids)

        regular_text = regular_output.outputs[0].text
        lookahead_text = lookahead_output.outputs[0].text

        lookahead_finish = lookahead_output.outputs[0].finish_reason

        print(f"Prompt {i}: '{prompt[:40]}...'")
        print(f"  Regular:   {len(regular_ids)} tokens, '{regular_text[:50]}...'")
        print(f"  Lookahead: {len(lookahead_ids)} tokens, '{lookahead_text[:50]}...'")
        print(f"  Lookahead finish_reason: {lookahead_finish}")

        if regular_ids == lookahead_ids:
            print(f"  MATCH: Token-for-token identical")
        else:
            print(f"  MISMATCH!")
            all_match = False
            # Find first difference
            for j, (r, l) in enumerate(zip(regular_ids, lookahead_ids)):
                if r != l:
                    print(f"    First diff at token {j}:")
                    print(f"      Regular:   {r} = '{tokenizer.decode([r])}'")
                    print(f"      Lookahead: {l} = '{tokenizer.decode([l])}'")
                    break
            if len(regular_ids) != len(lookahead_ids):
                print(f"    Length mismatch: {len(regular_ids)} vs {len(lookahead_ids)}")
        print()

    print("="*70)
    if all_match:
        print("RESULT: ALL PROMPTS MATCH - Lookahead with threshold=0 is correct!")
    else:
        print("RESULT: SOME PROMPTS MISMATCH - There may be a bug in lookahead")
    print("="*70)

    return all_match


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
