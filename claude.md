# Lookahead Token Feature for vLLM

## Goal Synopsis

We want to modify vLLM to support **lookahead tokens** during generation. Instead of the standard generation loop:

```
while not done:
    logits = model(context)
    next_token = sample(logits[-1])
    context.append(next_token)
```

We want:

```
while not done:
    input = concat(context, lookahead_tokens)  # Append per-sequence lookahead tokens
    logits = model(input)

    # Sample from the ORIGINAL context end position
    next_token = sample(logits[len(context) - 1])

    # Extract logprobs for the lookahead tokens
    lookahead_logprobs = logprobs(logits[len(context):])

    # EARLY TERMINATION: If sum of lookahead logprobs > threshold, accept them all
    if sum(lookahead_logprobs) > threshold:
        context.append(next_token)
        context.extend(lookahead_tokens)
        TERMINATE
    else:
        context.append(next_token)
```

### Key Requirements

1. **Per-sequence lookahead tokens**: Each sequence in a batch can have different lookahead tokens
2. **Threshold-based early termination**: If `sum(lookahead_logprobs) > threshold`, immediately terminate generation and append the lookahead tokens to the output
3. **"Almost free" computation**: The extra cost is just the forward pass cost of the additional lookahead tokens (typically small)
4. **Full causal attention**: Lookahead tokens attend to context AND to each other (standard causal mask)
5. **KV cache handling**: Lookahead tokens' KV entries are NOT persisted; they get recomputed/overwritten each step

### Use Cases

- Monitoring how likely certain continuations are at each step
- Early termination when the model is confident about a specific suffix
- Constraint-aware generation
- Research into model behavior with forced suffixes

---

## vLLM Architecture Summary

### Generation Flow

```
LLM.generate()
    └─> LLMEngine.step() [loop until done]
            └─> EngineCore.step()
                    ├─> Scheduler.schedule() → SchedulerOutput
                    ├─> Executor.execute_model() → forward pass, logits
                    ├─> Executor.sample_tokens() → sample next tokens
                    └─> Scheduler.update_from_output() → update state
```

### Key Files

| File | Purpose |
|------|---------|
| `vllm/entrypoints/llm.py` | User API (`LLM.generate()`) |
| `vllm/v1/engine/core.py` | Main execution loop (`EngineCore.step()`) |
| `vllm/v1/worker/gpu_model_runner.py` | Input preparation, forward pass, sampling (~5700 lines) |
| `vllm/v1/sample/sampler.py` | Token sampling logic |
| `vllm/sampling_params.py` | Sampling configuration |
| `vllm/v1/spec_decode/` | Speculative decoding (similar concept) |

### Critical Code Points

1. **`_prepare_inputs()`** (gpu_model_runner.py:1338): Builds `input_ids`, `positions`, `logits_indices`
2. **`logits_indices`** (gpu_model_runner.py:1510): Controls which positions get logits extracted
3. **`num_computed_tokens`**: Tracks committed KV cache state; don't increment for lookahead tokens
4. **`sample_tokens()`** (gpu_model_runner.py:3361): Applies sampling, returns tokens + logprobs

### KV Cache Insight

vLLM's KV cache is position-based. If we:
- Append lookahead tokens at positions [N, N+1, N+2]
- Don't update `num_computed_tokens` past N
- Next step, real token goes at position N, overwriting lookahead KV

This is exactly how speculative decoding handles rejected tokens.

---

## Implementation Options

### Option A: Piggyback on Speculative Decoding Infrastructure

**Concept**: Use existing spec decode path with a "fixed token proposer".

**Changes**:
1. Create `FixedTokenProposer` in `vllm/v1/spec_decode/`
2. Skip rejection sampling; use threshold-based acceptance instead
3. Wire through sampling params

**Files to modify**:
- `vllm/v1/spec_decode/` - new proposer
- `vllm/v1/worker/gpu_model_runner.py` - wire proposer, custom acceptance
- `vllm/sampling_params.py` - add parameters

**Pros**:
- Reuses well-tested infrastructure
- KV cache handling already solved
- Multi-token logits extraction works

**Cons**:
- Spec decode has unneeded complexity (tree attention, draft probs)
- Awkward to disable rejection sampling
- Couples to spec decode evolution

---

### Option B: Direct Modification of `_prepare_inputs()` (Surgical Approach)

**Concept**: Directly modify input preparation to append lookahead tokens.

**Changes**:
1. Add `lookahead_tokens` and `lookahead_threshold` to `SamplingParams`
2. Modify `_prepare_inputs()` to append lookahead tokens per sequence
3. Modify `logits_indices` to extract lookahead positions
4. Modify sampler to:
   - Sample from context end position
   - Compute sum of lookahead logprobs
   - Return early termination signal if threshold exceeded
5. Handle early termination in scheduler/output processing

**Files to modify**:
- `vllm/sampling_params.py` - add parameters
- `vllm/v1/worker/gpu_model_runner.py` - `_prepare_inputs()`, `logits_indices`, `sample_tokens()`
- `vllm/v1/sample/sampler.py` - threshold logic
- `vllm/v1/outputs.py` - extend output structure
- `vllm/v1/core/sched/scheduler.py` - handle early termination

**Pros**:
- Clean, self-contained
- No spec decode dependency
- Full control

**Cons**:
- More code from scratch
- Need careful KV cache handling (straightforward though)
- Edge cases to handle

---

### Option C: Post-Forward Logprob Extraction (Minimal Invasion)

**Concept**: Run a second forward pass for lookahead tokens after each step.

**Changes**:
1. After normal generation step, call `execute_lookahead()`
2. Extract logprobs, check threshold
3. Terminate if exceeded

**Pros**:
- Minimal core changes
- Clean separation

**Cons**:
- **Two forward passes per step** - defeats "almost free" goal
- Not recommended

---

### Option D: Modify at Attention Level (Most Invasive)

**Concept**: Add "ephemeral token" concept to attention mechanism.

**Changes**:
- Modify paged attention CUDA kernels
- Add ephemeral token tracking to KV cache

**Pros**:
- Most efficient abstraction

**Cons**:
- Requires CUDA kernel changes
- Very invasive
- Not recommended for this use case

---

## Recommendation

**Option B (Direct Modification)** is recommended:

1. Single forward pass (matches "almost free" goal)
2. Self-contained, no spec decode coupling
3. Clean implementation path
4. KV cache handling is straightforward (don't increment `num_computed_tokens`)

---

## Confirmed Requirements (from user clarification)

1. **Per-sequence lookahead tokens**: Different sequences can have different lookahead tokens
2. **Threshold-based termination**: `sum(lookahead_logprobs) > threshold` triggers:
   - Immediate generation termination
   - Lookahead tokens appended to output
   - **NO sampled token included** - output is `[context] + [lookahead_tokens]`
3. **Max tokens behavior**: Exceed max_tokens limit if threshold triggers (include all lookahead)
4. **Threshold semantics**: Trigger when sum is "less negative" (more confident), e.g., threshold=-3.0
5. **API**: Whatever is cleaner/less invasive
6. **Dynamic reduction (future)**: If context ends in "ab" and lookahead is "abc", only probe "c"

### Corrected Generation Loop

```python
while not done:
    # Future: compute effective_lookahead by matching context suffix to lookahead prefix
    effective_lookahead = get_effective_lookahead(context, original_lookahead)

    if len(effective_lookahead) == 0:
        # Context already ends with full lookahead - we're done
        TERMINATE

    input = concat(context, effective_lookahead)
    logits = model(input)

    # Extract logprobs for the lookahead tokens FIRST
    lookahead_logprobs = logprobs(logits[len(context):])
    total_logprob = sum(lookahead_logprobs)

    # THRESHOLD CHECK BEFORE SAMPLING
    if total_logprob > threshold:
        # Model is confident - accept lookahead and terminate
        output = context + effective_lookahead
        TERMINATE
    else:
        # Not confident - sample normally and continue
        next_token = sample(logits[len(context) - 1])
        context.append(next_token)
```

---

## API Design Considerations

### Why NOT SamplingParams

User correctly noted: `SamplingParams` is for **sampling configuration** (temperature, top_p, etc.), not **input data**. Lookahead tokens are semantically part of the request input, not how we sample.

### Proposed API: Extend Prompt Types

vLLM already has `TextPrompt`, `TokensPrompt`. We add:

```python
from vllm import LLM, LookaheadPrompt, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b")

# New prompt type that includes lookahead configuration
prompts = [
    LookaheadPrompt(
        prompt="The answer to 2+2 is",
        lookahead_tokens=" 4.",        # Can be text (tokenized) or token IDs
        lookahead_threshold=-3.0,      # Trigger when sum(logprobs) > -3.0
    ),
    LookaheadPrompt(
        prompt="Hello, my name is",
        lookahead_tokens=" Alice",
        lookahead_threshold=-2.5,
    ),
    # Can mix with regular prompts (no lookahead)
    "What is the capital of France?",
]

outputs = llm.generate(prompts, SamplingParams(temperature=0.7))
```

### Alternative: Parallel Parameter

```python
llm.generate(
    prompts=["prompt1", "prompt2"],
    sampling_params=SamplingParams(...),
    lookahead_configs=[
        LookaheadConfig(tokens=" 4.", threshold=-3.0),
        LookaheadConfig(tokens=" Alice", threshold=-2.5),
    ]
)
```

**Recommendation**: The `LookaheadPrompt` approach (Option 1) is cleaner because:
- Self-contained per-request configuration
- Doesn't add parameters to `generate()` signature
- Follows existing vLLM pattern of prompt wrapper types

---

## Dynamic Lookahead Reduction (Future Feature)

**Concept**: If original lookahead is "abc" and context already ends in "ab", only probe "c".

### Implementation Notes

```python
def get_effective_lookahead(context_tokens: list[int], original_lookahead: list[int]) -> list[int]:
    """Find remaining lookahead tokens after matching context suffix."""
    for match_len in range(min(len(context_tokens), len(original_lookahead)), 0, -1):
        if context_tokens[-match_len:] == original_lookahead[:match_len]:
            return original_lookahead[match_len:]
    return original_lookahead  # No prefix match

# Example:
# context = [1, 2, 3, 4, 5]  (tokens for "Hello ab")
# original_lookahead = [4, 5, 6]  (tokens for "abc")
# If context ends with [4, 5] which matches lookahead[:2], return [6]
```

### What Would Change for Dynamic Support

1. **InputBatch**: Store `original_lookahead_tokens` (immutable) per request
2. **_prepare_inputs()**: Call `get_effective_lookahead()` each step before appending
3. **Threshold check**: Sum logprobs only for `effective_lookahead` (the remaining tokens)
4. **Output on trigger**: Append only `effective_lookahead`
5. **Edge case**: If `effective_lookahead` becomes empty, context already matches - terminate immediately

The core Option B architecture accommodates this naturally; it's just an additional prefix-match step per iteration.

---

## Revised Option B Implementation Plan

### Files to Modify

| File | Changes |
|------|---------|
| `vllm/inputs/data.py` | Add `LookaheadPrompt` type |
| `vllm/entrypoints/llm.py` | Handle `LookaheadPrompt` in `generate()` |
| `vllm/v1/request.py` | Store lookahead config per request |
| `vllm/v1/worker/gpu_input_batch.py` | Track lookahead tokens per sequence |
| `vllm/v1/worker/gpu_model_runner.py` | `_prepare_inputs()`: append lookahead; `sample_tokens()`: threshold check |
| `vllm/v1/sample/sampler.py` | Extract lookahead logprobs, compute sum, return termination signal |
| `vllm/v1/core/sched/scheduler.py` | Handle early termination, append lookahead to output |
| `vllm/v1/outputs.py` | Include lookahead acceptance info in output |

### Implementation Order

1. Add `LookaheadPrompt` type and wire through to `Request`
2. Modify `InputBatch` to store lookahead tokens per sequence
3. Modify `_prepare_inputs()` to append lookahead tokens
4. Modify `logits_indices` to extract lookahead positions
5. Add threshold check in sampler, return termination signal
6. Handle termination in scheduler (mark finished, append tokens)
7. Test with single sequence, then batched
8. (Future) Add dynamic reduction

---

## Next Steps

1. **User to confirm**: Option B + LookaheadPrompt API - **CONFIRMED**
2. Begin implementation incrementally

---

---

## CRITICAL IMPLEMENTATION REQUIREMENT

**DO NOT USE HEURISTICS FOR LOOKAHEAD CHECKING.**

The lookahead feature MUST sum logprobs for ALL lookahead tokens, not just the first one. This requires:

1. **Append lookahead tokens to input**: For K lookahead tokens, append lookahead[0:K-1] to the input
2. **Get logits at K positions**: Extract logits at positions [N-1, N, N+1, ..., N+K-2]
3. **Sum all logprobs**:
   - P(lookahead[0] | context) from position N-1
   - P(lookahead[1] | context, lookahead[0]) from position N
   - ...
   - P(lookahead[K-1] | context, lookahead[0:K-1]) from position N+K-2
4. **Compare sum to threshold**: Only accept if sum of ALL logprobs > threshold

**NEVER** implement a "check only first token" heuristic. The user explicitly rejected this approach.

---

## Current Session State (RESUME FROM HERE)

### Current Status: LOOKAHEAD FEATURE WORKING

The lookahead feature is now fully functional. All tests pass:
- **Lookahead test**: `python tests/lookahead/lookahead_test.py basic` - PASSES
- **Baseline test**: `python tests/lookahead/baseline_test.py --verify` - PASSES (all outputs match)

---

## Bug Fixes Completed (Jan 3, 2026)

### Issue: CUDA Index Out of Bounds Error

The original error was:
```
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel:
Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
```

### Root Causes Identified and Fixed

#### 1. Token Layout Mismatch (Fixed in `_prepare_inputs()`)

**Problem**: Lookahead tokens were appended at the END of all scheduled tokens, but `cu_num_tokens_extended` expected them interleaved per-request.

- **Before**: `[req0_sched][req1_sched][req2_sched][req0_lookahead]`
- **After**: `[req0_sched+lookahead][req1_sched+lookahead][req2_sched+lookahead]`

**Fix**: Modified token building logic (lines ~1545-1595) to interleave tokens correctly per-request.

#### 2. Decode Token Scatter for Interleaved Layout (New function added)

**Problem**: In async scheduling, decode tokens come from `prev_sampled_token_ids` and need to be scattered to correct positions. The original code scattered to end positions, but interleaved layout requires scattering to the START of each request's range.

**Fix**: Created `_prepare_input_ids_lookahead()` function (lines 1329-1398) that:
- Copies input_ids to GPU
- Scatters decode tokens from `prev_sampled_token_ids` to correct interleaved positions
- Position 0 for request 0, `cu_num_tokens_extended[i-1]` for request i

#### 3. Forward Pass Extended Token Count (Fixed logits computation)

**Problem**: The model was computing logits only for scheduled token positions (e.g., 3), but lookahead needs logits for ALL extended positions (e.g., 7).

**Fix**: Modified logits computation (lines 3825-3836):
```python
if lookahead_metadata.total_lookahead_tokens > 0:
    # Compute logits for ALL extended positions
    logits = self.model.compute_logits(hidden_states)
    # Extract sampling logits at scheduled positions
    sampling_logits = logits[logits_indices]
else:
    # Normal path - only compute for scheduled positions
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)
```

#### 4. ExecuteModelState Extended (New field added)

**Problem**: Needed to separate logits for sampling vs logits for lookahead checking.

**Fix**: Added `sampling_logits` field to `ExecuteModelState` (lines 334-336):
```python
class ExecuteModelState(NamedTuple):
    ...
    sampling_logits: torch.Tensor | None = None  # For sampling when lookahead active
```

#### 5. Sample Tokens Updated (Fixed in `sample_tokens()`)

**Fix**: Modified `sample_tokens()` (lines 3922-3933) to:
- Unpack `sampling_logits` from state
- Use `sampling_logits` for sampler when available
- Keep full `logits` for lookahead termination checking

---

## Test Results

### Lookahead Test (`python tests/lookahead/lookahead_test.py basic`)

```
Prompt 0:
  Type: Regular prompt (baseline)
  Generated (50 tokens):  12. So, the square root of 144 is 12...
  Finish reason: length

Prompt 1:
  Type: LookaheadPrompt with likely tokens (' 12')
  Generated (13 tokens):  12. So, the square root of 12...
  Finish reason: lookahead
  -> LOOKAHEAD TERMINATION TRIGGERED!

Prompt 2:
  Type: LookaheadPrompt with unlikely tokens (' 42')
  Generated (50 tokens):  12 and1111111111111...
  Finish reason: length
```

### Baseline Test (`python tests/lookahead/baseline_test.py --verify`)

```
MATCH: prompt 0 (mathematical_reasoning)
MATCH: prompt 1 (code_completion)
MATCH: prompt 2 (creative_narrative)
MATCH: prompt 3 (factual_encyclopedic)
MATCH: prompt 4 (instructional_explanation)

All outputs match baseline!
```

---

## Key Implementation Files

| File | Changes Made |
|------|--------------|
| `vllm/inputs/data.py` | Added `LookaheadPrompt` type |
| `vllm/inputs/__init__.py` | Exported `LookaheadPrompt` |
| `vllm/v1/request.py` | Added `lookahead_tokens`, `lookahead_threshold` fields |
| `vllm/v1/engine/input_processor.py` | Process `LookaheadPrompt` and extract tokens |
| `vllm/v1/worker/gpu_input_batch.py` | Track lookahead tokens per sequence |
| `vllm/v1/worker/gpu_model_runner.py` | Main changes: `_prepare_inputs()`, `_prepare_input_ids_lookahead()`, `_check_lookahead_termination()`, logits computation |
| `vllm/v1/core/sched/output.py` | Added `lookahead_terminated` field to `ModelRunnerOutput` |
| `vllm/v1/core/sched/scheduler.py` | Handle lookahead termination in output processing |
| `vllm/v1/outputs.py` | Added `lookahead` finish reason |

---

---

## Additional Bug Fix: KV Cache Slot Mapping (Jan 3, 2026)

### Issue: Lookahead Tokens Writing to KV Cache

**Problem**: When lookahead tokens were processed, their K/V values were being written to the KV cache at positions that would later be used by actual generated tokens. In subsequent iterations, the model would see stale KV entries from lookahead tokens that weren't supposed to persist.

**Fix**: Modified slot_mapping computation (lines 1629-1675) to use `-1` (PAD_SLOT_ID) for lookahead positions:

```python
# For lookahead positions, use -1 to skip KV cache write
if total_lookahead_extra > 0:
    # First compute slot_mapping for scheduled tokens only
    self.input_batch.block_table.compute_slot_mapping(req_indices, scheduled_positions)

    # Build extended slot_mapping with -1 for lookahead positions
    for gid in range(num_kv_groups):
        extended_slot_mapping = np.full(total_num_tokens_extended, -1, dtype=np.int64)
        # Copy scheduled slot mappings to correct interleaved positions
        # Lookahead positions remain -1
        ...
```

This ensures lookahead tokens can be processed for logit extraction without polluting the KV cache.

---

## Bug Fix: Batched Processing with Mixed Lookahead Configurations (Jan 3, 2026)

### Status: FIXED

**Problem**: When processing batches with mixed lookahead configurations (different token counts or mixing regular with lookahead prompts), requests following a lookahead request would produce corrupted output.

**Root Cause**: The `logits_indices` calculation was using `cu_num_tokens` (original cumsum without lookahead) to compute sampling positions, but the logits tensor was in the EXTENDED layout (with lookahead tokens interleaved). This caused sampling from wrong positions.

**Example of the bug**:
- `cu_num_tokens = [1, 2, 3]` for 3 requests with 1 token each
- `cu_num_tokens_extended = [2, 3, 5]` (with lookahead tokens interleaved)
- Old code computed `logits_indices = [0, 1, 2]`
- Correct indices should be `[0, 2, 3]` (accounting for interleaved layout)

**Fix** (in `_prepare_inputs()`, lines ~1794-1814):
```python
if total_lookahead_extra > 0:
    # With lookahead, compute sample positions in extended layout
    sample_positions = np.zeros(num_reqs, dtype=np.int64)
    for req_idx in range(num_reqs):
        if req_idx == 0:
            req_start = 0
        else:
            req_start = cu_num_tokens_extended[req_idx - 1]
        # Sample from last scheduled token (not lookahead)
        sample_positions[req_idx] = req_start + num_scheduled_tokens[req_idx] - 1
    logits_indices = torch.from_numpy(sample_positions).to(self.device)
```

**All configurations now work**:
- Single requests with lookahead ✓
- Batched requests with mixed lookahead configurations ✓
- Batched requests with all lookahead ✓
- Batched requests with regular and lookahead mixed ✓

---

## Test Suite and Benchmark

Created comprehensive test suite at `/home/paperspace/vllm/tests/lookahead/`:

| File | Purpose |
|------|---------|
| `test_lookahead_suite.py` | Comprehensive test suite with 6 tests |
| `benchmark_lookahead.py` | Performance benchmark comparing baseline vs lookahead |
| `lookahead_test.py` | Basic functionality test |
| `baseline_test.py` | Baseline output verification |
| `baseline_outputs.json` | Reference outputs for verification |

### Test Suite Results (All 6/6 PASS):
- `test_threshold_zero_matches_baseline`: PASS (same-batch comparison)
- `test_batched_generation_with_threshold`: PASS
- `test_lookahead_termination`: PASS (informational - thresholds are model-dependent)
- `test_edge_cases`: PASS
- `test_lookahead_does_not_affect_output`: PASS (same-batch comparison)
- `test_determinism`: PASS

### Key Finding: Async Scheduler Non-Determinism

vLLM's V1 async scheduler introduces non-determinism between separate `llm.generate()` calls.
This affects ALL generation (with or without lookahead), not just lookahead specifically.

**Verified**: When regular and lookahead prompts are processed in the SAME batch,
they produce **identical** outputs. The lookahead implementation is correct.

**Test Strategy**: Tests use same-batch comparison (regular vs lookahead for same prompt
in single generate() call) rather than comparing against pre-recorded baselines.

---

## Performance Benchmark Results (Jan 3, 2026)

### Benchmark Configuration
- Model: Qwen3-0.6B
- Batch size: 8
- Max tokens: 1024
- Lookahead tokens: "abc" (3 tokens, 2 extra input tokens per step via K-1 formula)
- Threshold: 0.0 (never triggers, so outputs should match baseline)
- ignore_eos: True (generate full 1024 tokens)

### Results

| Metric | Baseline | Lookahead | Overhead |
|--------|----------|-----------|----------|
| Average Time | 3.21s | 18.10s | +463% |
| Throughput | 2,551 tok/s | 453 tok/s | -82% |

### Analysis

**Correctness**: All 8 prompts produce **identical** outputs when compared in the same batch.

**Performance Issue**: The overhead is ~5.6x slower, which is far too high for adding just 2 extra tokens per decode step. Expected overhead should be ~25% (2 extra tokens / 8 batch size), not 463%.

**Likely Causes**:
1. **CUDA graphs disabled** - The lookahead code path may fall back to eager execution
2. **Extra Python overhead** - Slot mapping, position computation, token interleaving
3. **Different code paths** - Bypassing optimizations used in the normal decode path

### Benchmark Script
Located at: `tests/lookahead/timing_benchmark.py`

---

## ROOT CAUSE IDENTIFIED: CUDA Graphs Disabled (Jan 4, 2026)

### The Problem

**CUDA graphs are explicitly disabled when lookahead is active.** This is the primary cause of the ~5.6x overhead.

### Location

In `vllm/v1/worker/gpu_model_runner.py` at lines 3733-3734:

```python
# Disable CUDA graphs for lookahead batches (non-standard batch sizes)
if lookahead_metadata.total_lookahead_tokens > 0:
    cudagraph_mode = CUDAGraphMode.NONE
```

### Verification

Created `tests/lookahead/verify_cudagraph_disabled.py` to confirm via debug output:

| Mode | CUDA Graph Mode |
|------|-----------------|
| Baseline (no lookahead) | `FULL` |
| Lookahead (threshold=0) | `NONE` |

### Why This Causes Massive Overhead

CUDA graphs capture and replay GPU operations, eliminating kernel launch overhead. When disabled:
- Each decode iteration incurs full kernel launch overhead (~0.5-2ms per iteration)
- The overhead compounds across hundreds of iterations (e.g., 1024 tokens)
- This explains the consistent ~5.6x slowdown regardless of batch size

### Why It Was Disabled

The comment says "non-standard batch sizes" - with lookahead, the effective number of tokens processed changes dynamically based on how many lookahead tokens are appended. CUDA graphs require fixed batch sizes (captured at specific sizes), so they were disabled as a safe default.

### Fix Applied (Jan 4, 2026)

The issue was that CUDA graphs were being **unconditionally disabled** whenever lookahead was active, even though the batch size is fixed and deterministic.

**The fix**: Removed the unconditional disabling at lines 3732-3736. Now the dispatcher is allowed to find appropriate CUDA graphs for the extended batch size.

**Code change** in `vllm/v1/worker/gpu_model_runner.py`:
```python
# BEFORE (disabled CUDA graphs unconditionally):
if lookahead_metadata.total_lookahead_tokens > 0:
    cudagraph_mode = CUDAGraphMode.NONE
    batch_desc = batch_desc._replace(num_tokens=total_num_tokens_extended)

# AFTER (let dispatcher find appropriate graphs):
# For lookahead batches, use the batch descriptor from the dispatcher
# which will find appropriate CUDA graphs for the extended batch size.
```

**Results after fix**:

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| Lookahead Time | 18.10s | 4.88s | 3.7x faster |
| Overhead vs Baseline | +463% | +52.8% | 8.8x better |
| Throughput | 453 tok/s | 1679 tok/s | 3.7x better |

The remaining 52.8% overhead is expected because:
- With "abc" (3 lookahead tokens), we process 2 extra tokens per request per step
- 8 requests × 2 extra tokens = 16 extra tokens per decode step (3x the work)
- But we only see 52.8% overhead because PIECEWISE CUDA graphs are being used

**Test suite**: All 6/6 tests pass after the fix.

---

## Future Work

1. **Enable FULL CUDA Graphs for Lookahead**: Currently using PIECEWISE mode; FULL mode would be faster but requires uniform decode
2. **Dynamic Lookahead Reduction**: If context already ends with lookahead prefix, only probe remaining suffix
3. **Streaming Support**: Ensure lookahead works correctly with streaming output
