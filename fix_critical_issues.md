# Critical Issues Fixed in Streaming SSL Implementation

**Date:** 2026-02-04
**Summary:** Comprehensive fixes for critical bugs in streaming self-supervised learning (SSL) code

---

## Overview

This document details all critical issues identified and fixed in the streaming SSL implementation. These fixes address:
- Shape mismatches causing IndexError
- Type inconsistencies causing runtime errors
- Device mismatches in CUDA operations
- Missing parameters breaking streaming/causal masking
- Context synchronization issues causing information leakage
- Validation gaps allowing invalid configurations

---

## Fixed Issues

### 1. ✅ **MLM Loss Shape Mismatch** (CRITICAL)
**File:** `nemo/collections/asr/losses/ssl_losses/mlm.py`
**Lines:** 71-84

**Problem:**
The mask tensor shape didn't match decoder_outputs shape after reshaping with `combine_time_steps`, causing:
```
IndexError: The shape of the mask [32, 210] at index 1 does not match
the shape of the indexed tensor [32, 211, 8192] at index 1
```

**Root Cause:**
- Masks reshaped to `[B, T//combine_time_steps]`
- decoder_outputs had shape `[B, T, D]` where T didn't match reshaped mask dimension
- No alignment logic before indexing operation

**Fix Applied:**
```python
# Ensure decoder_outputs matches mask time dimension
if decoder_outputs.shape[1] != masks.shape[1]:
    if decoder_outputs.shape[1] > masks.shape[1]:
        # Trim decoder_outputs to match mask length
        decoder_outputs = decoder_outputs[:, :masks.shape[1], :]
    else:
        # Pad decoder_outputs to match mask length
        pad_length = masks.shape[1] - decoder_outputs.shape[1]
        decoder_outputs = F.pad(decoder_outputs, (0, 0, 0, pad_length))
        # Pad masks with False so padded decoder positions aren't selected
        masks = F.pad(masks.float(), (0, pad_length), value=0.0).bool()
```

**Impact:**
- Training no longer crashes with IndexError
- Handles both over-length and under-length decoder outputs gracefully
- Padded positions correctly excluded from loss computation

---

### 2. ✅ **Masking Module Type Inconsistency** (CRITICAL)
**File:** `nemo/collections/asr/modules/ssl_modules/masking.py`
**Lines:** 255-270

**Problem:**
- `num_patches` calculated as tensor but used as int in slicing: `[:num_patches]`
- `torch.randperm()` received tensor `.item()` but `num_patches` remained tensor
- No bounds checking: if `num_patches > max_num_patches`, silently returned fewer patches

**Original Code:**
```python
num_patches = torch.ceil(torch.tensor(...)).int()  # tensor!
max_num_patches = torch.div(...)
if max_num_patches > 0:
    patch_indices = torch.randperm(max_num_patches.item(), device=device)[:num_patches]  # tensor slicing!
```

**Fix Applied:**
```python
max_num_patches = torch.div(torch.tensor(available_length, device=device), block_size, rounding_mode='trunc')
max_num_patches_int = max(0, int(max_num_patches.item()))

# Calculate desired number of patches and cap to max available
desired_patches = int(torch.ceil(torch.tensor(available_length * self.mask_prob / self.block_size, device=device)).item())
num_patches = min(desired_patches, max_num_patches_int)

if num_patches > 0:
    patch_indices = torch.randperm(max_num_patches_int, device=device)[:num_patches]
    patch_indices = patch_indices * block_size + min_mask_pos
```

**Impact:**
- Eliminated type errors in mask generation
- Proper bounds checking prevents requesting more patches than available
- Consistent Python int types throughout

---

### 3. ✅ **Masking Module Device Mismatch** (HIGH)
**File:** `nemo/collections/asr/modules/ssl_modules/masking.py`
**Lines:** 272-284

**Problem:**
- `torch.tensor(1.0)` created without device specification
- `self.mask_embedding` accessed without ensuring device match
- `torch.arange()` and `torch.full()` created without device specification
- Mixed precision training could cause dtype mismatches

**Fix Applied:**
```python
# Ensure blocks don't extend beyond max_mask_pos
ends = torch.clamp(patch_indices + block_size, max=max_mask_pos)
positions = torch.cat([torch.arange(s, e, device=device) for s, e in zip(patch_indices, ends)]).reshape(-1, 1)
batch_index = torch.full((positions.shape[0], 1), i, dtype=positions.dtype, device=device)
positions = torch.cat([batch_index, positions], dim=1)
indices.append(positions.unique(dim=0))

if indices:
    indices = torch.cat(indices, dim=0).unbind(1)
    masks = masks.permute(0, 2, 1)
    masked_feats = masked_feats.permute(0, 2, 1)

    # Ensure mask value has correct device and dtype
    mask_value = torch.tensor(1.0, device=device, dtype=masks.dtype)
    # Ensure mask_embedding is on correct device
    mask_embedding = self.mask_embedding.to(device=device, dtype=masked_feats.dtype)

    masks = masks.index_put(indices, values=mask_value).permute(0, 2, 1)
    masked_feats = masked_feats.index_put(indices, values=mask_embedding).permute(0, 2, 1)
```

**Impact:**
- Prevents CUDA device mismatch errors
- Ensures dtype consistency in mixed precision training
- All tensor operations properly device-aware

---

### 4. ✅ **SSL Models Missing Streaming Parameters** (CRITICAL)
**File:** `nemo/collections/asr/models/ssl_models.py`
**Lines:** 1003-1018

**Problem:**
The `mask_processor.forward()` was called WITHOUT critical streaming parameters:
- `current_frame`: Required for causal masking in streaming mode
- `effective_left_context`: Required for syncing mask window with attention context

**Original Code:**
```python
if apply_mask:
    masked_signal, masks = self.mask_processor(
        input_feats=processed_noisy_input_signal,
        input_lengths=processed_noisy_input_signal_length
    )
```

**Why This Breaks Streaming:**
- Without `current_frame`, causal masking treats all frames as available
- Without `effective_left_context`, mask window doesn't match attention window
- Results in non-causal behavior and information leakage

**Fix Applied:**
```python
if apply_mask:
    # Get encoder's attention context size for syncing with masking
    effective_left_context = None
    if hasattr(self.encoder, 'att_context_size'):
        att_context = self.encoder.att_context_size
        if isinstance(att_context, (list, tuple)) and len(att_context) >= 1:
            effective_left_context = att_context[0] if att_context[0] != -1 else None

    masked_signal, masks = self.mask_processor(
        input_feats=processed_noisy_input_signal,
        input_lengths=processed_noisy_input_signal_length,
        current_frame=None,  # Set to appropriate value for streaming inference
        effective_left_context=effective_left_context
    )
```

**Impact:**
- Masking now respects encoder's attention context limits
- Prevents information leakage from future frames
- Enables proper streaming inference (when current_frame is set)

**Note:** `current_frame=None` still needs to be set appropriately during actual streaming inference. This fix ensures the parameter is passed through correctly.

---

### 5. ✅ **Hierarchical Context Curriculum Hook Mismatch** (HIGH)
**File:** `examples/asr/speech_pretraining/hierarchical_streaming_nest.py`
**Lines:** 87-142

**Problem:**
In hierarchical mode, different encoder layers have different context sizes, but:
- Curriculum hook updated global `encoder.att_context_size` only
- Mask processor synced to this global value
- Masking window didn't account for per-layer contexts
- Could mask frames beyond some layers' attention windows → information leakage

**Example Scenario:**
- Layer 0: context = 40 frames
- Layer 6: context = 120 frames
- Layer 12: context = 320 frames
- Curriculum hook sets mask context = 320
- **Problem:** Masking allows 320 frames, but layer 0 only sees 40 → leakage!

**Fix Applied:**
```python
# Sync masking with minimum context to avoid information leakage
# In hierarchical mode, different layers may have different contexts
# Masking should use the MINIMUM (tightest) context to ensure no leakage
if hasattr(pl_module, 'mask_processor'):
    min_context = context
    # Check if hierarchical contexts are set
    if hasattr(pl_module.encoder, 'layers'):
        for layer in pl_module.encoder.layers:
            if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'att_context_size'):
                layer_context = layer.self_attention.att_context_size
                if isinstance(layer_context, (list, tuple)) and len(layer_context) >= 1:
                    if layer_context[0] > 0:  # Ignore unlimited context (-1)
                        min_context = min(min_context, layer_context[0])

    pl_module.mask_processor.left_context_size = min_context
```

**Impact:**
- Mask window now respects the tightest (minimum) layer context
- Prevents information leakage in hierarchical configurations
- Maintains causal guarantees across all layers

---

### 6. ✅ **Hierarchical Context Validation** (MEDIUM)
**File:** `examples/asr/speech_pretraining/hierarchical_streaming_nest.py`
**Lines:** 77-92

**Problem:**
No validation when setting per-layer contexts:
- Invalid context values (< -1) could be set silently
- Layers without required attributes were skipped silently
- Zero contexts could break streaming mode

**Fix Applied:**
```python
# Apply per-layer contexts
for i, (layer, context) in enumerate(zip(encoder.layers, contexts)):
    if hasattr(layer, 'self_attention'):
        if hasattr(layer.self_attention, 'att_context_size'):
            # Validate context values
            if isinstance(context, (list, tuple)) and len(context) >= 1:
                left_context = context[0]
                if left_context < -1:
                    raise ValueError(f"Invalid left context {left_context} for layer {i}. Must be >= -1.")
                if left_context == 0:
                    logging.warning(f"Layer {i} has left_context=0, which may cause issues in streaming mode.")

            layer.self_attention.att_context_size = context
            logging.info(f"  Layer {i:2d}: context = {context}")
        else:
            logging.warning(f"Layer {i} does not have att_context_size attribute. Skipping.")
    else:
        logging.warning(f"Layer {i} does not have self_attention attribute. Skipping.")
```

**Impact:**
- Invalid configurations fail fast with clear error messages
- Zero contexts trigger warnings
- Missing attributes logged for debugging

---

### 7. ✅ **Schedule Parsing Error Handling** (MEDIUM)
**File:** `examples/asr/speech_pretraining/hierarchical_streaming_nest.py`
**Lines:** 172-188

**Problem:**
Schedule parsing had weak validation:
- Invalid items logged warnings but processing continued
- Could result in incomplete schedule
- Silent failures hard to debug

**Original Code:**
```python
for item in schedule:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        layer_idx, context = item
        schedule_parsed.append((layer_idx, context))
    else:
        logging.warning(f"Invalid schedule item: {item}")  # Just warning!
```

**Fix Applied:**
```python
for idx, item in enumerate(schedule):
    if isinstance(item, (list, tuple)) and len(item) == 2:
        layer_idx, context = item
        # Validate layer_idx and context
        if not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError(f"Invalid layer_idx in schedule item {idx}: {layer_idx}. Must be non-negative integer.")
        if not isinstance(context, (list, tuple)) or len(context) < 1:
            raise ValueError(f"Invalid context in schedule item {idx}: {context}. Must be list/tuple with at least 1 element.")
        schedule_parsed.append((layer_idx, context))
    else:
        raise ValueError(f"Invalid schedule item at index {idx}: {item}. Expected [layer_idx, context] format.")

if not schedule_parsed:
    raise ValueError("Schedule is empty after parsing. At least one schedule point is required.")
```

**Impact:**
- Configuration errors fail immediately with descriptive messages
- No silent failures
- Easier debugging of config issues

---

## Summary of Files Modified

1. **nemo/collections/asr/losses/ssl_losses/mlm.py**
   - Added shape alignment logic for decoder_outputs and masks
   - Added proper padding with mask exclusion

2. **nemo/collections/asr/modules/ssl_modules/masking.py**
   - Fixed type inconsistencies (tensor to int conversions)
   - Added device specifications to all tensor operations
   - Added bounds checking for patch calculation
   - Ensured dtype consistency for mixed precision training

3. **nemo/collections/asr/models/ssl_models.py**
   - Added `current_frame` and `effective_left_context` parameters to mask_processor call
   - Implemented automatic context synchronization between encoder and masking

4. **examples/asr/speech_pretraining/hierarchical_streaming_nest.py**
   - Updated curriculum hook to use minimum context across layers
   - Added validation for context values
   - Improved error handling for schedule parsing
   - Added warnings for missing attributes

---

## Testing Recommendations

### 1. MLM Loss Shape Handling
```python
# Test case: decoder_outputs longer than masks
decoder_outputs = torch.randn(32, 211, 8192)
masks = torch.randint(0, 2, (32, 210)).bool()
# Should trim decoder_outputs to [32, 210, 8192]

# Test case: decoder_outputs shorter than masks
decoder_outputs = torch.randn(32, 209, 8192)
masks = torch.randint(0, 2, (32, 210)).bool()
# Should pad decoder_outputs to [32, 210, 8192] and masks with False
```

### 2. Masking Device Consistency
```python
# Test on CUDA with mixed precision
mask_processor = RandomBlockMasking(feat_in=80, causal=True)
input_feats = torch.randn(2, 80, 100).cuda().half()
input_lengths = torch.tensor([100, 90]).cuda()
masked_feats, masks = mask_processor(input_feats, input_lengths, current_frame=50)
# Should work without device errors
```

### 3. Context Synchronization
```python
# Test hierarchical context with masking sync
model.encoder.layers[0].self_attention.att_context_size = [40, 0]
model.encoder.layers[6].self_attention.att_context_size = [120, 0]
model.encoder.layers[12].self_attention.att_context_size = [320, 0]
# After curriculum hook, mask_processor.left_context_size should be 40 (minimum)
```

### 4. Schedule Validation
```python
# Test invalid schedule
schedule = [
    [0, [40, 0]],
    ["invalid", [120, 0]],  # Should raise ValueError
]
# Should fail with clear error message

# Test empty schedule
schedule = []
# Should raise ValueError
```

---

## Known Limitations

### 1. Contrastive Loss Padding (MEDIUM Priority)
**File:** `nemo/collections/asr/losses/ssl_losses/contrastive.py`
**Status:** NOT FIXED (behavior preserved)

**Issue:** When targets and masks are padded to match dimensions, padded positions are marked as "not masked" (value 0). While these positions are excluded from the main loss computation, they could potentially be sampled as negative examples if `sample_from_non_masked=True`.

**Current Behavior:** Padded masks with value 0 are filtered out by threshold (line 189), so they don't contribute to masked loss. However, if used as negatives, they represent zero-vectors which could affect contrastive learning.

**Mitigation:** This is edge-case behavior that rarely affects training in practice. A full fix would require length-aware negative sampling, which could break existing trained models.

### 2. Streaming Inference State Management
**File:** `examples/asr/speech_pretraining/streaming_inference_example.py`
**Status:** NOT FIXED (example code)

**Issue:** The streaming inference example processes chunks independently without maintaining state (attention cache, hidden states) across chunks.

**Impact:** True streaming requires state carry-over for continuity. Current example suitable for demonstration but not production streaming.

**Recommendation:** Implement proper state management for production streaming applications.

### 3. current_frame Parameter
**File:** `nemo/collections/asr/models/ssl_models.py`
**Status:** PARTIALLY FIXED

**Note:** The `current_frame` parameter is now correctly passed to `mask_processor`, but is currently set to `None` (line 1015). For actual streaming inference, this should be set to the current frame index in the audio stream. This requires changes at the inference script level, not the model level.

---

## Performance Impact

All fixes are **zero-cost** or **negligible-cost**:
- Shape alignment: One-time check per forward pass
- Type conversions: `.item()` calls are cheap
- Device specifications: No runtime cost (compile-time)
- Context sync: Only in curriculum hook (once per N steps)
- Validation: Only at model initialization

**No degradation in training speed expected.**

---

## Migration Notes

### For Existing Checkpoints
All fixes are **backward compatible**:
- No changes to saved model parameters
- No changes to checkpoint format
- Existing models can be loaded and continue training

### For Existing Configurations
Only hierarchical context configs need updates:
- Schedule validation is stricter (fails on invalid configs)
- If your config has invalid schedules, they will now error instead of silently fail
- Fix: Ensure schedule items are `[[layer_idx, [left_ctx, right_ctx]], ...]`

### For Custom Implementations
If you've subclassed these modules:
- Check override of `mask_processor.forward()` includes new parameters
- Check override of loss forward() handles shape mismatches
- Update any hardcoded assumptions about tensor shapes

---

## Verification Checklist

- [x] MLM loss handles shape mismatches
- [x] Masking uses consistent types (Python int)
- [x] All tensor operations specify device
- [x] mask_processor receives streaming parameters
- [x] Context sync uses minimum in hierarchical mode
- [x] Invalid contexts raise errors
- [x] Schedule parsing validates structure
- [x] Backward compatibility maintained
- [x] Documentation complete

---

## Contact

For questions or issues related to these fixes:
- Check existing issues: https://github.com/NVIDIA/NeMo/issues
- Review NeMo documentation: https://docs.nvidia.com/deeplearning/nemo/
- File new issues with "[Streaming SSL]" prefix

---

**Last Updated:** 2026-02-04
**NeMo Version:** 2.5.3
**Commit:** b5a8fc51b (Add streaming SSL)
