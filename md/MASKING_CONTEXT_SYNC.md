# Synchronizing Masking Context with Dynamic Attention Context

## ⚠️ The Issue

When using dynamic `att_context_size`, the masking `left_context_size` should also be updated!

### Problem Scenario

```python
# Set attention context to 160 frames (1.6s)
model.encoder.att_context_size = [160, 0]

# But masking still uses unlimited context
model.mask_processor.left_context_size = -1  # ❌ Inconsistent!

# Result: Masking can mask frames beyond attention window
# Frame 180 gets masked, but attention can only see up to frame 160
# This creates train-inference mismatch!
```

### Why This Matters

1. **Training consistency**: Masking should respect attention constraints
2. **Computational efficiency**: No point masking beyond attention window
3. **Better learning**: Model should only predict what it can attend to

---

## ✅ The Solution

### Approach 1: Sync Manually (Simple)

```python
def sync_masking_to_attention(model):
    """Sync masking context with attention context"""
    att_context = model.encoder.att_context_size[0]  # Left context

    if att_context == -1:
        # Unlimited attention → unlimited masking
        model.mask_processor.left_context_size = -1
    else:
        # Limited attention → limit masking to same or slightly less
        model.mask_processor.left_context_size = att_context

    print(f"Synced: att_context={att_context}, mask_context={model.mask_processor.left_context_size}")

# Usage
model.encoder.att_context_size = [160, 0]
sync_masking_to_attention(model)
```

### Approach 2: Automatic Sync (Better)

Modify the model to auto-sync when attention context changes.

---

## 🔧 Implementation: Auto-Sync Wrapper

```python
class DynamicContextModel:
    """
    Wrapper that automatically syncs masking context with attention context
    """

    def __init__(self, model):
        self.model = model
        self._att_context_size = model.encoder.att_context_size

    @property
    def att_context_size(self):
        return self._att_context_size

    @att_context_size.setter
    def att_context_size(self, value):
        """Set attention context and auto-sync masking"""
        self._att_context_size = value
        self.model.encoder.att_context_size = value

        # Auto-sync masking
        left_context = value[0] if isinstance(value, (list, tuple)) else value

        if hasattr(self.model, 'mask_processor'):
            if left_context == -1:
                self.model.mask_processor.left_context_size = -1
            else:
                # Match attention context (or slightly less for safety)
                self.model.mask_processor.left_context_size = left_context

            print(f"✓ Synced contexts: attention={value}, masking={self.model.mask_processor.left_context_size}")

        # Also sync pre_encoder mask if using post_conv masking
        if hasattr(self.model, 'pre_encoder') and self.model.pre_encoder is not None:
            if hasattr(self.model.pre_encoder, 'masking'):
                if left_context == -1:
                    self.model.pre_encoder.masking.left_context_size = -1
                else:
                    self.model.pre_encoder.masking.left_context_size = left_context

# Usage
model = EncDecDenoiseMaskedTokenPredModel.restore_from("model.nemo")
dynamic_model = DynamicContextModel(model)

# Now setting attention auto-syncs masking
dynamic_model.att_context_size = [160, 0]
# Output: ✓ Synced contexts: attention=[160, 0], masking=160

dynamic_model.att_context_size = [-1, 0]
# Output: ✓ Synced contexts: attention=[-1, 0], masking=-1
```

---

## 🎯 Updated Training Config

```yaml
# train with multiple contexts - masking will auto-adapt during training
encoder:
  att_context_size: [
    [80, 0],
    [160, 0],
    [320, 0],
    [-1, 0],
  ]

masking:
  causal: true
  # Don't set left_context_size here - it should follow att_context_size
  # During training, NeMo samples att_context, masking should respect it
  left_context_size: -1  # Default, will be overridden per forward pass
```

---

## 🔧 Fix: Modified Masking Forward

Update the masking forward to respect encoder's current context:

```python
# In masking.py - add parameter to forward_causal
def forward_causal(
    self,
    input_feats: torch.Tensor,
    input_lengths: torch.Tensor,
    current_frame: Optional[int] = None,
    effective_left_context: Optional[int] = None  # NEW: override left_context_size
):
    """
    Args:
        effective_left_context: If provided, overrides self.left_context_size
                                Useful for syncing with attention context
    """
    # Use effective context if provided, otherwise use configured
    left_context = effective_left_context if effective_left_context is not None else self.left_context_size

    # ... rest of masking logic using left_context
```

---

## 📝 Updated Model Forward

```python
# In ssl_models.py - EncDecDenoiseMaskedTokenPredModel.forward()

def forward(self, input_signal, input_signal_length, apply_mask=False):
    # ... preprocessing ...

    if apply_mask:
        # Get current attention context
        att_left_context = self.encoder.att_context_size[0]

        # Sync masking context
        if att_left_context != -1:
            # Use same context as attention (or slightly less)
            effective_mask_context = att_left_context
        else:
            # Unlimited attention → unlimited masking
            effective_mask_context = None

        # Apply masking with synced context
        if self.pre_encoder is not None:
            # Post-conv masking
            self.pre_encoder.set_masking_enabled(apply_mask=True)
            # Set effective context on pre_encoder masking
            self.pre_encoder.masking.left_context_size = effective_mask_context or -1
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            # ...
        else:
            # Pre-conv masking
            masked_signal, masks = self.mask_processor(
                input_feats=processed_noisy_input_signal,
                input_lengths=processed_noisy_input_signal_length,
                current_frame=None,  # Training mode
                effective_left_context=effective_mask_context  # NEW: sync with attention
            )
    # ...
```

---

## 🚀 Complete Solution: Enhanced Dynamic Context

```python
class EnhancedDynamicContextModel:
    """
    Full solution: Dynamic attention + auto-synced masking
    """

    def __init__(self, model):
        self.model = model
        self._current_context = model.encoder.att_context_size

        # Store original contexts for later use
        self.context_presets = {
            'very_low': [80, 0],
            'low': [160, 0],
            'medium': [240, 0],
            'high': [320, 0],
            'unlimited': [-1, 0],
        }

    def set_context(self, mode_or_context):
        """
        Set attention context and auto-sync masking

        Args:
            mode_or_context: Either a preset name ('low', 'medium', etc.)
                           or explicit context like [160, 0]
        """
        if isinstance(mode_or_context, str):
            # Use preset
            if mode_or_context not in self.context_presets:
                raise ValueError(f"Unknown mode: {mode_or_context}")
            context = self.context_presets[mode_or_context]
        else:
            # Use explicit context
            context = mode_or_context

        # Update attention context
        self.model.encoder.att_context_size = context
        self._current_context = context

        # Sync masking context
        self._sync_masking_context(context[0])

        print(f"✓ Context set to {context}")
        print(f"  Attention: {context}")
        print(f"  Masking: {self._get_masking_context()}")

    def _sync_masking_context(self, left_context):
        """Sync masking context with attention context"""
        # Handle different masking configurations
        if hasattr(self.model, 'mask_processor'):
            # Pre-conv masking
            if left_context == -1:
                self.model.mask_processor.left_context_size = -1
            else:
                self.model.mask_processor.left_context_size = left_context

        if hasattr(self.model, 'pre_encoder') and self.model.pre_encoder is not None:
            # Post-conv masking
            if hasattr(self.model.pre_encoder, 'masking'):
                if left_context == -1:
                    self.model.pre_encoder.masking.left_context_size = -1
                else:
                    # Account for subsampling if post-conv
                    subsampling_factor = getattr(self.model.encoder, 'subsampling_factor', 8)
                    self.model.pre_encoder.masking.left_context_size = left_context * subsampling_factor

    def _get_masking_context(self):
        """Get current masking context"""
        if hasattr(self.model, 'mask_processor'):
            return self.model.mask_processor.left_context_size
        elif hasattr(self.model, 'pre_encoder') and self.model.pre_encoder is not None:
            if hasattr(self.model.pre_encoder, 'masking'):
                return self.model.pre_encoder.masking.left_context_size
        return None

    def process(self, audio, lengths):
        """Process with synced contexts"""
        return self.model.forward(
            input_signal=audio,
            input_signal_length=lengths,
            apply_mask=False  # Inference mode
        )

# Usage
model = EncDecDenoiseMaskedTokenPredModel.restore_from("model.nemo")
dynamic_model = EnhancedDynamicContextModel(model)

# Set context - masking auto-syncs
dynamic_model.set_context('low')
# ✓ Context set to [160, 0]
#   Attention: [160, 0]
#   Masking: 160

# Process audio
output = dynamic_model.process(audio, lengths)
```

---

## 📊 Context Sync Rules

| Attention Context | Masking Context | Rationale |
|-------------------|-----------------|-----------|
| `[80, 0]` | `80` | Match exactly |
| `[160, 0]` | `160` | Match exactly |
| `[320, 0]` | `320` | Match exactly |
| `[-1, 0]` | `-1` | Both unlimited |

**Rule**: Masking context should **match** attention context to ensure consistency.

---

## ⚠️ Important Notes

### During Training

With multi-context training, NeMo samples different `att_context_size` per batch:

```python
# Batch 1: att_context = [80, 0]
# → Masking should respect 80-frame window

# Batch 2: att_context = [320, 0]
# → Masking should respect 320-frame window

# Currently NOT auto-synced in training!
# This is a limitation we need to address
```

### Current Training Behavior

```yaml
# Config says:
att_context_size: [[80,0], [160,0], [320,0]]
masking:
  left_context_size: -1  # Fixed at unlimited

# Problem: Masking uses unlimited context even when attention is limited!
```

### Proper Training Behavior (Desired)

We need NeMo to:
1. Sample `att_context_size` for current batch
2. Automatically set masking `left_context_size` to match
3. Forward pass uses consistent contexts

---

## 🔧 Training Fix: Custom Collate Function

```python
from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch

def create_synced_context_collate(base_collate_fn, model, context_options, context_probs):
    """
    Wrapper that syncs masking context with sampled attention context
    """
    import random

    def collate_with_sync(*args, **kwargs):
        # Get batch using base collate
        batch = base_collate_fn(*args, **kwargs)

        # Sample attention context for this batch
        sampled_context = random.choices(context_options, weights=context_probs)[0]

        # Set attention context
        model.encoder.att_context_size = sampled_context

        # Sync masking context
        left_context = sampled_context[0]
        if hasattr(model, 'mask_processor'):
            model.mask_processor.left_context_size = left_context

        return batch

    return collate_with_sync

# Use in dataloader setup
context_options = [[80,0], [160,0], [320,0], [-1,0]]
context_probs = [0.25, 0.25, 0.25, 0.25]

train_dl = ... # your dataloader
original_collate = train_dl.collate_fn
synced_collate = create_synced_context_collate(
    original_collate, model, context_options, context_probs
)
train_dl.collate_fn = synced_collate
```

---

## ✅ Recommended Implementation

### 1. Update Masking Module

Add `effective_left_context` parameter to `forward_causal()`:

```python
# In masking.py line ~191
def forward_causal(
    self,
    input_feats: torch.Tensor,
    input_lengths: torch.Tensor,
    current_frame: Optional[int] = None,
    effective_left_context: Optional[int] = None,  # NEW
):
    # Use effective if provided, else use self.left_context_size
    left_context = effective_left_context if effective_left_context is not None else self.left_context_size

    # ... rest of logic using left_context ...
```

### 2. Update Model Forward

Sync contexts in model forward pass:

```python
# In ssl_models.py - forward()
if apply_mask:
    att_left_context = self.encoder.att_context_size[0]
    effective_mask_context = att_left_context if att_left_context != -1 else None

    masked_signal, masks = self.mask_processor(
        input_feats=...,
        input_lengths=...,
        effective_left_context=effective_mask_context  # NEW
    )
```

### 3. Use Wrapper for Inference

```python
dynamic_model = EnhancedDynamicContextModel(model)
dynamic_model.set_context('low')  # Auto-syncs everything
```

---

## 📝 Summary

**Problem**: Masking context doesn't auto-sync with dynamic attention context

**Impact**:
- ❌ Inconsistent training (masking beyond attention window)
- ❌ Computational waste
- ❌ Potential train-inference mismatch

**Solution**:
1. ✅ Add `effective_left_context` parameter to masking
2. ✅ Sync in model forward pass
3. ✅ Use wrapper for clean inference API

**Status**: Implementation needed (see code above)

---

## 🚀 Quick Fix for Current Usage

Until full implementation, manually sync:

```python
def set_dynamic_context(model, context):
    """Helper to set context and sync masking"""
    model.encoder.att_context_size = context

    left = context[0] if isinstance(context, list) else context
    if hasattr(model, 'mask_processor'):
        model.mask_processor.left_context_size = left
    if hasattr(model, 'pre_encoder') and model.pre_encoder:
        if hasattr(model.pre_encoder, 'masking'):
            model.pre_encoder.masking.left_context_size = left

    print(f"Context synced: att={context}, mask={left}")

# Usage
set_dynamic_context(model, [160, 0])
```
