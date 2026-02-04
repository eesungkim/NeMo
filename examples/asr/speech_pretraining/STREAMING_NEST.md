# Streaming NEST (Best-RQ) SSL Pretraining

This document describes the modifications made to enable **streaming/causal** SSL pretraining with NEST (Best-RQ).

## Overview

Standard NEST pretraining uses bidirectional context (can see future frames), which is not suitable for online/streaming ASR applications. The streaming version modifies the architecture to only use past context, enabling real-time processing.

## Key Modifications

### 1. Normalization: LayerNorm for Streaming

**⚠️ CRITICAL**: Use LayerNorm, not BatchNorm, for streaming models!

```yaml
encoder:
  conv_norm_type: 'layer_norm'  # ✓ Streaming-friendly
  # NOT 'batch_norm' - causes train-inference mismatch
```

**Why LayerNorm is Required**:
- **Batch-independent**: Normalizes each sample independently
- **Deterministic**: Same input → same output, always
- **Streaming-friendly**: No dependency on batch statistics
- **Consistent**: Training and inference behave identically

**Why NOT BatchNorm**:
- **Batch-dependent**: Statistics change with batch composition
- **Non-deterministic**: Same chunk can produce different outputs
- **Training-inference gap**: Full utterances (training) vs chunks (inference)
- **Inconsistent streaming**: Results vary based on what else is in the batch

**Performance**: LayerNorm may have 0-2% relative WER difference vs BatchNorm, but the consistency gain is worth it for production streaming.

### 2. Causal Masking (`RandomBlockMasking`)

**Location**: `nemo/collections/asr/modules/ssl_modules/masking.py`

**New Parameters**:
- `causal: bool = False` - Enable causal masking (only mask past context)
- `left_context_size: int = -1` - Maximum left context to mask (-1 for unlimited)

**How it works**:
- In causal mode, the masker only masks frames up to the current position
- Future frames are never masked, ensuring the model learns to operate without future context
- Optional `left_context_size` limits how far back masking can occur (useful for computational efficiency)

**Example configuration**:
```yaml
masking:
  _target_: nemo.collections.asr.modules.RandomBlockMasking
  block_size: 40
  mask_prob: 0.01
  feat_in: 80
  freeze: true
  allow_overlap: true
  causal: true  # Enable streaming mode
  left_context_size: -1  # Unlimited left context (or set to e.g., 320 for ~3.2s)
```

### 2. Causal Encoder Configuration

**Location**: `examples/asr/conf/ssl/nest/nest_fast-conformer_streaming.yaml`

**Key changes**:

#### a) Causal Self-Attention
```yaml
encoder:
  att_context_size: [-1, 0]  # Unlimited left, no right context
```
- `[-1, 0]` means unlimited past context, zero future context
- For limited lookahead: `[-1, 2]` allows 2 frames of future context
- For limited context window: `[320, 0]` uses only 320 frames (~3.2s) of past context

#### b) Causal Convolution
```yaml
encoder:
  conv_context_size: causal  # Only use left context in convolutions
```
- `causal` setting ensures convolutions only see past frames
- Equivalent to `[kernel_size-1, 0]` (e.g., `[8, 0]` for kernel_size=9)

#### c) Causal Downsampling
```yaml
encoder:
  causal_downsampling: true
```
- Ensures the initial convolutional downsampling is also causal

## Training with Streaming Mode

### Basic Training Command

```bash
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    --config-path=../conf/ssl/nest \
    --config-name=nest_fast-conformer_streaming \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    trainer.devices=4 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp"
```

### Switching from Non-streaming to Streaming

If you have a non-streaming checkpoint, you can continue training in streaming mode:

```bash
python examples/asr/speech_pretraining/masked_token_pred_pretrain.py \
    --config-path=../conf/ssl/nest \
    --config-name=nest_fast-conformer_streaming \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    +init_from_pretrained_model=/path/to/non_streaming_checkpoint.nemo \
    trainer.devices=4
```

## Configuration Variants

### 1. Fully Causal (Zero Lookahead)
Best for true streaming applications:
```yaml
encoder:
  att_context_size: [-1, 0]  # No future context
  conv_context_size: causal
  causal_downsampling: true
```

### 2. Limited Lookahead
Trade latency for performance (e.g., 40ms lookahead):
```yaml
encoder:
  att_context_size: [-1, 4]  # 4 frames = 40ms lookahead
  conv_context_size: causal
  causal_downsampling: true
```

### 3. Limited Context Window
For memory efficiency in very long sequences:
```yaml
encoder:
  att_context_size: [320, 0]  # 3.2s past context, no future
masking:
  left_context_size: 320  # Match encoder context
```

## Implementation Details

### Forward Pass Flow (Streaming Mode)

```python
# Training
1. Load batch → (clean_audio, noise_audio)
2. Apply MultiSpeakerNoiseAugmentation → (clean_audio, noisy_audio)
3. Preprocessor → (clean_spec, noisy_spec) [mel spectrograms]
4. Quantizer(clean_spec) → target_tokens [only past context used]
5. RandomBlockMasking(noisy_spec, causal=True) → (masked_noisy_spec, masks)
   # Only masks past frames, preserves causality
6. CausalEncoder(masked_noisy_spec) → encoded_features
   # Encoder only attends to past context
7. MultiSoftmaxDecoder(encoded_features) → log_probs
8. MultiMLMLoss(log_probs, target_tokens, masks) → loss
```

### Causal Masking Algorithm

```python
def forward_causal(input_feats, input_lengths, current_frame=None):
    """
    Args:
        current_frame: Current frame index for streaming inference
                      If None, uses full sequence length (offline training)
    """
    for each sequence:
        # Determine masking window
        if current_frame is not None:
            max_mask_pos = min(current_frame, input_length)
        else:
            max_mask_pos = input_length

        # Apply left context limit
        if left_context_size > 0:
            min_mask_pos = max(0, max_mask_pos - left_context_size)
        else:
            min_mask_pos = 0

        # Sample blocks only within [min_mask_pos, max_mask_pos]
        sample_blocks_in_range(min_mask_pos, max_mask_pos)
```

## Performance Considerations

### Accuracy Impact
- Fully causal models typically have **0.5-2% higher WER** compared to bidirectional
- Limited lookahead (40-80ms) can recover most of the gap
- The trade-off depends on your application's latency requirements

### Latency
- **Fully causal**: Algorithmic latency = only model processing time
- **Limited lookahead**: Algorithmic latency = lookahead + processing time
- Real-world latency also includes I/O, buffering, etc.

### Training Time
- Streaming and non-streaming models train at similar speeds
- Memory usage is similar or slightly lower with limited context windows

## Inference with Streaming Models

After pretraining, you can fine-tune the streaming encoder for downstream ASR tasks. The causal encoder can be used with streaming CTC/RNN-T decoders for real-time transcription.

### Example: Streaming ASR Fine-tuning

```python
# Load pretrained streaming encoder
from nemo.collections.asr.models import EncDecCTCModel

# Initialize with streaming pretrained encoder
asr_model = EncDecCTCModel.from_pretrained("path/to/streaming_nest.nemo")

# Fine-tune for streaming ASR
# The encoder is already causal, ready for streaming inference
```

## Validation

To verify causality, check that:
1. Attention masks are lower-triangular (no attention to future)
2. Convolution padding is asymmetric (only left padding)
3. Masking only affects past frames

## Recommended Configurations by Use Case

| Use Case | att_context_size | conv_context_size | Latency | Accuracy |
|----------|------------------|-------------------|---------|----------|
| Real-time transcription | `[-1, 0]` | `causal` | ~20ms | Baseline |
| Live captions (low latency) | `[-1, 2]` | `causal` | ~40ms | +0.5% rel |
| Live captions (standard) | `[-1, 4]` | `causal` | ~60ms | +1.0% rel |
| Near-real-time | `[-1, 8]` | `causal` | ~100ms | +1.5% rel |

*Accuracy improvements are relative to fully causal baseline*

## FAQ

**Q: Can I convert a non-streaming checkpoint to streaming?**
A: Not directly. The attention patterns and convolution weights learn different features. However, you can use it as initialization for streaming fine-tuning.

**Q: Does causal masking hurt pretraining performance?**
A: Slightly. The model has less context during pretraining, but this matches the streaming inference condition. The gap is usually recovered during task-specific fine-tuning.

**Q: What left_context_size should I use?**
A: Start with `-1` (unlimited). If memory is an issue or you want to enforce a maximum latency, set it to match your encoder's `att_context_size[0]`.

**Q: Can I use chunked attention for better efficiency?**
A: Yes! Set `att_context_style: chunked_limited` and use fixed context sizes like `att_context_size: [320, 0]` for chunk-based processing.

## References

- NEST Paper: [arXiv:2408.13106](https://arxiv.org/abs/2408.13106)
- Best-RQ Paper: [arXiv:2202.01855](https://arxiv.org/abs/2202.01855)
- FastConformer: [arXiv:2305.05084](https://arxiv.org/abs/2305.05084)
