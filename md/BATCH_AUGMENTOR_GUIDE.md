# 🎚️ Batch Augmentation in Streaming SSL Training

## 📋 Overview

`batch_augmentor` is a data augmentation technique that adds **noise and multi-speaker overlaps** to clean audio during training. This makes the SSL model more robust to real-world conditions.

## 🎯 Purpose

SSL models trained on clean audio often struggle with:
- ❌ Background noise (traffic, music, crowd)
- ❌ Multi-speaker scenarios (meetings, conversations)
- ❌ Varying acoustic conditions
- ❌ Real-world audio quality

**Solution**: Add synthetic noise and speaker overlaps during training!

---

## 🔧 Configuration in `nest_fast-conformer_streaming.yaml`

```yaml
train_ds:
  batch_augmentor:
    _target_: nemo.collections.asr.modules.ssl_modules.MultiSpeakerNoiseAugmentation
    prob: 0.5                    # 50% chance to augment each sample
    noise_ratio: 0.5             # 50% noise, 50% speech augmentation
    min_r_speech: -5.0           # Min SNR for speech aug (dB)
    max_r_speech: 5.0            # Max SNR for speech aug (dB)
    min_r_noise: -5.0            # Min SNR for noise aug (dB)
    max_r_noise: 20.0            # Max SNR for noise aug (dB)
    min_mix_rate: 0.5            # Min % of audio to augment
    max_mix_rate: 0.5            # Max % of audio to augment
    min_num_segments: 1          # Min number of noise segments
    max_num_segments: 1          # Max number of noise segments
    min_num_speakers: 1          # Min number of extra speakers
    max_num_speakers: 1          # Max number of extra speakers
```

---

## 🎲 The `prob` Parameter

### What is `prob`?

**`prob`** controls the **probability that augmentation is applied** to each sample in the batch.

```
prob: 0.5  →  50% of samples get augmented
              50% remain clean

prob: 1.0  →  100% of samples get augmented
              (always augment)

prob: 0.0  →  0% of samples get augmented
              (never augment, clean training)
```

### How It Works

```python
for each sample in batch:
    if random.random() > prob:
        # Skip augmentation for this sample
        continue
    else:
        # Apply augmentation
        add_noise_or_speakers()
```

### Visualization

```
Batch of 8 samples with prob=0.5:

Sample 0: random() = 0.23  < 0.5  → ✅ AUGMENT
Sample 1: random() = 0.67  > 0.5  → ❌ Skip (clean)
Sample 2: random() = 0.45  < 0.5  → ✅ AUGMENT
Sample 3: random() = 0.89  > 0.5  → ❌ Skip (clean)
Sample 4: random() = 0.12  < 0.5  → ✅ AUGMENT
Sample 5: random() = 0.78  > 0.5  → ❌ Skip (clean)
Sample 6: random() = 0.34  < 0.5  → ✅ AUGMENT
Sample 7: random() = 0.91  > 0.5  → ❌ Skip (clean)

Result: 4 augmented, 4 clean
        (approximately 50%)
```

---

## 🎭 Two Types of Augmentation

### 1. Noise Augmentation (`noise_ratio` probability)

Add actual noise from `noise_manifest`:

```
Clean Audio:      [Speech ~~~~~~~~~~~~~~~~]
                        +
Noise:            [Traffic noise ##########]
                        ↓
Noisy Audio:      [Speech + noise ~~~~~~~~##]
```

**Energy Ratio**: `min_r_noise` to `max_r_noise` (dB)
- Negative values: noise louder than speech
- Positive values: speech louder than noise

```
SNR (dB) = 10 * log10(Speech_Energy / Noise_Energy)

Example with r_noise = 20 dB:
    Speech is 100x louder than noise (very clean)

Example with r_noise = -5 dB:
    Noise is 3.16x louder than speech (very noisy!)
```

### 2. Speech Augmentation (`1 - noise_ratio` probability)

Add other speakers from the same batch:

```
Original (Speaker A):  [Hello, how are you?]
                              +
Speaker B (from batch): [I'm doing great!]
                              ↓
Mixed:                  [Hello + I'm doing, how + great are you?]
```

**Energy Ratio**: `min_r_speech` to `max_r_speech` (dB)

This simulates multi-speaker scenarios (meetings, overlapping speech).

---

## 📊 Parameter Breakdown

### `noise_ratio: 0.5`

Controls the split between noise and speech augmentation:

```
When augmentation is triggered:
├─ 50% chance → Add noise from noise_manifest
└─ 50% chance → Add other speakers from batch

noise_ratio: 0.0  →  Always add speakers (no noise)
noise_ratio: 1.0  →  Always add noise (no speakers)
noise_ratio: 0.5  →  50/50 split
```

### `min_r_speech` & `max_r_speech`

Signal-to-Noise Ratio (SNR) range for speech augmentation:

```yaml
min_r_speech: -5.0   # Overlapping speaker can be 3x louder
max_r_speech: 5.0    # Overlapping speaker can be 3x quieter
```

```
┌─────────────────────────────────────────┐
│  SNR Scale (dB)                         │
│                                         │
│  -5 dB  ─┼─  Other speaker louder      │
│   0 dB  ─┼─  Equal volume              │
│  +5 dB  ─┼─  Original speaker louder   │
└─────────────────────────────────────────┘

Random uniform sampling between [-5, 5]
```

### `min_r_noise` & `max_r_noise`

SNR range for noise augmentation:

```yaml
min_r_noise: -5.0    # Very noisy (noise 3x louder)
max_r_noise: 20.0    # Very clean (speech 100x louder)
```

```
┌─────────────────────────────────────────┐
│  SNR Scale (dB)                         │
│                                         │
│  -5 dB  ─┼─  Very noisy                │
│   0 dB  ─┼─  Equal energy              │
│  10 dB  ─┼─  Moderate noise            │
│  20 dB  ─┼─  Light background noise    │
└─────────────────────────────────────────┘

Wide range → Model learns robustness across conditions
```

### `min_mix_rate` & `max_mix_rate`

Percentage of audio duration to augment:

```yaml
min_mix_rate: 0.5    # Augment at least 50% of audio
max_mix_rate: 0.5    # Augment at most 50% of audio
```

```
Audio duration: 10 seconds
mix_rate: 0.5  →  5 seconds augmented

Timeline:
┌──────────────────────────┐
│ 0s    2.5s    5s    7.5s 10s
│ [Clean] [Noisy] [Clean]    ← Example with 5s augmented
└──────────────────────────┘

If min_mix_rate = max_mix_rate:
    Always augment exactly that proportion

If min_mix_rate < max_mix_rate:
    Random between the two values
```

### `min_num_segments` & `max_num_segments`

Number of discrete noise/speech segments:

```yaml
min_num_segments: 1
max_num_segments: 1
```

```
1 segment (current config):
┌──────────────────────────┐
│ Clean  [Noise segment]  Clean │
└──────────────────────────┘

3 segments (if max_num_segments=3):
┌──────────────────────────┐
│ [N1] Clean [N2] Clean [N3] │
└──────────────────────────┘

More segments → More complex augmentation
```

### `min_num_speakers` & `max_num_speakers`

Number of extra speakers to add (for speech augmentation):

```yaml
min_num_speakers: 1
max_num_speakers: 1
```

```
1 speaker:
Original: Speaker A
  + Add: Speaker B
  = 2 total speakers (overlapping)

3 speakers (if max_num_speakers=3):
Original: Speaker A
  + Add: Speakers B, C, D
  = 4 total speakers (meeting scenario)

More speakers → Simulates crowded environments
```

---

## 🔄 Full Augmentation Flow

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Load Batch                                      │
│   - Clean audio from train_ds.manifest_filepath        │
│   - Noise audio from train_ds.noise_manifest (if any)  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: For Each Sample in Batch                       │
│                                                         │
│   2a. Random Check:                                     │
│       random() < prob?                                  │
│         NO  → Keep clean, skip to next sample          │
│         YES → Continue to augmentation                  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Choose Augmentation Type                       │
│                                                         │
│   random() < noise_ratio?                              │
│     YES → Noise Augmentation (use noise_manifest)      │
│     NO  → Speech Augmentation (use batch samples)      │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Determine Mix Parameters                       │
│                                                         │
│   - mix_rate: random between [min_mix_rate, max_mix_rate]
│   - mix_len = audio_length * mix_rate                  │
│   - num_segments: random [min_num_segments, max_num_segments]
│   - num_speakers: random [min_num_speakers, max_num_speakers]
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: Noise Augmentation Path                        │
│                                                         │
│   5a. Get noise from noise_manifest                    │
│   5b. Repeat/trim noise to match mix_len              │
│   5c. Randomly select SNR from [min_r_noise, max_r_noise]
│   5d. Calculate scale factor:                          │
│       scale = sqrt(speech_energy / (10^(SNR/10) * noise_energy))
│   5e. Add scaled noise to audio                        │
└─────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 6: Speech Augmentation Path                       │
│                                                         │
│   6a. Select random speakers from batch                │
│   6b. Split mix_len into num_segments pieces           │
│   6c. For each segment, pick a speaker                 │
│   6d. Randomly select SNR from [min_r_speech, max_r_speech]
│   6e. Calculate scale factor                           │
│   6f. Add scaled speaker segments to audio             │
└─────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 7: Output                                          │
│                                                         │
│   Return batch with:                                    │
│     - audio: original clean audio                       │
│     - noise: computed noise signal                      │
│     - noisy_audio: audio + noise (for SSL training)    │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 Practical Examples

### Example 1: Current Config (Conservative)

```yaml
prob: 0.5
noise_ratio: 0.5
min_r_speech: -5.0
max_r_speech: 5.0
min_r_noise: -5.0
max_r_noise: 20.0
min_mix_rate: 0.5
max_mix_rate: 0.5
min_num_segments: 1
max_num_segments: 1
min_num_speakers: 1
max_num_speakers: 1
```

**Behavior**:
- 50% of samples augmented
- When augmented: 50% noise, 50% extra speaker
- Single continuous segment
- Moderate noise levels

**Use case**: General-purpose SSL training

### Example 2: Aggressive Augmentation

```yaml
prob: 1.0                    # Always augment
noise_ratio: 0.7             # Prefer noise over speakers
min_r_speech: -10.0          # Very loud overlapping speakers
max_r_speech: 10.0
min_r_noise: -10.0           # Very noisy conditions
max_r_noise: 15.0
min_mix_rate: 0.7            # Augment 70-100% of audio
max_mix_rate: 1.0
min_num_segments: 1          # 1-5 noise segments
max_num_segments: 5
min_num_speakers: 1          # 1-3 extra speakers
max_num_speakers: 3
```

**Behavior**:
- Every sample augmented
- Mostly noise (70%)
- Very noisy conditions
- Multiple noise segments
- Up to 3 overlapping speakers

**Use case**: Training for extremely challenging conditions (call centers, crowded spaces)

### Example 3: Clean Training (No Augmentation)

```yaml
prob: 0.0                    # Never augment
# Other params don't matter
```

**Behavior**:
- All samples remain clean
- No noise or speaker overlap

**Use case**: Baseline comparison or clean-data-only training

### Example 4: Multi-Speaker Focused

```yaml
prob: 0.8
noise_ratio: 0.2             # Prefer speakers over noise
min_r_speech: -8.0
max_r_speech: 8.0
min_r_noise: 10.0            # Light background noise only
max_r_noise: 25.0
min_mix_rate: 0.3
max_mix_rate: 0.7
min_num_segments: 2          # Multiple speech segments
max_num_segments: 4
min_num_speakers: 2          # Multiple speakers
max_num_speakers: 4
```

**Behavior**:
- 80% of samples augmented
- Mostly multi-speaker scenarios (80%)
- 2-4 overlapping speakers
- Multiple speech segments (conversations)

**Use case**: Training for meeting transcription

---

## 📈 Impact on Training

### With Augmentation (prob=0.5)

```
Training batch visualization:

Sample 0 (Clean):      [Speech ~~~~~~~~~~~~~~~~]
Sample 1 (Augmented):  [Speech + noise ~~~~####]
Sample 2 (Clean):      [Speech ~~~~~~~~~~~~~~~~]
Sample 3 (Augmented):  [Speech A + B ~~~~^^^^~~]
Sample 4 (Augmented):  [Speech + noise ####~~~~]
Sample 5 (Clean):      [Speech ~~~~~~~~~~~~~~~~]
Sample 6 (Augmented):  [Speech A + B + C ^^~~##]
Sample 7 (Clean):      [Speech ~~~~~~~~~~~~~~~~]

Mixed batch → Model learns from both clean and noisy
```

### Benefits

✅ **Robustness**: Model handles real-world noise
✅ **Generalization**: Better transfer to downstream tasks
✅ **Multi-speaker**: Handles overlapping speech
✅ **Variability**: Diverse training conditions

### Trade-offs

⚠️ **Training Time**: Augmentation adds ~5-10% overhead
⚠️ **Convergence**: May need more steps to converge
⚠️ **Clean Performance**: Might slightly reduce clean-data WER

---

## 🎯 Recommended Settings

### For Different Use Cases

| Use Case | prob | noise_ratio | SNR Range | mix_rate |
|----------|------|-------------|-----------|----------|
| **General ASR** | 0.5 | 0.5 | [-5, 20] dB | 0.5 |
| **Noisy Environments** | 0.8 | 0.8 | [-10, 10] dB | 0.7-1.0 |
| **Multi-speaker** | 0.7 | 0.2 | [-8, 8] dB | 0.4-0.7 |
| **Clean Speech Only** | 0.0 | - | - | - |
| **Balanced Robustness** | 0.6 | 0.5 | [-8, 15] dB | 0.5-0.8 |

---

## 🔍 Debugging Tips

### Check Augmentation is Working

```python
from nemo.collections.asr.data.ssl_dataset import AudioNoiseBatch
from nemo.collections.asr.modules.ssl_modules import MultiSpeakerNoiseAugmentation

# Create augmentor
augmentor = MultiSpeakerNoiseAugmentation(prob=1.0)  # Always augment for testing

# Check output
batch = load_batch()
augmented_batch = augmentor(batch)

# Verify augmentation
assert not torch.equal(batch.audio, augmented_batch.noisy_audio)
print(f"Noise added: {augmented_batch.noise.abs().max().item()}")
```

### Visualize Augmented Audio

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Original
axes[0].plot(batch.audio[0].cpu().numpy())
axes[0].set_title("Clean Audio")

# Noise
axes[1].plot(augmented_batch.noise[0].cpu().numpy())
axes[1].set_title("Added Noise/Speaker")

# Augmented
axes[2].plot(augmented_batch.noisy_audio[0].cpu().numpy())
axes[2].set_title("Noisy Audio (Clean + Noise)")

plt.tight_layout()
plt.savefig("augmentation_example.png")
```

---

## 🚀 Summary

```
┌──────────────────────────────────────────────────────────┐
│              Batch Augmentation Quick Reference          │
│                                                          │
│  prob: 0.5                                               │
│    → 50% of samples get augmented                       │
│                                                          │
│  noise_ratio: 0.5                                        │
│    → 50% noise, 50% extra speakers                      │
│                                                          │
│  min_r_noise: -5.0, max_r_noise: 20.0                   │
│    → Noise SNR varies from very noisy to clean          │
│                                                          │
│  min_r_speech: -5.0, max_r_speech: 5.0                  │
│    → Speaker overlap SNR varies around 0 dB             │
│                                                          │
│  mix_rate: 0.5                                           │
│    → Augment 50% of each audio's duration               │
│                                                          │
│  Result: Robust SSL representations!                     │
└──────────────────────────────────────────────────────────┘
```

---

## 📚 Related Files

- **Implementation**: `nemo/collections/asr/modules/ssl_modules/augmentation.py`
- **Config**: `examples/asr/conf/ssl/nest/nest_fast-conformer_streaming.yaml`
- **Dataset**: `nemo/collections/asr/data/ssl_dataset.py`

---

**Key Takeaway**: `prob` controls how often augmentation happens, while other parameters control what kind of augmentation is applied! 🎯
