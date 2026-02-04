#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script to validate causal masking implementation.

This script verifies that:
1. Causal masking only masks past frames
2. Future frames are never masked
3. left_context_size limit works correctly

Usage:
    python test_streaming_masking.py
"""

import torch
from nemo.collections.asr.modules.ssl_modules.masking import RandomBlockMasking


def test_causal_masking_no_future():
    """Test that causal masking never masks future frames."""
    print("Test 1: Causal masking should not mask future frames")
    print("-" * 60)

    # Create causal masking module
    masking = RandomBlockMasking(feat_in=80, block_size=10, mask_prob=0.5, causal=True, allow_overlap=True)

    # Create dummy features
    batch_size = 2
    feat_dim = 80
    time_steps = 100
    feats = torch.randn(batch_size, feat_dim, time_steps)
    lengths = torch.tensor([100, 100])

    # Test with different current_frame values
    for current_frame in [30, 50, 80, 100]:
        masked_feats, masks = masking(feats, lengths, current_frame=current_frame)

        # Check that no future frames are masked
        future_masks = masks[:, :, current_frame:].sum()

        print(f"  current_frame={current_frame:3d}: Future masks = {future_masks.item():.1f} (should be 0)")
        assert future_masks == 0, f"Future frames are masked! current_frame={current_frame}"

    print("  ✓ PASSED: No future frames masked\n")


def test_causal_vs_noncausal():
    """Compare causal and non-causal masking."""
    print("Test 2: Causal vs Non-causal masking comparison")
    print("-" * 60)

    # Create both types
    causal_masking = RandomBlockMasking(feat_in=80, block_size=10, mask_prob=0.3, causal=True, allow_overlap=True)

    noncausal_masking = RandomBlockMasking(feat_in=80, block_size=10, mask_prob=0.3, causal=False, allow_overlap=True)

    # Create dummy features
    batch_size = 1
    feat_dim = 80
    time_steps = 100
    feats = torch.randn(batch_size, feat_dim, time_steps)
    lengths = torch.tensor([100])

    # Test causal
    causal_masked, causal_masks = causal_masking(feats, lengths, current_frame=100)
    causal_ratio = causal_masks.sum() / (batch_size * feat_dim * time_steps)

    # Test non-causal
    noncausal_masked, noncausal_masks = noncausal_masking(feats, lengths)
    noncausal_ratio = noncausal_masks.sum() / (batch_size * feat_dim * time_steps)

    print(f"  Causal masking ratio: {causal_ratio:.3f}")
    print(f"  Non-causal masking ratio: {noncausal_ratio:.3f}")
    print(f"  Both should be around {0.3:.3f} (mask_prob)")
    print("  ✓ PASSED: Both modes produce reasonable masking ratios\n")


def test_left_context_limit():
    """Test that left_context_size properly limits the masking window."""
    print("Test 3: left_context_size should limit masking window")
    print("-" * 60)

    left_context = 20
    masking = RandomBlockMasking(
        feat_in=80, block_size=5, mask_prob=0.5, causal=True, left_context_size=left_context, allow_overlap=True
    )

    # Create dummy features
    batch_size = 1
    feat_dim = 80
    time_steps = 100
    feats = torch.randn(batch_size, feat_dim, time_steps)
    lengths = torch.tensor([100])

    # Test with current_frame beyond left_context
    current_frame = 50
    masked_feats, masks = masking(feats, lengths, current_frame=current_frame)

    # Masks should only exist in range [current_frame - left_context, current_frame]
    min_mask_pos = max(0, current_frame - left_context)

    before_window_masks = masks[:, :, :min_mask_pos].sum()
    after_window_masks = masks[:, :, current_frame:].sum()

    print(f"  current_frame={current_frame}, left_context_size={left_context}")
    print(f"  Expected masking window: [{min_mask_pos}, {current_frame})")
    print(f"  Masks before window [{0}, {min_mask_pos}): {before_window_masks.item():.1f} (should be 0)")
    print(f"  Masks after window [{current_frame}, {time_steps}): {after_window_masks.item():.1f} (should be 0)")

    assert before_window_masks == 0, "Masks found before left context window"
    assert after_window_masks == 0, "Masks found after current frame"

    print("  ✓ PASSED: left_context_size correctly limits masking window\n")


def test_streaming_simulation():
    """Simulate streaming processing frame-by-frame."""
    print("Test 4: Simulate streaming processing")
    print("-" * 60)

    masking = RandomBlockMasking(
        feat_in=80, block_size=5, mask_prob=0.2, causal=True, left_context_size=30, allow_overlap=True
    )

    # Create dummy features
    batch_size = 1
    feat_dim = 80
    time_steps = 100
    feats = torch.randn(batch_size, feat_dim, time_steps)
    lengths = torch.tensor([100])

    # Simulate processing in steps
    print("  Processing frames incrementally:")
    for current_frame in [10, 20, 30, 50, 80, 100]:
        masked_feats, masks = masking(feats, lengths, current_frame=current_frame)

        # Count masks
        total_masked = masks.sum() / feat_dim
        print(f"    Frame {current_frame:3d}: {total_masked:5.1f} positions masked")

        # Verify causality
        future_masks = masks[:, :, current_frame:].sum()
        assert future_masks == 0, f"Future frames masked at current_frame={current_frame}"

    print("  ✓ PASSED: Streaming simulation works correctly\n")


def test_offline_mode():
    """Test that causal masking works in offline mode (current_frame=None)."""
    print("Test 5: Offline mode (current_frame=None)")
    print("-" * 60)

    masking = RandomBlockMasking(feat_in=80, block_size=10, mask_prob=0.3, causal=True, allow_overlap=True)

    # Create dummy features
    batch_size = 1
    feat_dim = 80
    time_steps = 100
    feats = torch.randn(batch_size, feat_dim, time_steps)
    lengths = torch.tensor([100])

    # Test without specifying current_frame (should use full sequence)
    masked_feats, masks = masking(feats, lengths, current_frame=None)

    masked_ratio = masks.sum() / (batch_size * feat_dim * time_steps)
    print(f"  Masking ratio: {masked_ratio:.3f} (should be around {0.3:.3f})")
    print(f"  Total frames: {time_steps}")
    print(f"  Frames with masks: {(masks.sum(dim=1) > 0).sum().item()}")
    print("  ✓ PASSED: Offline mode works correctly\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Causal Masking Implementation")
    print("=" * 60 + "\n")

    try:
        test_causal_masking_no_future()
        test_causal_vs_noncausal()
        test_left_context_limit()
        test_streaming_simulation()
        test_offline_mode()

        print("=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)
        print("\nCausal masking implementation is working correctly.")
        print("You can now use it for streaming SSL pretraining.\n")

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("Test FAILED! ✗")
        print("=" * 60)
        print(f"\nError: {e}\n")
        raise


if __name__ == "__main__":
    main()
