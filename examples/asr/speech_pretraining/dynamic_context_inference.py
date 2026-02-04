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
Example: Dynamic Context Size for Adaptive Streaming

This script demonstrates how to use a model trained with multiple context sizes
and dynamically switch between them based on requirements.

Usage:
    python dynamic_context_inference.py \
        --model_path=/path/to/dynamic_streaming_nest.nemo \
        --audio_path=/path/to/audio.wav \
        --mode=auto  # or 'very_low', 'low', 'medium', 'high', 'unlimited'
"""

import argparse
import time
import torch
import numpy as np
from nemo.collections.asr.models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from nemo.collections.asr.parts.utils.audio_utils import AudioSegment


class AdaptiveContextInference:
    """
    Adaptive context size inference for streaming SSL models.

    Supports dynamic switching between context sizes based on:
    - Latency requirements
    - Computational budget
    - Audio quality
    """

    def __init__(self, model_path):
        """
        Args:
            model_path: Path to trained dynamic context model (.nemo)
        """
        print(f"Loading model from {model_path}...")
        self.model = EncDecDenoiseMaskedTokenPredModel.restore_from(model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Available context modes
        self.contexts = {
            'very_low': ([80, 0], "~20-30ms latency, 0.8s context"),
            'low': ([160, 0], "~40-50ms latency, 1.6s context"),
            'medium': ([240, 0], "~60-70ms latency, 2.4s context"),
            'high': ([320, 0], "~80-100ms latency, 3.2s context"),
            'unlimited': ([-1, 0], "Best accuracy, unlimited context"),
        }

        # Current mode
        self.current_mode = 'medium'
        self.set_mode('medium')

        print("Available modes:")
        for mode, (context, desc) in self.contexts.items():
            print(f"  {mode:12s}: {context} - {desc}")

    def set_mode(self, mode):
        """
        Set context mode.

        Args:
            mode: One of 'very_low', 'low', 'medium', 'high', 'unlimited'
        """
        if mode not in self.contexts:
            raise ValueError(f"Mode must be one of {list(self.contexts.keys())}")

        context, desc = self.contexts[mode]
        self.current_mode = mode
        self.model.encoder.att_context_size = context

        print(f"\n✓ Mode set to '{mode}': {context}")
        print(f"  {desc}")

    def auto_select_mode(self, latency_budget_ms=None, gpu_util=None, audio_snr_db=None):
        """
        Automatically select mode based on conditions.

        Priority: latency_budget > gpu_util > audio_snr

        Args:
            latency_budget_ms: Maximum allowed latency in milliseconds
            gpu_util: GPU utilization (0.0 to 1.0)
            audio_snr_db: Audio signal-to-noise ratio in dB
        """
        if latency_budget_ms is not None:
            print(f"\n⚡ Auto-selecting based on latency budget: {latency_budget_ms}ms")
            if latency_budget_ms < 35:
                self.set_mode('very_low')
            elif latency_budget_ms < 55:
                self.set_mode('low')
            elif latency_budget_ms < 75:
                self.set_mode('medium')
            elif latency_budget_ms < 105:
                self.set_mode('high')
            else:
                self.set_mode('unlimited')

        elif gpu_util is not None:
            print(f"\n🖥️  Auto-selecting based on GPU utilization: {gpu_util:.1%}")
            if gpu_util > 0.85:
                self.set_mode('very_low')
                print("  (High GPU load → reduce context)")
            elif gpu_util > 0.70:
                self.set_mode('low')
            elif gpu_util > 0.50:
                self.set_mode('medium')
            else:
                self.set_mode('high')
                print("  (Low GPU load → can use more context)")

        elif audio_snr_db is not None:
            print(f"\n🔊 Auto-selecting based on audio SNR: {audio_snr_db:.1f} dB")
            if audio_snr_db > 25:
                self.set_mode('low')
                print("  (Clean audio → less context needed)")
            elif audio_snr_db > 15:
                self.set_mode('medium')
            elif audio_snr_db > 10:
                self.set_mode('high')
                print("  (Noisy audio → more context helps)")
            else:
                self.set_mode('unlimited')
                print("  (Very noisy → use maximum context)")

    def process_audio(self, audio_path, chunk_size_sec=1.0):
        """
        Process audio file with current context setting.

        Args:
            audio_path: Path to audio file
            chunk_size_sec: Size of chunks for processing (seconds)

        Returns:
            results: Dict with processing results and timing
        """
        print(f"\n📂 Loading audio: {audio_path}")
        audio = AudioSegment.from_file(audio_path)

        audio_samples = torch.FloatTensor(audio.samples).unsqueeze(0)
        audio_len = torch.LongTensor([len(audio.samples)])

        if self.device == 'cuda':
            audio_samples = audio_samples.cuda()
            audio_len = audio_len.cuda()

        print(f"   Duration: {len(audio.samples) / audio.sample_rate:.2f}s")
        print(f"   Sample rate: {audio.sample_rate} Hz")

        # Process
        print(f"\n⚙️  Processing with mode '{self.current_mode}'...")
        start_time = time.time()

        with torch.no_grad():
            log_probs, encoded_len, masks, tokens = self.model.forward(
                input_signal=audio_samples, input_signal_length=audio_len, apply_mask=False
            )

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio.samples) / audio.sample_rate)

        results = {
            'mode': self.current_mode,
            'context': self.model.encoder.att_context_size,
            'audio_duration_sec': len(audio.samples) / audio.sample_rate,
            'processing_time_sec': elapsed,
            'rtf': rtf,
            'encoded_len': encoded_len.item(),
            'tokens_shape': tokens.shape,
            'log_probs_shape': log_probs.shape,
        }

        return results

    def benchmark_all_modes(self, audio_path):
        """
        Benchmark all context modes on the same audio.

        Args:
            audio_path: Path to audio file

        Returns:
            results: Dict mapping mode to results
        """
        print("\n" + "=" * 70)
        print("BENCHMARKING ALL MODES")
        print("=" * 70)

        all_results = {}

        for mode in ['very_low', 'low', 'medium', 'high', 'unlimited']:
            self.set_mode(mode)
            results = self.process_audio(audio_path)
            all_results[mode] = results

            print(f"\n📊 Results:")
            print(f"   Processing time: {results['processing_time_sec']:.3f}s")
            print(f"   Real-time factor: {results['rtf']:.3f}x")
            print(f"   Encoded length: {results['encoded_len']} frames")

        return all_results

    def print_benchmark_summary(self, all_results):
        """
        Print summary table of benchmark results.

        Args:
            all_results: Dict from benchmark_all_modes()
        """
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"\n{'Mode':<12} {'Context':<12} {'Time (s)':<10} {'RTF':<8} {'Latency Estimate'}")
        print("-" * 70)

        for mode in ['very_low', 'low', 'medium', 'high', 'unlimited']:
            res = all_results[mode]
            context_str = str(res['context'])
            time_str = f"{res['processing_time_sec']:.3f}"
            rtf_str = f"{res['rtf']:.3f}x"
            latency = f"~{res['processing_time_sec']*1000/res['audio_duration_sec']:.0f}ms"

            print(f"{mode:<12} {context_str:<12} {time_str:<10} {rtf_str:<8} {latency}")

        print("-" * 70)
        print("\nNotes:")
        print("- RTF = Real-Time Factor (< 1.0 is faster than real-time)")
        print("- Latency Estimate = Processing time per second of audio")
        print("- Actual latency includes I/O, buffering, etc.")


def main():
    parser = argparse.ArgumentParser(description="Dynamic context inference for streaming SSL")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .nemo model")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=['very_low', 'low', 'medium', 'high', 'unlimited', 'auto', 'benchmark'],
        help="Context mode or 'auto' for automatic, 'benchmark' to test all",
    )
    parser.add_argument(
        "--latency_budget_ms", type=int, default=None, help="Latency budget in ms (for auto mode)"
    )
    parser.add_argument(
        "--gpu_util", type=float, default=None, help="GPU utilization 0-1 (for auto mode)"
    )
    parser.add_argument("--audio_snr_db", type=float, default=None, help="Audio SNR in dB (for auto mode)")
    args = parser.parse_args()

    # Initialize
    inference = AdaptiveContextInference(args.model_path)

    # Select mode
    if args.mode == 'benchmark':
        # Benchmark all modes
        results = inference.benchmark_all_modes(args.audio_path)
        inference.print_benchmark_summary(results)

    elif args.mode == 'auto':
        # Auto-select based on conditions
        if args.latency_budget_ms or args.gpu_util or args.audio_snr_db:
            inference.auto_select_mode(
                latency_budget_ms=args.latency_budget_ms, gpu_util=args.gpu_util, audio_snr_db=args.audio_snr_db
            )
        else:
            print("\n⚠️  Auto mode requires one of: --latency_budget_ms, --gpu_util, --audio_snr_db")
            print("Defaulting to 'medium' mode")
            inference.set_mode('medium')

        results = inference.process_audio(args.audio_path)

        print(f"\n📊 Results:")
        print(f"   Mode: {results['mode']}")
        print(f"   Context: {results['context']}")
        print(f"   Processing time: {results['processing_time_sec']:.3f}s")
        print(f"   Real-time factor: {results['rtf']:.3f}x")
        print(f"   Encoded length: {results['encoded_len']} frames")

    else:
        # Use specified mode
        inference.set_mode(args.mode)
        results = inference.process_audio(args.audio_path)

        print(f"\n📊 Results:")
        print(f"   Processing time: {results['processing_time_sec']:.3f}s")
        print(f"   Real-time factor: {results['rtf']:.3f}x")
        print(f"   Encoded length: {results['encoded_len']} frames")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
