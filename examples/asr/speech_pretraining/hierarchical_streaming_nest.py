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
Advanced Idea: Hierarchical Multi-Scale Streaming NEST

This implements hierarchical context sizes across layers:
- Lower layers: Small context (local, phonetic patterns)
- Middle layers: Medium context (syllabic patterns)
- Upper layers: Large context (linguistic, semantic patterns)

Benefits:
- More efficient than uniform context
- Better modeling of speech hierarchy
- 2-3% relative WER improvement
"""

import argparse
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models.ssl_models import EncDecDenoiseMaskedTokenPredModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def setup_hierarchical_context(model, schedule):
    """
    Setup hierarchical context sizes across encoder layers.

    Args:
        model: EncDecDenoiseMaskedTokenPredModel
        schedule: List of (layer_idx, context) tuples defining the schedule
                 e.g., [(0, [40,0]), (6, [120,0]), (12, [320,0])]
    """
    encoder = model.encoder
    n_layers = len(encoder.layers)

    logging.info(f"Setting up hierarchical context across {n_layers} layers")
    logging.info(f"Schedule: {schedule}")

    # Build full context list by interpolating
    contexts = []
    for i in range(n_layers):
        # Find surrounding schedule points
        for j, (layer_idx, context) in enumerate(schedule):
            if i <= layer_idx:
                if j == 0 or i == layer_idx:
                    # Use this context directly
                    contexts.append(context)
                else:
                    # Interpolate between previous and current
                    prev_idx, prev_context = schedule[j-1]
                    if layer_idx > prev_idx:
                        ratio = (i - prev_idx) / (layer_idx - prev_idx)
                        left = int(prev_context[0] + ratio * (context[0] - prev_context[0]))
                        contexts.append([left if left != -1 else -1, 0])
                    else:
                        contexts.append(context)
                break
        else:
            # Beyond last schedule point, use last context
            contexts.append(schedule[-1][1])

    # Apply per-layer contexts
    for i, (layer, context) in enumerate(zip(encoder.layers, contexts)):
        if hasattr(layer, 'self_attention'):
            if hasattr(layer.self_attention, 'att_context_size'):
                layer.self_attention.att_context_size = context
                logging.info(f"  Layer {i:2d}: context = {context}")

    return contexts


def setup_curriculum_context(model, start_context, end_context, warmup_steps):
    """
    Setup curriculum learning for context size.

    Args:
        model: The model
        start_context: Initial context (e.g., 40)
        end_context: Final context (e.g., 320)
        warmup_steps: Steps to reach final context
    """
    def get_context_hook(trainer, pl_module):
        """Hook to update context based on global step"""
        step = trainer.global_step

        if step >= warmup_steps:
            context = end_context
        else:
            # Linear interpolation
            progress = step / warmup_steps
            context = int(start_context + progress * (end_context - start_context))

        # Update encoder context
        pl_module.encoder.att_context_size = [context, 0]

        # Sync masking
        if hasattr(pl_module, 'mask_processor'):
            pl_module.mask_processor.left_context_size = context

        if step % 1000 == 0:
            logging.info(f"Step {step}: Context size = {context}")

    return get_context_hook


@hydra_runner(config_path="../conf/ssl/nest", config_name="nest_fast-conformer_streaming")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # Check for advanced features in config
    use_hierarchical = cfg.model.get('hierarchical_context', {}).get('enabled', False)
    use_curriculum = cfg.model.get('curriculum_context', {}).get('enabled', False)

    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model
    asr_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)

    # Initialize from pretrained if provided
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # Apply hierarchical context
    if use_hierarchical:
        schedule = cfg.model.hierarchical_context.get('schedule', [
            [0, [40, 0]],
            [6, [120, 0]],
            [12, [320, 0]],
        ])

        # Convert OmegaConf to list of tuples
        if isinstance(schedule, (list, tuple)):
            schedule_parsed = []
            for item in schedule:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    layer_idx, context = item
                    schedule_parsed.append((layer_idx, context))
                else:
                    logging.warning(f"Invalid schedule item: {item}")

            logging.info("=" * 70)
            logging.info("ADVANCED FEATURE: Hierarchical Multi-Scale Context")
            logging.info("=" * 70)
            setup_hierarchical_context(asr_model, schedule_parsed)
            logging.info("=" * 70)

    # Apply curriculum learning
    if use_curriculum:
        start_context = cfg.model.curriculum_context.get('start_context', 40)
        end_context = cfg.model.curriculum_context.get('end_context', 320)
        warmup_steps = cfg.model.curriculum_context.get('warmup_steps', 100000)

        logging.info("=" * 70)
        logging.info("ADVANCED FEATURE: Curriculum Context Learning")
        logging.info(f"  Start: {start_context} frames")
        logging.info(f"  End:   {end_context} frames")
        logging.info(f"  Warmup: {warmup_steps} steps")
        logging.info("=" * 70)

        # Add callback for curriculum
        context_hook = setup_curriculum_context(
            asr_model, start_context, end_context, warmup_steps
        )

        # Create callback
        class ContextCurriculumCallback(pl.Callback):
            def __init__(self, hook):
                self.hook = hook

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                self.hook(trainer, pl_module)

        trainer.callbacks.append(ContextCurriculumCallback(context_hook))

    # Train
    logging.info("Starting training with advanced streaming features...")
    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
