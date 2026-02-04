# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Utilities for streaming ASR models with dynamic context sizes.

from typing import Union, List, Tuple


def sync_masking_context_with_attention(model, att_context=None):
    """Sync masking left_context_size with encoder attention context."""
    if att_context is None:
        att_context = getattr(model.encoder, 'att_context_size', [-1, -1])
    
    left_context = att_context[0] if isinstance(att_context, (list, tuple)) else att_context
    
    if hasattr(model, 'mask_processor') and model.mask_processor is not None:
        if hasattr(model.mask_processor, 'left_context_size'):
            model.mask_processor.left_context_size = left_context
            return True
    return False


def set_dynamic_context(model, context, sync_masking=True):
    """Set attention context and optionally sync masking."""
    presets = {
        'very_low': [80, 0],
        'low': [160, 0],
        'medium': [240, 0],
        'high': [320, 0],
        'unlimited': [-1, 0],
    }
    
    resolved = presets[context] if isinstance(context, str) else context
    model.encoder.att_context_size = resolved
    
    if sync_masking:
        sync_masking_context_with_attention(model, resolved)
    
    return resolved
