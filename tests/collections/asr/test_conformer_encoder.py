# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import MALAMultiHeadAttention


class TestStochasticDepth:
    """Testing stochastic depth functionality."""

    def test_stochastic_depth_model_creation(self):
        """Testing basic model creation and the drop probs are correctly assigned."""
        n_layers = 4
        model = ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8)

        # checking that by default SD is disabled
        assert model.layer_drop_probs == [0.0] * n_layers

        # linear mode
        for drop_prob in [0.3, 0.5, 0.9]:
            for start_layer in [1, 3]:
                model = ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_start_layer=start_layer,
                )
                L = n_layers - start_layer
                assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob * l / L for l in range(1, L + 1)]

        # uniform mode
        for drop_prob in [0.3, 0.5, 0.9]:
            model = ConformerEncoder(
                feat_in=10,
                n_layers=n_layers,
                d_model=4,
                feat_out=8,
                stochastic_depth_drop_prob=drop_prob,
                stochastic_depth_mode="uniform",
                stochastic_depth_start_layer=start_layer,
            )
            L = n_layers - start_layer
            assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob] * L

        # checking for errors
        for drop_prob in [-1.0, 1.0]:
            with pytest.raises(ValueError, match="stochastic_depth_drop_prob has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_mode="uniform",
                )

        with pytest.raises(ValueError, match="stochastic_depth_mode has to be one of"):
            ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8, stochastic_depth_mode="weird")

        for start_layer in [-1, 0, 5]:
            with pytest.raises(ValueError, match="stochastic_depth_start_layer has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_start_layer=start_layer,
                )

    @pytest.mark.pleasefixme
    def test_stochastic_depth_forward(self):
        """Testing that forward works and we get randomness during training, but not during eval."""
        random_input = torch.rand((1, 2, 2))
        random_length = torch.tensor([2], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=2,
            n_layers=3,
            d_model=4,
            feat_out=4,
            stochastic_depth_drop_prob=0.8,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff > 0

        model.eval()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff == 0


class TestBypassPreEncode:
    """Testing bypass pre-encode functionality."""

    def test_bypass_pre_encode_forward(self):
        """Testing that forward works with "bypass pre-encode" mode."""
        # For pre-encoded embeddings, the shape is (batch_size, n_frames, emb_dim)
        batch_size = 2
        n_frames, emb_dim, feat_out = 17, 16, 8
        random_input = torch.rand((batch_size, n_frames, emb_dim))
        random_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=10,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

    def test_error_shape_invalid_bypass_pre_encode_forward(self):
        """
        Testing that error messages are correctly triggered regarding "bypass pre-encode" mode.
        Both correct samples and wrongs samples are tested.

        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)
        """
        batch_size = 2
        n_frames, emb_dim, feat_in, feat_out = 17, 16, 10, 8

        pre_encode_input = torch.rand((batch_size, n_frames, emb_dim))
        feat_input = torch.rand((batch_size, feat_in, n_frames))
        input_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=feat_in,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        sub_sampled_n_frames = np.ceil(n_frames / model.subsampling_factor)

        # Test with bypass_pre_encode = True, should be pre_encode_input but given feat_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        # Test with bypass_pre_encode = True, given the correct input pre_encode_input.
        model.train()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        # Test with bypass_pre_encode = False, should be feat_input but given pre_encode_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        # Test with bypass_pre_encode = False, given the correct input feat_input.
        model.train()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)


class TestMALA:
    def test_mala_attention_streaming_matches_full_causal(self):
        torch.manual_seed(0)
        batch_size, time, dim, n_head, chunk = 2, 24, 16, 4, 6

        attn = MALAMultiHeadAttention(
            n_head=n_head, n_feat=dim, dropout_rate=0.0, use_bias=True, lepe_kernel_size=1
        ).eval()

        x = torch.randn(batch_size, time, dim)
        full_mask = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        with torch.no_grad():
            full_out = attn(query=x, key=x, value=x, mask=full_mask)

            cache = torch.zeros(batch_size, attn.recurrent_cache_size, dim, dtype=x.dtype, device=x.device)
            stream_out = []
            for start in range(0, time, chunk):
                end = min(start + chunk, time)
                x_chunk = x[:, start:end, :]
                t_chunk = end - start
                chunk_mask = torch.triu(torch.ones(t_chunk, t_chunk, dtype=torch.bool), diagonal=1).unsqueeze(0)
                chunk_mask = chunk_mask.expand(batch_size, -1, -1)
                y_chunk, cache = attn(query=x_chunk, key=x_chunk, value=x_chunk, mask=chunk_mask, cache=cache)
                stream_out.append(y_chunk)

            stream_out = torch.cat(stream_out, dim=1)

        torch.testing.assert_close(stream_out, full_out, rtol=1e-4, atol=1e-4)

    def test_mala_conformer_encoder_forward_shape(self):
        model = ConformerEncoder(
            feat_in=80,
            n_layers=4,
            d_model=256,
            self_attention_model='mala',
            n_heads=4,
            subsampling=None,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            dropout_att=0.0,
            conv_kernel_size=3,
            conv_norm_type='layer_norm',
        ).eval()

        audio_signal = torch.randn(2, 80, 100)
        length = torch.tensor([100, 80], dtype=torch.int64)

        with torch.no_grad():
            out, out_len = model(audio_signal=audio_signal, length=length)

        assert out.shape == (2, 256, 100)
        torch.testing.assert_close(out_len, length)

    def test_mala_conformer_encoder_streaming_forward(self):
        model = ConformerEncoder(
            feat_in=80,
            n_layers=2,
            d_model=32,
            self_attention_model='mala',
            n_heads=4,
            subsampling=None,
            att_context_size=[-1, 0],
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            dropout_att=0.0,
            conv_kernel_size=1,
            conv_norm_type='layer_norm',
        ).eval()

        audio_signal = torch.randn(1, 80, 20)
        length = torch.tensor([20], dtype=torch.int64)
        cache_last_channel, cache_last_time, cache_last_channel_len = model.get_initial_cache_state(
            batch_size=1, dtype=audio_signal.dtype, device=audio_signal.device
        )

        with torch.no_grad():
            out, out_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len = model(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )

        assert out.shape == (1, 32, 20)
        torch.testing.assert_close(out_len, length)
        assert cache_last_channel_next.shape[2] == model.streaming_cfg.last_channel_cache_size
        assert cache_last_time_next.shape[0] == len(model.layers)
        torch.testing.assert_close(
            cache_last_channel_next_len,
            torch.full_like(cache_last_channel_next_len, model.streaming_cfg.last_channel_cache_size),
        )
