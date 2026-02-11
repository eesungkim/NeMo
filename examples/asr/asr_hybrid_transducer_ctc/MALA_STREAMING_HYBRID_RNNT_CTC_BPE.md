# MALA Streaming Hybrid RNNT-CTC (BPE)

This guide shows how to train a cache-aware streaming FastConformer Hybrid RNNT-CTC model using `self_attention_model=mala`.

## Files

- Config:
  `examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming_mala.yaml`
- Training entrypoint:
  `examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_streaming_mala.py`

## Main MALA Encoder Settings

In the YAML:

- `model.encoder.self_attention_model: mala`
- `model.encoder.att_context_size: [-1, 0]`
- `model.encoder.att_context_style: regular`
- `model.encoder.conv_context_size: causal`

These settings enable causal streaming with unlimited left context and zero right look-ahead in attention.

## Train

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_streaming_mala.py \
  model.train_ds.manifest_filepath=/path/to/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
  model.tokenizer.dir=/path/to/tokenizer_dir \
  model.tokenizer.type=bpe \
  trainer.devices=1 \
  trainer.max_epochs=100
```

## Optional Overrides

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_streaming_mala.py \
  model.train_ds.manifest_filepath=/path/to/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
  model.tokenizer.dir=/path/to/tokenizer_dir \
  model.tokenizer.type=bpe \
  model.train_ds.batch_size=8 \
  model.optim.lr=0.001 \
  trainer.devices=4 \
  trainer.precision=bf16 \
  exp_manager.create_wandb_logger=true \
  exp_manager.wandb_logger_kwargs.project=my_project \
  exp_manager.wandb_logger_kwargs.name=fastconformer_mala_streaming
```

## Notes

- Set `model.test_ds.manifest_filepath` in the config or via CLI if you want test evaluation after training.
- Keep `model.preprocessor.normalize="NA"` for streaming-friendly behavior.
