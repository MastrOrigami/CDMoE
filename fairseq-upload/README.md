# fairseq-upload (Anonymous Submission Package)

This repository is the upload-ready code package for anonymous paper submission.
All commands below are intended to run inside this repository (`fairseq-upload`).

## Environment

- Python 3.8+
- PyTorch with CUDA
- 2 GPUs (example uses `CUDA_VISIBLE_DEVICES=0,1`)

Recommended environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Training (Translation MoE)

```bash
cd /path/to/fairseq-upload
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

n_experts=6

fairseq-train data-bin/wmt19_eng_deu_tok \
  --ddp-backend legacy_ddp \
  --task translation_moe \
  --user-dir examples/translation_moe/translation_moe_src \
  --arch transformer_nllb_1b_moe \
  --num-experts $n_experts --method hMoElp --mean-pool-gating-network \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
  --clip-norm 1.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.2 --attention-dropout 0.1 --activation-dropout 0.1 \
  --weight-decay 1e-4 \
  --max-tokens 2048 --update-freq 3 \
  --max-update 10000000 \
  --save-dir checkpoints/wmt19_eng_deu_tok_${n_experts}_new/nllb_moe_1b_preln \
  --validate-interval-updates 1000 --save-interval-updates 1000 \
  --keep-interval-updates 5 \
  --share-all-embeddings \
  --distributed-world-size 2 \
  --fp16 --memory-efficient-fp16 \
  --checkpoint-activations \
  --skip-invalid-size-inputs-valid-test \
  --tensorboard-logdir train-logs/wmt19_nllb_moe_${n_experts}/nllb_moe_1b_preln \
  --log-format simple \
  --log-interval 1
```

## Generation / Inference

```bash
cd /path/to/fairseq-upload
export CUDA_VISIBLE_DEVICES=0,1

n_experts=6
batch=nllb_moe_1b_preln

fairseq-generate data-bin/wmt19_eng_deu_tok \
  --path checkpoints/wmt19_eng_deu_tok_6_new/nllb_moe_1b_preln/checkpoint_best.pt \
  --beam 5 \
  --remove-bpe \
  --task translation_moe \
  --user-dir examples/translation_moe/translation_moe_src \
  --arch transformer_nllb_1b_moe \
  --method hMoElp \
  --mean-pool-gating-network \
  --num-experts $n_experts \
  --distributed-world-size 2
```

## Notes

- The commands above were adapted from `命令.md`.
- Update dataset paths and checkpoint paths to match your local setup if needed.
- This package is prepared for reproducibility and anonymous review.
