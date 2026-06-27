# #!/bin/bash
# # RCJiT-S/16 on MIMIC-CXR, conditioned on MAE-pretrained ViT-S/16 CLS token
# # Usage: bash cxr-ssl/train_rcjit_mae.sh [extra args...]

# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# python "${SCRIPT_DIR}/train_rcjit_mae.py" \
#   --encoder_ckpt  "${SCRIPT_DIR}/cxr-ssl/ssnlqx8u/checkpoints/epoch=799-step=576000.ckpt" \
#   --data_dir      /mnt/nvme1/mimic-cxr-jpg \
#   --output_dir    "${SCRIPT_DIR}/../output/rcjit_s16_mae_cxr" \
#   --model_variant S_16 \
#   --img_size      224 \
#   --batch_size    128 \
#   --num_workers   12 \
#   --num_gpus      -1 \
#   --lr            5e-5 \
#   --weight_decay  0.0 \
#   --max_steps     300000 \
#   --warmup_steps  10000 \
#   --cond_drop_prob 0.1 \
#   --cfg           1.5 \
#   --ode_steps     50 \
#   --ode_method    heun \
#   --interval_min  0.0 \
#   --interval_max  1.0 \
#   --save_every    10000 \
#   --sample_every  10000 \
#   --num_samples   8 \
#   --ema_decay1    0.9999 \
#   --ema_decay2    0.9996 \
#   --P_mean        -0.8 \
#   --P_std         0.8 \
#   --noise_scale   1.0 \
#   --t_eps         0.05 \
#   --log_every     50 \
#   --wandb_project rcjit-s16-mae-cxr \
#   --compile \
#   "$@"


python cxr-ssl/train_rcjit_mae.py \
      --encoder_ckpt cxr-ssl/cxr-ssl/ssnlqx8u/checkpoints/epoch=799-step=576000.ckpt \
      --data_dir /mnt/nvme1/mimic-cxr-jpg \
      --output_dir output/rcjit_s16_mae_pool_cxr \
      --wandb_run_name rcjit-s16-mae-pool-300k