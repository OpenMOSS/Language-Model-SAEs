# tc

WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train_tc_BT4.py --lr 2e-3 --layer 14 --k 30 --exp_factor 16

# lorsa

WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29450 \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer 14 \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625