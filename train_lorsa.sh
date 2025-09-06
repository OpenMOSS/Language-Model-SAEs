# script 1
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc-per-node=1 \
    --master-port=29320 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 

# script 2
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc-per-node=1 \
    --master-port=29370 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace


# script 3
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc-per-node=1 \
    --master-port=29330 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder


# script 4
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc-per-node=1 \
    --master-port=29340 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen


# script 5
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29350 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen

# script 6
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29300 \
    exp/train_lorsa.py \
    --lr 1e-4 \
    --layer 1 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen


# 按顺序训练 # 这个配置看起来还行
for layer in {0..2}; do
    echo "开始训练 layer $layer"
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=29300 \
        exp/train_lorsa.py \
        --lr 1e-4 \
        --layer 1 \
        --k 20 \
        --exp_factor 16 \
        --initialize_lorsa_attn_scale_from_encoder \
        --use_smolgen
done


for layer in {3..5}; do
    echo "开始训练 layer $layer"
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=29310 \
        exp/train_lorsa.py \
        --lr 2e-4 \
        --layer 1 \
        --k 20 \
        --exp_factor 16 \
        --initialize_lorsa_attn_scale_from_encoder \
        --use_smolgen
done

for layer in {6..8}; do
    echo "开始训练 layer $layer"
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=29320 \
        exp/train_lorsa.py \
        --lr 2e-4 \
        --layer 1 \
        --k 20 \
        --exp_factor 16 \
        --initialize_lorsa_attn_scale_from_encoder \
        --use_smolgen
done

for layer in {9..11}; do
    echo "开始训练 layer $layer"
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=29330 \
        exp/train_lorsa.py \
        --lr 2e-4 \
        --layer 1 \
        --k 20 \
        --exp_factor 16 \
        --initialize_lorsa_attn_scale_from_encoder \
        --use_smolgen
done    

for layer in {12..14}; do
    echo "开始训练 layer $layer"
    WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc-per-node=1 \
        --master-port=29330 \
        exp/train_lorsa.py \
        --lr 2e-4 \
        --layer 1 \
        --k 20 \
        --exp_factor 16 \
        --initialize_lorsa_attn_scale_from_encoder \
        --use_smolgen
done