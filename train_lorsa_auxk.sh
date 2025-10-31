

# sweep learning rate a-1
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LR in 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3; do
  echo "===> lr $LR"
  torchrun --nproc-per-node=1 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa.py \
      --lr "$LR" \
      --layer 6 \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_l6_lr_${LR}.log" 2>&1
done
1e-4最佳







# a-1
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29200 \
    exp/train_lorsa.py \
    --lr 2e-3 \
    --layer 6 \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen \
    --k_aux 2048 \
    --dead_threshold 100000






# 分布式训练脚本 a-1
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 5e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_layer_${L}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 5e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 5e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 5e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 5e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_auxk_layer_${L}.log" 2>&1
done




# rlin_train_lorsa_l6_sweep_learning_rate_a-2_k_aux_4096
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LR in 1e-5 2e-5 5e-5 8e-5 1e-4 2e-4 5e-4 1e-3; do
  echo "===> lr $LR"
  torchrun --nproc-per-node=1 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa.py \
      --lr "$LR" \
      --layer 6 \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_l6_lr_${LR}.log" 2>&1
done



# train a-2
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 100000 \
      --aux_coefficient 0.05 \
    > "$LOGDIR/lorsa_a_2_layer_${L}.log" 2>&1
done



# train a-3
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

L=14
echo "===> layer $L"
torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
  exp/train_lorsa.py \
    --lr 8e-5 \
    --layer "$L" \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen \
    --k_aux 4096 \
    --dead_threshold 1000000 \
    --aux_coefficient 0.0625 \
  > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1



# BT4
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
torchrun --nproc-per-node=1 --master-port=29460 \
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




cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 8e-5 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}.log" 2>&1
done



# sweep learning rate

# 在以下配置下2e-4是最好的


# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate

# export WANDB_MODE=offline WANDB_CONSOLE=off
# export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# L=10 

# for LR in 8e-5 4e-5 2e-5 1e-5; do
#   echo "===> layer $L, lr=$LR"

#   torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
#     exp/train_lorsa_BT4.py \
#     --lr "$LR" \
#     --layer "$L" \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 4096 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.0625 \
#     > "$LOGDIR/lorsa_a_3_layer_${L}_lr_${LR}.log" 2>&1
# done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate

export WANDB_MODE=offline WANDB_CONSOLE=off
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

L=10 

for LR in 8e-4 4e-4 2e-4 1e-4; do
  echo "===> layer $L, lr=$LR"

  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
    --lr "$LR" \
    --layer "$L" \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen \
    --k_aux 4096 \
    --dead_threshold 1000000 \
    --aux_coefficient 0.0625 \
    > "$LOGDIR/lorsa_a_3_layer_${L}_lr_${LR}.log" 2>&1
done


# sweep aux_coefficient
#1 
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate

export WANDB_MODE=offline WANDB_CONSOLE=off
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

L=10
LR=2e-4

for AUX in 0.25 0.125 0.0625; do
  echo "===> layer $L, lr=$LR, aux=$AUX"

  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
    --lr "$LR" \
    --layer "$L" \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen \
    --k_aux 4096 \
    --dead_threshold 1000000 \
    --aux_coefficient "$AUX" \
    > "$LOGDIR/lorsa_a_3_layer_${L}_lr_${LR}_aux_${AUX}.log" 2>&1
done

#2
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate

export WANDB_MODE=offline WANDB_CONSOLE=off
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

L=10
LR=2e-4

for AUX in 0.03 0.01 0.005; do
  echo "===> layer $L, lr=$LR, aux=$AUX"

  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
    --lr "$LR" \
    --layer "$L" \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen \
    --k_aux 4096 \
    --dead_threshold 1000000 \
    --aux_coefficient "$AUX" \
    > "$LOGDIR/lorsa_a_3_layer_${L}_lr_${LR}_aux_${AUX}.log" 2>&1
done




# 发现0.125更好


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_layer_${L}.log" 2>&1
done



# 尝试一些其他的配置

# train lorsa 0 原始标准配置
# WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
#     --nproc-per-node=1 \
#     --master-port=29200 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 4096 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.125


# train lorsa 1
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"


# WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
#     --nproc-per-node=1 \
#     --master-port=29300 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 4096 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.25 \
#     > "$LOGDIR/lorsa_1029_l6_1.log" 2>&1



# # train lorsa 2
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"


# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29400 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --use_smolgen \
#     --k_aux 4096 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.25 \
#     > "$LOGDIR/lorsa_1029_l6_2.log" 2>&1


# train lorsa 3
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"


torchrun \
    --nproc-per-node=1 \
    --master-port=29500 \
    exp/train_lorsa_BT4.py \
    --lr 2e-4 \
    --layer 6 \
    --k 30 \
    --exp_factor 16 \
    --use_smolgen \
    --k_aux 4096 \
    --dead_threshold 1000000 \
    --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1029_l6_3.log" 2>&1



# # train lorsa 4
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29600 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 1024 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.125 \
#     > "$LOGDIR/lorsa_1029_l6_4.log" 2>&1



# # train lorsa 5
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29600 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 1024 \
#     --dead_threshold 100000 \
#     --aux_coefficient 0.125 \
#     > "$LOGDIR/lorsa_1029_l6_5.log" 2>&1


# # train lorsa 6
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29700 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --initialize_lorsa_with_mhsa \
#     --initialize_W_D_with_active_subspace \
#     --initialize_lorsa_attn_scale_from_encoder \
#     --use_smolgen \
#     --k_aux 4096 \
#     --dead_threshold 100000 \
#     --aux_coefficient 0.125 \
#     > "$LOGDIR/lorsa_1029_l6_6.log" 2>&1


# # train lorsa 7
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29700 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --k_aux 4096 \
#     --dead_threshold 1000000 \
#     --aux_coefficient 0.125 \
#     > "$LOGDIR/lorsa_1029_l6_7.log" 2>&1

# # train lorsa 8
# cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
# . .venv/bin/activate
# export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

# LOGDIR="$(pwd)/logs/BT4"
# mkdir -p "$LOGDIR"

# torchrun \
#     --nproc-per-node=1 \
#     --master-port=29700 \
#     exp/train_lorsa_BT4.py \
#     --lr 2e-4 \
#     --layer 6 \
#     --k 30 \
#     --exp_factor 16 \
#     --k_aux 1024 \
#     --dead_threshold 100000 \
#     --aux_coefficient 0.125 \
#     > "$LOGDIR/lorsa_1029_l6_8.log" 2>&1

