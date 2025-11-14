

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=12600 exp/train_lorsa_BT4.py \
    --lr 2e-4 \
    --layer 10 \
    --k 30 \
    --exp_factor 16 \
    --use_smolgen \
    --initialize_lorsa_with_mhsa

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=12500 exp/train_lorsa_BT4.py \
    --lr 2e-4 \
    --layer 10 \
    --k 30 \
    --exp_factor 16 \
    --use_smolgen \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=12501 exp/train_tc_T82.py \
--lr 2e-3 --layer 7 --k 30 --exp_factor 16



# train lorsa k_30
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4/final_lorsa"
mkdir -p "$LOGDIR"
for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_update_k_30_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4/final_lorsa"
mkdir -p "$LOGDIR"
for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_update_k_30_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4/final_lorsa"
mkdir -p "$LOGDIR"
for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_update_k_30_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4/final_lorsa"
mkdir -p "$LOGDIR"
for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_update_k_30_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4/final_lorsa"
mkdir -p "$LOGDIR"
for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --k 30 \
      --exp_factor 16 \
      --use_smolgen \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000 \
    > "$LOGDIR/lorsa_update_k_30_${L}.log" 2>&1
done

