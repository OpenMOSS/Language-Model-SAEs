
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 16 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000


# sweep lr
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for LR in 1e-4 2e-4; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 14 \
        --k 128 \
        --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l14_lr_${LR}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for LR in 5e-4 1e-3; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 14 \
        --k 128 \
        --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l14_lr_${LR}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for LR in 2e-3 5e-3; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 14 \
        --k 128 \
        --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l14_lr_${LR}.log" 2>&1
done

# lr 1e-4

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_tc_BT4.py \
      --lr 1e-4 \
      --layer "$L" \
      --k 128 \
      --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_tc_BT4.py \
      --lr 1e-4 \
      --layer "$L" \
      --k 128 \
      --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_tc_BT4.py \
      --lr 1e-4 \
      --layer "$L" \
      --k 128 \
      --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_tc_BT4.py \
      --lr 1e-4 \
      --layer "$L" \
      --k 128 \
      --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_tc_BT4.py \
      --lr 1e-4 \
      --layer "$L" \
      --k 128 \
      --exp_factor 128 \
    > "$LOGDIR/BT4_tc_l${L}.log" 2>&1
done
