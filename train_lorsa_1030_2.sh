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
      --k 50 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_2_layer_${L}.log" 2>&1
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
      --k 50 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_2_layer_${L}.log" 2>&1
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
      --k 50 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_2_layer_${L}.log" 2>&1
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
      --k 50 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_2_layer_${L}.log" 2>&1
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
      --k 50 \
      --exp_factor 16 \
      --use_smolgen \
      --k_aux 4096 \
      --dead_threshold 1000000 \
      --aux_coefficient 0.125 \
    > "$LOGDIR/lorsa_1030_2_layer_${L}.log" 2>&1
done

