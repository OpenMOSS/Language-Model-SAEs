cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate

# train tc


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train.py --lr 2e-3 --layer 0 --k 30 --exp_factor 16
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train_tc.py --lr 2e-3 --layer 1 --k 30 --exp_factor 16
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train_tc.py --lr 2e-3 --layer 2 --k 30 --exp_factor 16
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train_tc.py --lr 2e-3 --layer 14 --k 30 --exp_factor 16

WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29440 exp/train_tc.py --lr 2e-3 --layer 6 --k 30 --exp_factor 16



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_tc.py --lr 2e-3 --layer "$L" --k 30 --exp_factor 16 \
    > "$LOGDIR/layer_${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_tc.py --lr 2e-3 --layer "$L" --k 30 --exp_factor 16 \
    > "$LOGDIR/layer_${L}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_tc.py --lr 2e-3 --layer "$L" --k 30 --exp_factor 16 \
    > "$LOGDIR/layer_${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_tc.py --lr 2e-3 --layer "$L" --k 30 --exp_factor 16 \
    > "$LOGDIR/layer_${L}.log" 2>&1
done





cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  torchrun --nproc-per-node=1 --master-port=$((29440+L)) \
    exp/train_tc.py --lr 2e-3 --layer "$L" --k 30 --exp_factor 16 \
    > "$LOGDIR/layer_${L}.log" 2>&1
done
