cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

echo "===> BT4 layer 8"
torchrun --nproc-per-node=1 --master-port=29600 \
  exp/train_tc_BT4.py --lr 2e-3 --layer 8 --k 30 --exp_factor 16 \
  > "$LOGDIR/layer_8_1.log" 2>&1




cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

echo "===> BT4 layer 8"
torchrun --nproc-per-node=1 --master-port=29601 \
  exp/train_tc_BT4.py --lr 1e-3 --layer 8 --k 30 --exp_factor 16 \
  > "$LOGDIR/layer_8_2.log" 2>&1



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

echo "===> BT4 layer 8"
torchrun --nproc-per-node=1 --master-port=29602 \
  exp/train_tc_BT4.py --lr 5e-4 --layer 8 --k 30 --exp_factor 16 \
  > "$LOGDIR/layer_8_3.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4"
mkdir -p "$LOGDIR"

echo "===> BT4 layer 8"
torchrun --nproc-per-node=1 --master-port=29603 \
  exp/train_tc_BT4.py --lr 2e-4 --layer 8 --k 30 --exp_factor 16 \
  > "$LOGDIR/layer_8_4.log" 2>&1