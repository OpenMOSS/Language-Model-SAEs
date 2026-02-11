
# sweep lr

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_30_exp_32"
mkdir -p "$LOGDIR"

for LR in 4e-5 8e-5; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=4 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 13 \
        --tp 4 \
        --dp 1 \
        --k 30 \
        --exp_factor 32 \
    > "$LOGDIR/BT4_tc_l13_lr_${LR}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_30_exp_32"
mkdir -p "$LOGDIR"

for LR in 1e-4 2e-4; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=4 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 13 \
        --tp 4 \
        --dp 1 \
        --k 30 \
        --exp_factor 32 \
    > "$LOGDIR/BT4_tc_l13_lr_${LR}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_tc/k_30_exp_32"
mkdir -p "$LOGDIR"

for LR in 4e-4 8e-4; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=4 --master-port=$((29440+RANDOM%1000)) \
    exp/train_tc_BT4.py \
        --lr $LR \
        --layer 13 \
        --tp 4 \
        --dp 1 \
        --k 30 \
        --exp_factor 32 \
    > "$LOGDIR/BT4_tc_l13_lr_${LR}.log" 2>&1
done
