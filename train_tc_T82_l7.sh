cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/T82"
mkdir -p "$LOGDIR"

echo "===> layer 7"
torchrun --nproc-per-node=1 --master-port=29470\
  exp/train_tc_T82.py --lr 2e-3 --layer 7 --k 30 --exp_factor 16 \
  > "$LOGDIR/layer_7.log" 2>&1



# 1
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29510 exp/train_tc_T82.py \
    --lr 2e-4 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 

# 2
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29520 exp/train_tc_T82.py \
    --lr 1e-4 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 

# 3
CUDA_VISIBLE_DEVICES=4 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29530 exp/train_tc_T82.py \
    --lr 5e-4 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 

# 4
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29540 exp/train_tc_T82.py \
    --lr 1e-3 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 


# 5
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29550 exp/train_tc_T82.py \
    --lr 2e-3 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 

# 6
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29560 exp/train_tc_T82.py \
    --lr 4e-3 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 



# 8
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29500 exp/train_tc_T82.py \
    --lr 1e-3 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 \
    --use_auxk \
    --k_aux 2048 \
    --dead_threshold 100000

# # 8
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29500 exp/train_tc_T82.py \
#     --lr 1e-4 \
#     --layer 7 \
#     --k 30 \
#     --exp_factor 16 \

# # 9
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29500 exp/train_tc_T82.py \
#     --lr 5e-4 \
#     --layer 7 \
#     --k 30 \
#     --exp_factor 16 \

# # 10
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29500 exp/train_tc_T82.py \
#     --lr 1e-3 \
#     --layer 7 \
#     --k 30 \
#     --exp_factor 16 \


# # 11
# CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_CONSOLE=off torchrun --nproc-per-node=1 --master-port=29500 exp/train_tc_T82.py \
#     --lr 2e-3 \
#     --layer 7 \
#     --k 30 \
#     --exp_factor 16 \