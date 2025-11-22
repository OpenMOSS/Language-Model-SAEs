
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 11 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 2048 \
      --dead_threshold 100000


# sweep lr
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for LR in 5e-5 1e-4; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr "$LR" \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_lr_${LR}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for LR in 2e-4 5e-4; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr "$LR" \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_lr_${LR}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for LR in 1e-3 2e-3; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr "$LR" \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_lr_${LR}.log" 2>&1
done

# sweep k_aux

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for KAUX in 512 1024; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux "$KAUX" \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_kaux_${KAUX}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for KAUX in 2048 4096; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux "$KAUX" \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_kaux_${KAUX}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"
echo "===> layer $L"

for KAUX in 8192 16384; do
  echo "===> lr $LR"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+RANDOM%1000)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer 10 \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux "$KAUX" \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l10_kaux_${KAUX}.log" 2>&1
done




# lr 2e-4

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 0 1); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 2 3); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 4 5); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 6 7); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 8 9); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 10 11); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in $(seq 12 13); do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/BT4_lorsa/k_128_exp_128"
mkdir -p "$LOGDIR"

for L in 14; do
  echo "===> layer $L"
  torchrun --nnodes=1 --nproc_per_node=8 --master-port=$((29440+L)) \
    exp/train_lorsa_BT4.py \
      --lr 2e-4 \
      --layer "$L" \
      --tp 8 \
      --dp 1 \
      --k 128 \
      --exp_factor 128 \
      --use_smolgen \
      --init_search \
      --initialize_lorsa_with_mhsa \
      --initialize_W_D_with_active_subspace \
      --initialize_lorsa_attn_scale_from_encoder \
      --k_aux 8192 \
      --dead_threshold 100000 \
    > "$LOGDIR/BT4_lorsa_l${L}_lr2e-4.log" 2>&1
done