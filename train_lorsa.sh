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
    --lr 2e-4 \
    --layer 7 \
    --k 20 \
    --exp_factor 16 \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen


# script 7
# 6000:
  below_1e-5 : 11,964
  below_1e-6 : 11,310
  6 : 207.0876
  explained_variance  : 0.6692
  有点跳ev
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29300 \
    exp/train_lorsa.py \
    --lr 2e-3 \
    --layer 0 \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder 

# 7-2
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29300 \
    exp/train_lorsa.py \
    --lr 2e-3 \
    --layer 0 \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_attn_scale_from_encoder 





# script 8 
# 8-3
# it's good
  above_1e-1 : 73
  below_1e-6 : 12,111
  Training Metrics - Step 16000                         
  mse_loss     : 64.7556
  overall_loss : 64.7556
  explained_variance  : 0.8936
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun \
    --nproc-per-node=1 \
    --master-port=29500 \
    exp/train_lorsa.py \
    --lr 5e-4 \
    --layer 6 \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen


  above_1e-1 : 52
  below_1e-6 : 11,031
  Training Metrics - Step 16000                          
  mse_loss     : 744.4966
  overall_loss : 744.4966
  explained_variance  : 0.7393
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc-per-node=1 \
    --master-port=29200 \
    exp/train_lorsa.py \
    --lr 5e-4 \
    --layer 7 \
    --k 30 \
    --exp_factor 16 \
    --initialize_lorsa_with_mhsa \
    --initialize_W_D_with_active_subspace \
    --initialize_lorsa_attn_scale_from_encoder \
    --use_smolgen




# 按顺序训练 # 这个配置看起来还行
# 8-3
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
    > "$LOGDIR/lorsa_layer_${L}.log" 2>&1
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
    > "$LOGDIR/lorsa_layer_${L}.log" 2>&1
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
    > "$LOGDIR/lorsa_layer_${L}.log" 2>&1
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
    > "$LOGDIR/lorsa_layer_${L}.log" 2>&1
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
    > "$LOGDIR/lorsa_layer_${L}.log" 2>&1
done




# analyze
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LAYER in $(seq 0 2); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer "${LAYER}" \
      --n_tokens 100000000 \
    >"$LOGDIR/analyze_a_3_layer_${LAYER}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LAYER in $(seq 3 5); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer "${LAYER}" \
      --n_tokens 100000000 \
    >"$LOGDIR/analyze_a_3_layer_${LAYER}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LAYER in $(seq 6 8); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer "${LAYER}" \
      --n_tokens 100000000 \
    >"$LOGDIR/analyze_a_3_layer_${LAYER}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LAYER in $(seq 9 11); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer "${LAYER}" \
      --n_tokens 100000000 \
    >"$LOGDIR/analyze_a_3_layer_${LAYER}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for LAYER in $(seq 12 14); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer "${LAYER}" \
      --n_tokens 100000000 \
    >"$LOGDIR/analyze_a_3_layer_${LAYER}.log" 2>&1
done



# analyze lorsa

for LAYER in $(seq 0 2); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer ${LAYER} \
      --n_tokens 100000000
done

for LAYER in $(seq 3 5); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer ${LAYER} \
      --n_tokens 100000000
done

for LAYER in $(seq 6 8); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=1 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer ${LAYER} \
      --n_tokens 100000000
done

for LAYER in $(seq 9 11); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=0 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer ${LAYER} \
      --n_tokens 100000000
done

for LAYER in $(seq 12 14); do
  echo "==== Running layer ${LAYER} ===="
  CUDA_VISIBLE_DEVICES=1 \
  torchrun \
    --standalone \
    --max_restarts=3 \
    --master_port=$((30000 + LAYER)) \
    --nproc-per-node=1 \
    exp/analyze_lc0_lorsa.py \
      --layer ${LAYER} \
      --n_tokens 100000000
done







# 另外一种配置
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for layer in $(seq 0 2); do
  echo "==== Running layer ${layer} ===="
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc-per-node=1 \
      --master-port=$((29300 + layer)) \
      exp/train_lorsa.py \
      --lr 2e-3 \
      --layer ${layer} \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_attn_scale_from_encoder \
      > "$LOGDIR/lorsa_layer_${layer}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for layer in $(seq 3 5); do
  echo "==== Running layer ${layer} ===="
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc-per-node=1 \
      --master-port=$((29300 + layer)) \
      exp/train_lorsa.py \
      --lr 2e-3 \
      --layer ${layer} \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_attn_scale_from_encoder \
      > "$LOGDIR/lorsa_layer_${layer}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for layer in $(seq 6 8); do
  echo "==== Running layer ${layer} ===="
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc-per-node=1 \
      --master-port=$((29300 + layer)) \
      exp/train_lorsa.py \
      --lr 2e-3 \
      --layer ${layer} \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_attn_scale_from_encoder \
      > "$LOGDIR/lorsa_layer_${layer}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for layer in $(seq 9 11); do
  echo "==== Running layer ${layer} ===="
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc-per-node=1 \
      --master-port=$((29300 + layer)) \
      exp/train_lorsa.py \
      --lr 2e-3 \
      --layer ${layer} \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_attn_scale_from_encoder \
      > "$LOGDIR/lorsa_layer_${layer}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate
export WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

for layer in $(seq 12 14); do
  echo "==== Running layer ${layer} ===="
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun \
      --nproc-per-node=1 \
      --master-port=$((29300 + layer)) \
      exp/train_lorsa.py \
      --lr 2e-3 \
      --layer ${layer} \
      --k 30 \
      --exp_factor 16 \
      --initialize_lorsa_attn_scale_from_encoder \
      > "$LOGDIR/lorsa_layer_${layer}.log" 2>&1
done
