CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29448 \
  exp/analyze_lc0_tc_BT4.py --layer 8



# tc

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 0 3); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_tc_layer_${L}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 4 7); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_tc_layer_${L}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 8 11); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_tc_layer_${L}.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_tc_layer_${L}.log" 2>&1
done




# lorsa


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 0 3); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1030_2.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 4 7); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1030_2.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 8 11); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1030_2.log" 2>&1
done



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_BT4.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1030_2.log" 2>&1
done




# analyze T82




cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"

for L in $(seq 0 3); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"

for L in $(seq 4 7); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"

for L in $(seq 8 11); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> lorsa layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
    > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1
done


# analyze per layer
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=2
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=3
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=9
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=10
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=11
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=13
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=14
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=2
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=2 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1




cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=8
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=3 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1




cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=13
echo "===> lorsa layer $L"
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_lorsa_T82.py --layer "$L" \
  > "$LOGDIR/analyse_lorsa_layer_${L}_1031.log" 2>&1





cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=5
echo "===> tc layer $L"
CUDA_VISIBLE_DEVICES=5 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_tc.py --layer "$L" \
  > "$LOGDIR/analyse_tc_layer${L}.log" 2>&1



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=6
echo "===> tc layer $L"
CUDA_VISIBLE_DEVICES=6 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_tc.py --layer "$L" \
  > "$LOGDIR/analyse_tc_layer${L}.log" 2>&1



cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze/T82"
mkdir -p "$LOGDIR"
L=7
echo "===> tc layer $L"
CUDA_VISIBLE_DEVICES=7 torchrun --master_port=$((29440+L)) --nproc-per-node=1 exp/analyze_lc0_tc.py --layer "$L" \
  > "$LOGDIR/analyse_tc_layer${L}.log" 2>&1

