LOGDIR="$(pwd)/logs/analyze_tc"
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29141 \
  exp/analyze_lc0_tc_BT4.py --layer 12 \
  > "$LOGDIR/tc_BT4_bigger12.log" 2>&1

LOGDIR="$(pwd)/logs/analyze_tc"
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29142 \
  exp/analyze_lc0_tc_BT4.py --layer 13 \
  > "$LOGDIR/tc_BT4_bigger13.log" 2>&1

LOGDIR="$(pwd)/logs/analyze_tc"
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29148 \
  exp/analyze_lc0_tc_BT4.py --layer 14 \
  > "$LOGDIR/tc_BT4_bigger14.log" 2>&1


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_tc"
mkdir -p "$LOGDIR"

for L in $(seq 0 2); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/tc_BT4_bigger${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_tc"
mkdir -p "$LOGDIR"

for L in $(seq 3 5); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/tc_BT4_bigger${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_tc"
mkdir -p "$LOGDIR"

for L in $(seq 6 8); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/tc_BT4_bigger${L}.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_tc"
mkdir -p "$LOGDIR"

for L in $(seq 9 11); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/tc_BT4_bigger${L}.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_tc"
mkdir -p "$LOGDIR"

for L in $(seq 12 14); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_tc_BT4.py --layer "$L" \
    > "$LOGDIR/tc_BT4_bigger${L}.log" 2>&1
done
