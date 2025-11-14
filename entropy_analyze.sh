cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate   
LOGDIR="$(pwd)/logs_analyze"
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=21500 --nproc-per-node=1 exp/analyze_lc0_tc_entropy.py --layer 0

