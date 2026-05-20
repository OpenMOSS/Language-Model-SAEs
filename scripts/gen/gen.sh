#!/usr/bin/env bash
set -euo pipefail
# gen 1d training activation
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29620 --nproc_per_node=2 exp/gen_evo2_tc.py \
    --layers 26 \
    --total-tokens 10_100_000

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29630 --nproc_per_node=2 exp/gen_evo2_tc_2d.py \
    --layer 26 \
    --total-tokens 1_100_000


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_evo2_tc.py \
    --layers 26 \
    --total-tokens 10_100_000

# gen 2d analyzing activation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29630 --nproc_per_node=8 exp/gen_evo2_tc_2d.py \
    --layer 26 \
    --total-tokens 1_100_000