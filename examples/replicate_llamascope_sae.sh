#!/usr/bin/bash



exp_factor=$1
tc_in_abbr=$2
layer=$3

tc_out_abbr=$tc_in_abbr

if [ "$exp_factor" -eq 128 ]; then
    tp_size=2
else
    tp_size=1
fi

k=50


if [ "$exp_factor" -eq 8 ]; then
    total_training_tokens=800000000
    lr=8e-4
elif [ "$exp_factor" -eq 32 ]; then
    total_training_tokens=1600000000
    lr=8e-4
elif [ "$exp_factor" -eq 128 ]; then
    total_training_tokens=3200000000
    lr=2e-4
fi

if [ "$tc_in_abbr" = "TC" ]; then
    tc_out_abbr="M"
fi

WANDB_CONSOLE=off WANDB_MODE=offline torchrun --nproc_per_node=$tp_size --master_port=10110 ./examples/programmatic/train_llama_scope.py --total_training_tokens $total_training_tokens --layer $layer --lr $lr --clip_grad_norm 0.001 --exp_factor $exp_factor --batch_size 2048 --tp_size $tp_size --buffer_size 500000 --log_to_wandb false --store_batch_size 32 --k $k --tc_in_abbr $tc_in_abbr --tc_out_abbr $tc_out_abbr --fix_norm 0.5
