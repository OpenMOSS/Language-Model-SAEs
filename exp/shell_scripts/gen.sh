. /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Language-Model-SAEs/.venv/bin/activate

cd /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/zf_projects/Language-Model-SAEs/

HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  examples/diff3/boi-batch2d-b.py --start-shard 0; HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  examples/diff3/boi-batch2d-b.py --start-shard 8;