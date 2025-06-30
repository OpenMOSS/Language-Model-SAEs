. /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs/.venv/bin/activate

cd /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jiaxing_projects/Language-Model-SAEs



echo "Running boi-batch2d-b.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-b.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-b.py executed successfully."
else
    echo "boi-batch2d-b.py failed to execute."
fi


echo "Running boi-batch2d-b-8.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-b-8.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-b-8.py executed successfully."
else
    echo "boi-batch2d-b-8.py failed to execute."
fi

echo "Running boi-batch2d-i.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-i.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-i.py executed successfully."
else
    echo "boi-batch2d-i.py failed to execute."
fi


echo "Running boi-batch2d-i-8.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-i-8.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-i-8.py executed successfully."
else
    echo "boi-batch2d-i-8.py failed to execute."
fi

echo "Running boi-batch2d-o.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-o.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-o.py executed successfully."
else
    echo "boi-batch2d-o.py failed to execute."
fi


echo "Running boi-batch2d-o-8.py..."
HF_HUB_OFFLINE=1 torchrun --nproc-per-node=8  exp/programmatic/boi-batch2d-o-8.py

if [ $? -eq 0 ]; then
    echo "boi-batch2d-o-8.py executed successfully."
else
    echo "boi-batch2d-o-8.py failed to execute."
fi