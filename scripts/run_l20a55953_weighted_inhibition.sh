#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
set -a
source server/.env
set +a

echo "Scanning qwen3-1.7b-lorsa-32x-topk128-layer20 #55953"
.venv/bin/llamascopium inhibitory-global-weights \
  --series qwen3-1.7b \
  --sae-set qwen3-1.7b-full-32x-k128-final \
  --target-sae qwen3-1.7b-lorsa-32x-topk128-layer20 \
  --feature-id 55953 \
  --dataset qwen3 \
  --max-samples 100 \
  --depth 3 \
  --expansion-width 3 \
  --device cuda \
  --output results/L20A-55953-inhibitory.json
