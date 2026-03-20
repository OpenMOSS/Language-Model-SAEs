# Complete Replacement Models on Pythia-70M

This example reproduces the full Complete Replacement Model (CRM) pipeline in a resource-manageable setting:

1. Build a small local text dataset.
2. Generate cached activations for every attention and MLP replacement hook.
3. Train one transcoder per MLP layer.
4. Train one lorsa per attention layer.
5. Assemble a `ReplacementModel`.
6. Run attribution and report replacement/completeness scores.

It is a small-scale end-to-end reproduction of the CRM workflow from the repository README, not a paper-scale rerun of the released Qwen3-1.7B artifacts.

## Run

```bash
source .venv/bin/activate
python examples/reproduce_complete_replacement_models/run_pythia70m_crm.py
```

Useful flags:

```bash
python examples/reproduce_complete_replacement_models/run_pythia70m_crm.py \
  --workdir tmp/crm_pythia70m \
  --force \
  --activation-total-tokens 4096 \
  --transcoder-training-tokens 1024 \
  --lorsa-training-tokens 512 \
  --eval-prompt "The capital of France is"
```

For multi-GPU attribution / graph tracing, point the evaluation stage at multiple replica devices:

```bash
python examples/reproduce_complete_replacement_models/run_pythia70m_crm.py \
  --workdir tmp/crm_pythia70m \
  --parallel-devices cuda:0 cuda:1 cuda:2 cuda:3
```

This shards the expensive feature-attribution and QK-tracing phases across replicas. It does not pool VRAM across cards.

## Outputs

The script writes:

- `tmp/crm_pythia70m/dataset`
- `tmp/crm_pythia70m/activations`
- `tmp/crm_pythia70m/transcoders/layer_*`
- `tmp/crm_pythia70m/lorsas/layer_*`
- `tmp/crm_pythia70m/model_spec.json`
- `tmp/crm_pythia70m/eval_summary.json`

If a training stream ends before the requested number of steps, the script automatically resolves the newest checkpoint directory and uses it for CRM assembly.
