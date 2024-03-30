import torch
import os
import sys

sys.path.insert(0, os.getcwd())

from core.config import SAEConfig, FeaturesDecoderConfig
from core.runner import features_to_logits_runner

file_names = ['L0A-l1-0.00012-lr-0.001-32x', 'L0M-l1-0.00012-lr-0.001-32x',
              'L1A-l1-0.00012-lr-0.001-32x', 'L1M-l1-0.00012-lr-0.001-32x',
              'L2A-l1-0.00012-lr-0.001-32x', 'L2M-l1-0.00012-lr-0.001-32x',
              'L3A-l1-0.00012-lr-0.001-32x', 'L3M-l1-0.00012-lr-0.001-32x',
              'L4A-l1-0.00012-lr-0.001-32x', 'L4M-l1-0.00012-lr-0.001-32x',
              'L5A-l1-0.00012-lr-0.001-32x', 'L5M-l1-0.00012-lr-0.001-32x',
              'L6A-l1-0.00012-lr-0.001-32x', 'L6M-l1-0.00012-lr-0.001-32x',
              'L7A-l1-0.00012-lr-0.001-32x', 'L7M-l1-0.00012-lr-0.001-32x',
              'L8A-l1-0.00012-lr-0.001-32x', 'L8M-l1-0.00012-lr-0.001-32x',
              'L9A-l1-0.00012-lr-0.001-32x', 'L9M-l1-0.00012-lr-0.001-32x',
              'L10A-l1-0.00012-lr-0.001-32x', 'L10M-l1-0.00012-lr-0.001-32x',
              'L11A-l1-0.00012-lr-0.001-32x', 'L11M-l1-0.00012-lr-0.001-32x',]

for file_name in file_names:
    cfg = FeaturesDecoderConfig(
        **SAEConfig.get_hyperparameters(file_name, '/remote-home/share/research/mechinterp/gpt2-dictionary/results', 'pruned.pt'),
        file_path= './test',# '/remote-home/share/research/mechinterp/gpt2-dictionary/results/' + file_name + '/feature/feature_to_logits.arrow'
        device='cuda:6'
        )
    features_to_logits_runner(cfg)
