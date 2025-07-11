import os
import einops
import chess
import chess.svg
from jax import random as jrandom
import numpy as np

import torch
from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import jax
print(jax.devices())        # 应看到 CudaDevice

# newly added
import sys, pathlib

# Notebook 当前的工作目录默认就是文件所在目录
PROJECT_ROOT = "/inspire/hdd/global_user/hezhengfu-240208120186/models/chess" 
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import neural_engines

# @title Create the predictor.

policy = 'action_value'
num_return_buckets = 128

match policy:
  case 'action_value':
    output_size = num_return_buckets
  case 'behavioral_cloning':
    output_size = utils.NUM_ACTIONS
  case 'state_value':
    output_size = num_return_buckets
  case _:
    raise ValueError(f'Unknown policy: {policy}')

predictor_config = transformer.TransformerConfig(
    vocab_size=utils.NUM_ACTIONS,
    output_size=output_size,
    pos_encodings=transformer.PositionalEncodings.LEARNED,
    max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
    num_heads=8,
    num_layers=16,
    embedding_dim=1024,
    apply_post_ln=True,
    apply_qk_layernorm=False,
    use_causal_mask=False,
)

predictor = transformer.build_transformer_predictor(config=predictor_config)

# @title Load the predictor parameters

checkpoint_dir = os.path.join(
    os.getcwd(),
    # f'../checkpoints/270M/{policy}/',
    f'/inspire/hdd/global_user/hezhengfu-240208120186/models/chess/searchless_chess/checkpoints/270M',
)
dummy_params = predictor.initial_params(
    rng=jrandom.PRNGKey(0),
    targets=np.zeros((1, 1), dtype=np.uint32),
)
params = training_utils.load_parameters(
    checkpoint_dir=checkpoint_dir,
    params=dummy_params,
    use_ema_params=True,
    step=-1,
)


def convert_searchless_chess_weights(param, cfg: HookedTransformerConfig):
    """
    param: dict, searchless_chess权重
    n_layers: 层数
    d_model: embedding维度
    d_mlp: mlp隐藏层维度
    n_heads: 注意力头数
    """
    state_dict = {}

    # 1. Embedding
    if "embed" in param:
        state_dict["embed.token_embed.weight"] = torch.tensor(np.array(param["embed"]["embeddings"]))
    if "embed_1" in param:   
        state_dict["embed.pos_embed.weight"] = torch.tensor(np.array(param["embed_1"]["embeddings"]))

    # 2. Transformer blocks
    for l in range(cfg.n_layers):
        # LayerNorm1
        if l == 0:
            state_dict[f"blocks.{l}.ln1.w"] = torch.tensor(np.array(param["layer_norm"]["scale"]))
            state_dict[f"blocks.{l}.ln1.b"] = torch.tensor(np.array(param["layer_norm"]["offset"]))
        else:
            state_dict[f"blocks.{l}.ln1.w"] = torch.tensor(np.array(param[f"layer_norm_{2*l}"]["scale"]))
            state_dict[f"blocks.{l}.ln1.b"] = torch.tensor(np.array(param[f"layer_norm_{2*l}"]["offset"]))

        # Attention QKV/O
        if l == 0:
            prefix = f"multi_head_dot_product_attention"
        else:
            prefix = f"multi_head_dot_product_attention_{l}"
            
        W_Q = torch.tensor(np.array(param[f"{prefix}/linear"]["w"]))
        W_K = torch.tensor(np.array(param[f"{prefix}/linear_1"]["w"]))
        W_V = torch.tensor(np.array(param[f"{prefix}/linear_2"]["w"]))
        W_O = torch.tensor(np.array(param[f"{prefix}/linear_3"]["w"]))
        
        W_Q = einops.rearrange(W_Q, "m (n h)->n m h", n=cfg.n_heads) 
        W_K = einops.rearrange(W_K, "m (n h)->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (n h)->n m h", n=cfg.n_heads)
        W_O = einops.rearrange(W_O, "(n h) m->n h m", n=cfg.n_heads)
        
        b_Q = torch.zeros(cfg.d_model)
        b_K = torch.zeros(cfg.d_model)
        b_V = torch.zeros(cfg.d_model)
        b_O = torch.zeros(cfg.d_model)
        
        b_Q = einops.rearrange(b_Q, "(n h)->n h", n=cfg.n_heads)
        b_K = einops.rearrange(b_K, "(n h)->n h", n=cfg.n_heads)
        b_V = einops.rearrange(b_V, "(n h)->n h", n=cfg.n_heads)
        
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn.b_K"] = b_K
        state_dict[f"blocks.{l}.attn.b_V"] = b_V
        state_dict[f"blocks.{l}.attn.b_O"] = b_O
        
        # MLP
        # LayerNorm2
        state_dict[f"blocks.{l}.ln2.w"] = torch.tensor(np.array(param[f"layer_norm_{2*l+1}"]["scale"]))
        state_dict[f"blocks.{l}.ln2.b"] = torch.tensor(np.array(param[f"layer_norm_{2*l+1}"]["offset"]))
        
        # MLP weights
        if l == 0:
            state_dict[f"blocks.{l}.mlp.W_in"] = torch.tensor(np.array(param[f"linear_1"]["w"]))
            state_dict[f"blocks.{l}.mlp.W_gate"] = torch.tensor(np.array(param[f"linear"]["w"]))
            state_dict[f"blocks.{l}.mlp.W_out"] = torch.tensor(np.array(param[f"linear_2"]["w"]))
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = torch.tensor(np.array(param[f"linear_{3*l+1}"]["w"]))
            state_dict[f"blocks.{l}.mlp.W_gate"] = torch.tensor(np.array(param[f"linear_{3*l}"]["w"]))
            state_dict[f"blocks.{l}.mlp.W_out"] = torch.tensor(np.array(param[f"linear_{3*l+2}"]["w"]))
        
        # 偏置如果没有可以用0填充
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.b_gate"] = torch.zeros(cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model)

    # 3. Final LayerNorm
    state_dict["ln_final.w"] = torch.tensor(np.array(param["layer_norm_32"]["scale"]))
    state_dict["ln_final.b"] = torch.tensor(np.array(param["layer_norm_32"]["offset"]))

    # 4. Final linear layer
    state_dict["linear_final.weight"] = torch.tensor(np.array(param["linear_48"]["w"])).T
    state_dict["linear_final.bias"] = torch.tensor(np.array(param["linear_48"]["b"]))

    return state_dict


# 创建配置对象
cfg = HookedTransformerConfig(
    d_model=1024,
    d_head=1024 // 8,
    n_heads=8,
    d_mlp=1024 * 4,
    n_layers=16,
    n_ctx=79,
    eps=1e-05,
    act_fn="silu",
    normalization_type="LN",
    is_chess_model=True,
    gated_mlp=True,
    attention_dir="bidirectional",
    d_vocab=1968,
    shift_right=True,
)

print("开始转换权重...")
# 转换权重
state_dict = convert_searchless_chess_weights(params, cfg)

print(f"转换完成，共 {len(state_dict)} 个参数")
print("参数列表:")
for key in sorted(state_dict.keys()):
    print(f"  {key}: {state_dict[key].shape}")

# 保存为 PyTorch 格式
output_path = os.path.join(PROJECT_ROOT, "searchless_chess_pytorch.pt")

# 创建完整的 checkpoint 数据
checkpoint_data = {
    'state_dict': state_dict,
    'model_config': {
        'd_model': 1024,
        'd_head': 128,
        'n_heads': 8,
        'd_mlp': 4096,
        'n_layers': 16,
        'n_ctx': 79,
        'act_fn': 'silu',
        'normalization_type': 'LN',
        'is_chess_model': True,
        'gated_mlp': True,
        'attention_dir': 'bidirectional',
        'd_vocab': 1968,
        'shift_right': True,
        'num_return_buckets': 128,
        'tokenizer_name': 'searchless-chess',
    }
}

print(f"正在保存到: {output_path}")
torch.save(checkpoint_data, output_path)

print(f"✅ 成功保存 PyTorch checkpoint 到: {output_path}")
print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

# 验证保存的文件
print("\n🔍 验证保存的文件...")
try:
    loaded_data = torch.load(output_path, map_location='cpu')
    print(f"✅ 文件加载成功")
    print(f"包含的键: {list(loaded_data.keys())}")
    print(f"state_dict 中的参数数量: {len(loaded_data['state_dict'])}")
    
    # 计算总参数数量
    total_params = sum(p.numel() for p in loaded_data['state_dict'].values())
    print(f"总参数数量: {total_params:,}")
    
except Exception as e:
    print(f"❌ 验证失败: {e}")

print("\n🎉 转换完成！现在你可以使用这个 PyTorch checkpoint 了。") 