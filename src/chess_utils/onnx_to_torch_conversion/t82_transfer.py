from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnx  # type: ignore
    import onnx2torch  # type: ignore

    HAS_ONNX_SUPPORT = True
except Exception:
    HAS_ONNX_SUPPORT = False


class AttentionBody(nn.Module):
    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_embedding = nn.Linear(176, self.d_model)
        self.ma_gating_mul = nn.Parameter(torch.randn(64, self.d_model))
        self.ma_gating_add = nn.Parameter(torch.randn(64, self.d_model))
        self.pos_encoding_base = nn.Parameter(torch.randn(1, 64, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, 64, 112)
        pos_encoding = self.pos_encoding_base.expand(batch_size, 64, 64)
        x = torch.cat([x, pos_encoding], dim=-1)
        x = self.input_embedding(x)
        x = F.mish(x)
        x = x * self.ma_gating_mul.unsqueeze(0)
        x = x + self.ma_gating_add.unsqueeze(0)
        return x


class SmolGen(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 24) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.compress = nn.Linear(d_model, 32, bias=False)
        self.dense1 = nn.Linear(2048, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dense2 = nn.Linear(256, 6144)
        self.ln2 = nn.LayerNorm(6144)
        self.smol_weight_gen = nn.Linear(256, 4096, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        compressed = self.compress(x)
        x_flat = compressed.view(batch_size, -1)
        x = self.dense1(x_flat)
        x = F.silu(x)
        x = self.ln1(x)
        x = self.dense2(x)
        x = F.silu(x)
        x = self.ln2(x)
        x = x.view(batch_size, self.n_heads, 256)
        weights = self.smol_weight_gen(x)
        weights = weights.view(batch_size, self.n_heads, 64, 64)
        return weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 24) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_k = self.d_model // self.n_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.qk_scale = nn.Parameter(torch.tensor([1.0 / (self.d_k**0.5)]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale
        return scores, v


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 24, d_ff: int = 1024) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.mha = MultiHeadAttention(self.d_model, self.n_heads)
        self.smolgen = SmolGen(self.d_model, self.n_heads)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ffn_dense1 = nn.Linear(self.d_model, int(d_ff))
        self.ffn_dense2 = nn.Linear(int(d_ff), self.d_model)
        self.alpha_input = nn.Parameter(torch.ones(1))
        self.alpha_out1 = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_scores, v = self.mha(x)
        smol_weights = self.smolgen(x)
        combined_scores = attn_scores + smol_weights.unsqueeze(1)
        attn_weights = F.softmax(combined_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        batch_size, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads
        attn_out = (
            attn_out.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, seq_len, self.n_heads * head_dim)
        )
        attn_out = self.mha.out_proj(attn_out)

        x = attn_out + (residual * self.alpha_input)
        x = self.ln1(x)
        residual2 = x

        ffn_out = self.ffn_dense1(x)
        ffn_out = F.relu(ffn_out) ** 2
        ffn_out = self.ffn_dense2(ffn_out)

        x = ffn_out + (residual2 * self.alpha_out1)
        x = self.ln2(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, d_model: int = 768, policy_dim: int = 1858, n_heads: int = 24) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.dense1 = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.ones(1))
        self.promotion = nn.Linear(d_model, 4, bias=False)
        self.indices = nn.Parameter(torch.randn(policy_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = F.mish(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale

        promotion_slice = k[:, 56:64, :]
        promotion_out = self.promotion(promotion_slice)
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out_part1, promotion_out_part2 = torch.split(promotion_out, [3, 1], dim=1)
        promotion_out = torch.add(promotion_out_part1, promotion_out_part2)
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out = promotion_out.reshape(x.shape[0], 1, self.n_heads)

        promotion_slice2 = scores[:, 48:56, 56:64]
        promotion_out2 = promotion_slice2.reshape(-1, 64, 1)
        promotion_out2 = torch.cat([promotion_out2, promotion_out2, promotion_out2], dim=-1)
        promotion_out2 = promotion_out2.reshape(-1, 8, self.n_heads)
        promotion = promotion_out2 + promotion_out
        promotion = promotion.reshape(-1, 3, 64)

        policy = torch.cat([scores, promotion], dim=1)
        policy = policy.reshape(-1, 4288)
        indices_long = self.indices.detach().long()
        policy_logits = policy[:, indices_long]
        return policy_logits


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.embed = nn.Linear(self.d_model, 32)
        self.dense1 = nn.Linear(32 * 64, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size * 64, self.d_model)
        x = self.embed(x)
        x = F.mish(x)
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        x = F.mish(x)
        x = self.dense2(x)
        wdl = F.softmax(x, dim=-1)
        return wdl


class MLHHead(nn.Module):
    def __init__(self, d_model: int = 768) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.embed = nn.Linear(self.d_model, 8)
        self.dense1 = nn.Linear(8 * 64, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size * 64, self.d_model)
        x = self.embed(x)
        x = F.mish(x)
        x = x.view(batch_size, -1)
        x = self.dense1(x)
        x = F.mish(x)
        mlh = self.dense2(x)
        mlh = F.mish(mlh)
        return mlh


class CleanLC0Model(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 24,
        n_layers: int = 15,
        d_ff: int = 1024,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.max_seq_len = int(max_seq_len)

        self.attention_body = AttentionBody(self.d_model)
        self.encoders = nn.ModuleList(
            [EncoderLayer(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)]
        )
        self.policy_head = PolicyHead(self.d_model, n_heads=self.n_heads)
        self.value_head = ValueHead(self.d_model)
        self.mlh_head = MLHHead(self.d_model)

    def forward(self, board_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.attention_body(board_features)
        for encoder in self.encoders:
            x = encoder(x)
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        mlh_logits = self.mlh_head(x)
        return policy_logits, value_logits, mlh_logits

    def create_weight_mapping_from_graph(self, _onnx_model: Any) -> dict[str, str]:
        mapping: dict[str, str] = {}

        mapping.update(
            {
                "attention_body.input_embedding.weight": "initializers.onnx_initializer_6",
                "attention_body.input_embedding.bias": "initializers.onnx_initializer_7",
                "attention_body.ma_gating_mul": "initializers.onnx_initializer_9",
                "attention_body.ma_gating_add": "initializers.onnx_initializer_10",
                "attention_body.pos_encoding_base": "initializers.onnx_initializer_4",
                "_attn_body_reshape_0": "initializers.onnx_initializer_0",
                "_attn_body_batch_1": "initializers.onnx_initializer_1",
                "_attn_body_pos_encoding_3": "initializers.onnx_initializer_3",
                "_attn_body_reshape2_5": "initializers.onnx_initializer_5",
            }
        )

        for i in range(self.n_layers):
            encoder_prefix = f"encoders.{i}"
            base_idx = 12 + i * 28
            mapping.update(
                {
                    f"{encoder_prefix}.mha.q_proj.weight": f"initializers.onnx_initializer_{base_idx}",
                    f"{encoder_prefix}.mha.q_proj.bias": f"initializers.onnx_initializer_{base_idx + 1}",
                    f"{encoder_prefix}.mha.k_proj.weight": f"initializers.onnx_initializer_{base_idx + 3}",
                    f"{encoder_prefix}.mha.k_proj.bias": f"initializers.onnx_initializer_{base_idx + 4}",
                    f"{encoder_prefix}.mha.v_proj.weight": f"initializers.onnx_initializer_{base_idx + 6}",
                    f"{encoder_prefix}.mha.v_proj.bias": f"initializers.onnx_initializer_{base_idx + 7}",
                    f"{encoder_prefix}.mha.out_proj.weight": f"initializers.onnx_initializer_{base_idx + 20}",
                    f"{encoder_prefix}.mha.out_proj.bias": f"initializers.onnx_initializer_{base_idx + 21}",
                }
            )
            mapping.update(
                {
                    f"{encoder_prefix}.smolgen.compress.weight": f"initializers.onnx_initializer_{base_idx + 10}",
                    f"{encoder_prefix}.smolgen.dense1.weight": f"initializers.onnx_initializer_{base_idx + 12}",
                    f"{encoder_prefix}.smolgen.dense1.bias": f"initializers.onnx_initializer_{base_idx + 13}",
                    f"{encoder_prefix}.smolgen.dense2.weight": f"initializers.onnx_initializer_{base_idx + 14}",
                    f"{encoder_prefix}.smolgen.dense2.bias": f"initializers.onnx_initializer_{base_idx + 15}",
                    f"{encoder_prefix}.smolgen.smol_weight_gen.weight": f"initializers.onnx_initializer_{base_idx + 17}",
                }
            )
            mapping.update(
                {
                    f"{encoder_prefix}.ffn_dense1.weight": f"initializers.onnx_initializer_{base_idx + 23}",
                    f"{encoder_prefix}.ffn_dense1.bias": f"initializers.onnx_initializer_{base_idx + 24}",
                    f"{encoder_prefix}.ffn_dense2.weight": f"initializers.onnx_initializer_{base_idx + 25}",
                    f"{encoder_prefix}.ffn_dense2.bias": f"initializers.onnx_initializer_{base_idx + 26}",
                }
            )

            alpha_input_idx = 34 + i * 28
            alpha_out1_idx = 39 + i * 28
            mapping.update(
                {
                    f"{encoder_prefix}.alpha_input": f"initializers.onnx_initializer_{alpha_input_idx}",
                    f"{encoder_prefix}.alpha_out1": f"initializers.onnx_initializer_{alpha_out1_idx}",
                }
            )

            qk_scale_idx = 21 + i * 28
            mapping[f"{encoder_prefix}.mha.qk_scale"] = f"initializers.onnx_initializer_{qk_scale_idx}"
            mapping.update(
                {
                    f"{encoder_prefix}.ln1.weight": f"encoder{i}/ln1.weight",
                    f"{encoder_prefix}.ln1.bias": f"encoder{i}/ln1.bias",
                    f"{encoder_prefix}.ln2.weight": f"encoder{i}/ln2.weight",
                    f"{encoder_prefix}.ln2.bias": f"encoder{i}/ln2.bias",
                }
            )
            mapping.update(
                {
                    f"{encoder_prefix}.smolgen.ln1.weight": f"encoder{i}/smolgen/ln1.weight",
                    f"{encoder_prefix}.smolgen.ln1.bias": f"encoder{i}/smolgen/ln1.bias",
                    f"{encoder_prefix}.smolgen.ln2.weight": f"encoder{i}/smolgen/ln2.weight",
                    f"{encoder_prefix}.smolgen.ln2.bias": f"encoder{i}/smolgen/ln2.bias",
                }
            )

            shape_indices = [2, 5, 8, 11, 14, 17, 20, 21, 23, 28, 30, 31, 42, 45, 48, 49, 51, 56, 58, 59]
            for j, shape_idx in enumerate(shape_indices):
                actual_idx = base_idx + shape_idx - 12
                if 0 <= actual_idx < 467:
                    mapping[f"_shape_param_{i}_{j}"] = f"initializers.onnx_initializer_{actual_idx}"

        mapping.update(
            {
                "policy_head.dense1.weight": "initializers.onnx_initializer_432",
                "policy_head.dense1.bias": "initializers.onnx_initializer_433",
                "policy_head.q_proj.weight": "initializers.onnx_initializer_434",
                "policy_head.q_proj.bias": "initializers.onnx_initializer_435",
                "policy_head.k_proj.weight": "initializers.onnx_initializer_437",
                "policy_head.k_proj.bias": "initializers.onnx_initializer_438",
                "policy_head.scale": "initializers.onnx_initializer_440",
                "policy_head.promotion.weight": "initializers.onnx_initializer_443",
                "policy_head.indices": "initializers.onnx_initializer_452",
                "_policy_reshape_436": "initializers.onnx_initializer_436",
                "_policy_constant_439": "initializers.onnx_initializer_439",
                "_policy_slice_442": "initializers.onnx_initializer_442",
                "_policy_shape_444": "initializers.onnx_initializer_444",
                "_policy_reshape_445": "initializers.onnx_initializer_445",
                "_policy_reshape_446": "initializers.onnx_initializer_446",
                "_policy_reshape_447": "initializers.onnx_initializer_447",
                "_policy_reshape_449": "initializers.onnx_initializer_449",
                "value_head.embed.weight": "initializers.onnx_initializer_453",
                "value_head.embed.bias": "initializers.onnx_initializer_454",
                "_value_shape_455": "initializers.onnx_initializer_455",
                "value_head.dense1.weight": "initializers.onnx_initializer_456",
                "value_head.dense1.bias": "initializers.onnx_initializer_457",
                "value_head.dense2.weight": "initializers.onnx_initializer_458",
                "value_head.dense2.bias": "initializers.onnx_initializer_459",
                "mlh_head.embed.weight": "initializers.onnx_initializer_460",
                "mlh_head.embed.bias": "initializers.onnx_initializer_461",
                "_mlh_shape_462": "initializers.onnx_initializer_462",
                "mlh_head.dense1.weight": "initializers.onnx_initializer_463",
                "mlh_head.dense1.bias": "initializers.onnx_initializer_464",
                "mlh_head.dense2.weight": "initializers.onnx_initializer_465",
                "mlh_head.dense2.bias": "initializers.onnx_initializer_466",
            }
        )

        return mapping

    def load_from_onnx_model(self, onnx_model: nn.Module, *, verbose: bool = True) -> int:
        onnx_state_dict = onnx_model.state_dict()
        initializers_weights: dict[str, torch.Tensor] = {}
        direct_weights: dict[str, torch.Tensor] = {}
        for key, tensor in onnx_state_dict.items():
            if key.startswith("initializers."):
                initializers_weights[key] = tensor
            else:
                direct_weights[key] = tensor

        graph_mapping = self.create_weight_mapping_from_graph(onnx_model)
        all_onnx_weights = {**initializers_weights, **direct_weights}

        matched_weights = 0
        for name, param in self.named_parameters():
            if name not in graph_mapping:
                continue
            onnx_name = graph_mapping[name]
            if onnx_name not in all_onnx_weights:
                continue
            onnx_weight = all_onnx_weights[onnx_name]
            try:
                if any(
                    token in name
                    for token in (
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "out_proj.weight",
                        "policy_head.dense1.weight",
                    )
                ):
                    if tuple(param.shape) == tuple(reversed(onnx_weight.shape)):
                        param.data.copy_(onnx_weight.T)
                        matched_weights += 1
                    elif tuple(param.shape) == tuple(onnx_weight.shape):
                        param.data.copy_(onnx_weight)
                        matched_weights += 1
                else:
                    if tuple(param.shape) == tuple(onnx_weight.shape):
                        param.data.copy_(onnx_weight)
                        matched_weights += 1
                    elif tuple(param.shape) == tuple(reversed(onnx_weight.shape)):
                        param.data.copy_(onnx_weight.T)
                        matched_weights += 1
            except Exception:
                continue

        used_onnx_keys = set(graph_mapping.values())
        shape_matched = 0
        for name, param in self.named_parameters():
            if name in graph_mapping:
                continue
            for onnx_name, onnx_weight in all_onnx_weights.items():
                if onnx_name in used_onnx_keys:
                    continue
                if any(
                    token in name
                    for token in (
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "out_proj.weight",
                        "policy_head.dense1.weight",
                    )
                ):
                    if tuple(param.shape) == tuple(reversed(onnx_weight.shape)):
                        param.data.copy_(onnx_weight.T)
                        used_onnx_keys.add(onnx_name)
                        shape_matched += 1
                        break
                    if tuple(param.shape) == tuple(onnx_weight.shape):
                        param.data.copy_(onnx_weight)
                        used_onnx_keys.add(onnx_name)
                        shape_matched += 1
                        break
                else:
                    if tuple(param.shape) == tuple(onnx_weight.shape):
                        param.data.copy_(onnx_weight)
                        used_onnx_keys.add(onnx_name)
                        shape_matched += 1
                        break
                    if tuple(param.shape) == tuple(reversed(onnx_weight.shape)):
                        param.data.copy_(onnx_weight.T)
                        used_onnx_keys.add(onnx_name)
                        shape_matched += 1
                        break

        total_params = len(list(self.named_parameters()))
        total_matched = matched_weights + shape_matched
        if verbose:
            unused_onnx = [k for k in all_onnx_weights.keys() if k not in used_onnx_keys]
            print(
                f"Loaded parameters: mapped={matched_weights}, shape_matched={shape_matched}, "
                f"total_params={total_params}, total_loaded={total_matched}, unused_onnx={len(unused_onnx)}"
            )
        return total_matched


@torch.no_grad()
def convert_t82_onnx_to_clean_state_dict(
    *,
    onnx_model_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    verbose: bool = True,
) -> Path:
    if not HAS_ONNX_SUPPORT:
        raise ImportError("onnx and onnx2torch are required")

    onnx_model = onnx.load(str(onnx_model_path))
    converted_model = onnx2torch.convert(onnx_model)
    converted_model.to(device)

    clean_model = CleanLC0Model()
    clean_model.to(device)
    clean_model.load_from_onnx_model(converted_model, verbose=verbose)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_model.state_dict(), str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert T82 ONNX weights to CleanLC0Model state_dict.")
    p.add_argument("--onnx", required=True, help="Path to T82 ONNX file.")
    p.add_argument(
        "--out",
        default="/inspire/hdd/global_user/hezhengfu-240208120186/models/chess/LC0/T82/T82.pt",
        help="Output path for torch state_dict (.pt).",
    )
    p.add_argument("--device", default="cuda", help="Device for conversion (e.g., cuda or cpu).")
    p.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    out_path = convert_t82_onnx_to_clean_state_dict(
        onnx_model_path=args.onnx,
        output_path=args.out,
        device=device,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()