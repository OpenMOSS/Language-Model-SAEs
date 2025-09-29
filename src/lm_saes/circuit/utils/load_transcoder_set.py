from typing import Dict, List, Tuple, Union

import torch

from lm_saes import CrossLayerTranscoder, SparseAutoEncoder


def load_transcoder_set(
    transcoder_set: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Union[CrossLayerTranscoder, Dict[int, SparseAutoEncoder]], str, str]:
    if isinstance(transcoder_set, str):
        clt = CrossLayerTranscoder.from_pretrained(
            transcoder_set,
            device=device,
            dtype=dtype,
        )

        assert len(set(".".join(clt.cfg.hook_points_in[l].split(".")[2:]) for l in range(clt.cfg.n_layers))) == 1, (
            "CLT must have exactly one hook point position for input and output"
        )
        # feature_input_hook = ".".join(clt.cfg.hook_points_in[0].split(".")[2:])  # 2: is to remove the "blocks.L" prefix
        # feature_output_hook = ".".join(clt.cfg.hook_points_out[0].split(".")[2:])
        return clt, "mlp.hook_in", "mlp.hook_out"

    elif isinstance(transcoder_set, List):
        transcoders = {
            i: SparseAutoEncoder.from_pretrained(transcoder, device=device, dtype=dtype)
            for i, transcoder in enumerate(transcoder_set)
        }
        assert (
            len(
                set(
                    ".".join(list(transcoders.values())[0].cfg.hook_point_in[l].split(".")[2:])
                    for l in range(len(transcoders))
                )
            )
            == 1
        ), "All transcoders must have exactly one hook point position for input and output"
        feature_input_hook = ".".join(
            list(transcoders.values())[0].cfg.hook_point_in.split(".")[2:]
        )  # 2: is to remove the "blocks.L" prefix
        feature_output_hook = ".".join(list(transcoders.values())[0].cfg.hook_point_out.split(".")[2:])
        return transcoders, feature_input_hook, feature_output_hook

    else:
        raise ValueError(
            f"Transcoder set {transcoder_set} is not a string (loading CLTs) or list (loading a set of transcoders)"
        )
