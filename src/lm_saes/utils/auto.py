import importlib.util
import os
from enum import Enum

from huggingface_hub import constants, repo_exists, try_to_load_from_cache


class PretrainedSAEType(Enum):
    """The type of a pretrained sparse dictionary. Will determine the method to load the config and model."""

    LOCAL = "local"
    """Pretrained sparse dictionary is stored in a local directory."""

    HUGGINGFACE = "huggingface"
    """Pretrained sparse dictionary is stored in HuggingFace Hub, in a format consistent with local storage."""

    SAELENS = "saelens"
    """Pretrained sparse dictionary is registered in SAELens `pretrained_saes.yaml` file."""


def auto_infer_pretrained_sae_type(pretrained_name_or_path: str) -> PretrainedSAEType:
    """Automatically infer the type of a pretrained sparse dictionary based on the name or path."""

    if os.path.exists(pretrained_name_or_path):
        return PretrainedSAEType.LOCAL

    if not len(pretrained_name_or_path.split(":")) == 2:
        raise ValueError(
            f"Pretrained name or path {pretrained_name_or_path} is not on disk, nor given in HuggingFace/SAELens compatible format <repo_id/release>:<name>."
        )

    repo_id, name = pretrained_name_or_path.split(":")

    if importlib.util.find_spec("sae_lens") is not None:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

        lookups = get_pretrained_saes_directory()
        if lookups.get(repo_id) is not None and lookups[repo_id].saes_map.get(name) is not None:
            return PretrainedSAEType.SAELENS

    if constants.HF_HUB_OFFLINE:
        if try_to_load_from_cache(repo_id=repo_id, filename=f"{name}/config.json"):
            return PretrainedSAEType.HUGGINGFACE
        else:
            raise ValueError(
                f"Pretrained name or path {pretrained_name_or_path} is not found on disk nor in SAELens, and HuggingFace Hub is not accessible."
            )

    elif repo_exists(repo_id=repo_id):
        return PretrainedSAEType.HUGGINGFACE

    likely_saelens = "/" not in repo_id
    if likely_saelens:
        if importlib.util.find_spec("sae_lens") is None:
            raise ValueError(
                f"Pretrained name or path {pretrained_name_or_path} is likely in SAELens format, but SAELens is not installed."
            )
        else:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

            lookups = get_pretrained_saes_directory()
            if lookups.get(repo_id) is None:
                raise ValueError(
                    f"Pretrained name or path {pretrained_name_or_path} is likely in SAELens format, but {repo_id} is not a valid SAELens release. If you are sure this is a valid SAELens release, try upgrading SAELens to the latest version."
                )
            if lookups[repo_id].saes_map.get(name) is None:
                raise ValueError(
                    f"Pretrained name or path {pretrained_name_or_path} is likely in SAELens format, but {name} is not a valid ID in release {repo_id}. If you are sure this is a valid ID, try upgrading SAELens to the latest version."
                )
            return PretrainedSAEType.SAELENS

    raise ValueError(
        f"Pretrained name or path {pretrained_name_or_path} is not found on disk, nor on HuggingFace, nor in SAELens."
    )
