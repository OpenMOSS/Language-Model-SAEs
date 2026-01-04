from typing import Any, Iterable, Optional

import torch

from lm_saes.backend.language_model import LanguageModel
from lm_saes.utils.misc import pad_and_truncate_tokens

from .core import BaseActivationProcessor


class RawDatasetTokenProcessor(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for converting raw token datasets into model-ready token format.

    This processor takes an iterable of dictionaries containing raw data (e.g. text and images) and converts
    them into a tokens. The output is a dictionary with a "tokens" key, which contains the (non-padded and non-truncated)
    tokens. The "meta" key is preserved if it exists in the input.

    Args:
        prepend_bos: Whether to prepend beginning-of-sequence token. If None, uses model default.
    """

    def __init__(self, prepend_bos: bool | None = None):
        self.prepend_bos = prepend_bos

    def process(self, data: Iterable[dict[str, Any]], *, model: LanguageModel, **kwargs) -> Iterable[dict[str, Any]]:
        """Process raw data into tokens.

        Args:
            data: Iterable of dictionaries containing raw data (e.g. text and images)
            model: model to use for producing tokens
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict: Processed token data with optional info field
        """
        for d in data:
            # TODO: support TransformerLens backend
            tokens = model.to_tokens(d, prepend_bos=self.prepend_bos)  # type: ignore
            ret = {"tokens": tokens[0]}
            if "meta" in d:
                ret = ret | {"meta": d["meta"]}
            yield ret


class PadAndTruncateTokensProcessor(BaseActivationProcessor[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]):
    """Processor for padding and truncating tokens to a desired sequence length.

    This processor takes an iterable of dictionaries containing tokens and pads them to a desired sequence length.
    The output is a dictionary with a "tokens" key, which contains the padded tokens.

    Args:
        seq_len (int): The desired sequence length to pad/truncate to
        pad_token_id (int, optional): The token ID to use for padding. Defaults to 0.
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def process(
        self,
        data: Iterable[dict[str, Any]],
        *,
        pad_token_id: Optional[int] = None,
        model: Optional[LanguageModel] = None,
        **kwargs,
    ) -> Iterable[dict[str, Any]]:
        """Process tokens by padding or truncating to desired sequence length.

        Args:
            data (Iterable[dict[str, Any]]): Input data containing tokens to process
            pad_token_id (int, optional): The token ID to use for padding. Defaults to None.
                If not specified, the pad_token_id will be inferred from the model's tokenizer.
                If neither is provided, the pad_token_id will be 0.
            model (LanguageModel, optional): The model to use for padding. Defaults to None.
                If provided, the pad_token_id will be inferred from the model's tokenizer.
            **kwargs: Additional keyword arguments. Not used by this processor.

        Yields:
            dict[str, Any]: Dictionary containing processed tokens padded/truncated to seq_len,
                and original info field if present
        """

        # Infer pad_token_id if not provided
        if pad_token_id is None:
            if model is not None:
                pad_token_id = model.pad_token_id
            else:
                pad_token_id = 0

        for d in data:
            assert "tokens" in d and isinstance(d["tokens"], torch.Tensor)
            assert pad_token_id is not None, "pad_token_id must be provided"
            tokens = pad_and_truncate_tokens(d["tokens"], seq_len=self.seq_len, pad_token_id=pad_token_id)
            ret = {"tokens": tokens}
            if "meta" in d:
                ret = ret | {"meta": d["meta"]}
            yield ret
