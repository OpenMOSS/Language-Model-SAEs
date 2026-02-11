import re
import warnings
from abc import ABC, abstractmethod
from itertools import accumulate
from typing import Any, Optional, cast
import os

import torch
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from transformer_lens.components import SearchlessChessBehavioralCloningTokenizer, LeelaBoard, LeelaEmbed
from lm_saes.config import LanguageModelConfig, LLaDAConfig
from lm_saes.utils.misc import pad_and_truncate_tokens
from lm_saes.utils.timer import timer

def fen_to_longfen_behavioral_cloning(fen: str) -> str:
    '''
    input: fen
    output: longfen(wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..0)
    '''
    parts = fen.split()
    board_fen = parts[0]  # 棋盘部分
    active_color = parts[1]  # 当前走棋方 (w/b)
    castling = parts[2]  # 王车易位权利
    en_passant = parts[3]  # 过路兵目标格
    halfmove = parts[4]  # 半回合计数
    fullmove = parts[5]  # 全回合计数
    
    # 转换棋盘部分 (8x8 = 64个字符)
    longfen_board = ""
    for char in board_fen:
        if char == '/':
            continue  # 跳过行分隔符
        elif char.isdigit():
            # 数字表示连续的空格数
            longfen_board += '.' * int(char)
        else:
            # 棋子字符直接添加
            longfen_board += char
    
    # 确保棋盘部分正好64个字符
    assert len(longfen_board) == 64, f"棋盘应该有64个字符，实际有{len(longfen_board)}个"
    
    # 处理王车易位权利 (4个字符位置：KQkq)
    castling_longfen = ""
    for right in ['K', 'Q', 'k', 'q']:
        if right in castling:
            castling_longfen += right
        else:
            castling_longfen += '.'
    
    # 处理过路兵 (2个字符)
    if en_passant == '-':
        en_passant_longfen = ".."
    else:
        en_passant_longfen = en_passant
    
    # 处理半回合和全回合计数 (各3个字符)
    # 使用左对齐，右侧填充.
    halfmove_padded = halfmove.ljust(3, '.')  # 左对齐，右侧填充.
    fullmove_padded = fullmove.ljust(3, '.')  # 左对齐，右侧填充.
    
    # 组装longfen字符串
    longfen_behavioral_cloning = (
        active_color +  # 当前走棋方 (1字符)
        longfen_board +  # 棋盘 (64字符)
        castling_longfen +  # 王车易位 (4字符)
        en_passant_longfen +  # 过路兵 (2字符)
        halfmove_padded +  # 半回合 (3字符)
        fullmove_padded +  # 全回合 (3字符)
        # move +  # 走法
        "0"  # 结束标记
    )
    
    return longfen_behavioral_cloning

def fen_to_board_str(fen: str) -> str:
    '''
    input: fen
    output: board_str(wrnbqkbnrpppppppp................................PPPPPPPPRNBQKBNRKQkq..0..1..0)
    '''
    parts = fen.split()
    board_fen = parts[0]  # 棋盘部分    
    
    board_str = ""
    for char in board_fen:
        if char == '/':
            continue  # 跳过行分隔符
        elif char.isdigit():
            # 数字表示连续的空格数
            board_str += '.' * int(char)
        else:
            # 棋子字符直接添加
            board_str += char   
    return board_str


def get_input_with_manually_prepended_bos(tokenizer, input):
    """
    Manually prepends the bos token to the input.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for prepending the bos token.
        input (Union[str, List[str]]): The input to prepend the bos token to.

    Returns:
        Union[str, List[str]]: The input with the bos token manually prepended.
    """
    if isinstance(input, str):
        input = tokenizer.bos_token + input
    else:
        input = [tokenizer.bos_token + string for string in input]
    return input


def to_tokens(tokenizer, text, max_length, device="cpu"):
    tokenizer_prepends_bos = len(tokenizer.encode("")) > 0
    text = text if not tokenizer_prepends_bos else get_input_with_manually_prepended_bos(tokenizer, text)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return tokens.to(device)


def set_tokens(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


def _match_str_tokens_to_input(text: str, str_tokens: list[str]) -> list[Optional[tuple[int, int]]]:
    """Match the tokens to the input text, returning a list of tuples of the form (start_idx, end_idx) for each token."""
    # Initialize list to store token positions
    token_positions = []

    # Keep track of current position in text
    curr_pos = 0

    # For each token, try to find its position in the input text
    for token in str_tokens:
        # Search for token in remaining text
        pos = text.find(token, curr_pos)

        if pos != -1:
            # Found a match, store position and update curr_pos
            token_positions.append({"key": "text", "range": (pos, pos + len(token))})
            curr_pos = pos + len(token)
        else:
            # No match found. This is only allowed if the token is a special token
            # that doesn't appear in the input text, or if the token is a subword token
            # which cannot be decoded separately.
            # TODO: Deal with subword tokens properly
            if not ((token.startswith("<") and token.endswith(">")) or "�" in token):
                raise ValueError(f"Token {token} not found in input text")
            token_positions.append(None)

    return token_positions


# def _get_layer_indices_from_hook_points(hook_points: list[str]) -> list[int]:
#     residual_pattern = r"^blocks\.(\d+)\.hook_mlp_out$"
#     matches = [re.match(residual_pattern, hook_point) for hook_point in hook_points]
#     assert all(match is not None for match in matches), "hook_points must be residual stream hook points"
#     layer_indices = [int(cast(re.Match[str], match).group(1)) for match in matches]
#     return layer_indices

def _get_layer_indices_from_hook_points(hook_points: list[str]) -> list[int]:
    # 支持的 hook 点样式
    patterns = [
        r"^blocks\.(\d+)\.hook_mlp_out$",
        r"^blocks\.(\d+)\.resid_mid_after_ln$",
        r"^blocks\.(\d+)\.resid_post_after_ln$",
        r"^blocks\.(\d+)\.hook_attn_in$",
        r"^blocks\.(\d+)\.hook_attn_out$",
    ]

    layer_indices: list[int] = []
    for hook_point in hook_points:
        matched = False
        for pat in patterns:
            m = re.match(pat, hook_point)
            if m:
                layer_indices.append(int(m.group(1)))
                matched = True
                break
        if not matched:
            raise ValueError(
                f"hook_point '{hook_point}' 必须匹配以下任一格式："
                " blocks.<L>.hook_mlp_out | blocks.<L>.resid_mid_after_ln | "
                "blocks.<L>.resid_post_after_ln | blocks.<L>.hook_attn_in | blocks.<L>.hook_attn_out"
            )
    return layer_indices

class LanguageModel(ABC):
    @abstractmethod
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        """Trace how raw data is eventually aligned with tokens.

        Args:
            raw (dict[str, Any]): The raw data to trace.

        Returns:
            list[list[Any]]: The origins of the tokens in the raw data. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        """Convert raw data to activations.

        Args:
            raw (dict[str, Any]): The raw data to convert to activations.
            hook_points (list[str]): The hook points to use for activations.

        Returns:
            dict[str, torch.Tensor]: The activations. Shape: (batch_size, n_tokens, n_activations)
        """
        pass

    @abstractmethod
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the raw data.

        Args:
            raw (dict[str, Any]): The raw data to preprocess.

        Returns:
            dict[str, Any]: The preprocessed raw data.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """The ID of the end-of-sequence token."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """The ID of the beginning-of-sequence token."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """The ID of the padding token."""
        pass


class TransformerLensLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
                cache_dir=cfg.cache_dir,
                local_files_only=cfg.local_files_only,
                torch_dtype=cfg.dtype,
                trust_remote_code=True,
            )
            if cfg.load_ckpt
            else None
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            (cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path),
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.tokenizer = set_tokens(hf_tokenizer)
        self.model = (
            HookedTransformer.from_pretrained_no_processing(
                cfg.model_name,
                use_flash_attn=cfg.use_flash_attn,
                device=self.device,
                cache_dir=cfg.cache_dir,
                hf_model=hf_model,
                hf_config=hf_model.config,
                tokenizer=hf_tokenizer,
                dtype=cfg.dtype,  # type: ignore ; issue with transformer_lens
            )
            if hf_model
            else None
        )

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Tracing with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        tokens = to_tokens(self.tokenizer, raw["text"], max_length=self.cfg.max_length, device=self.cfg.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        decoded_str = self.tokenizer.decode(tokens)
        batch_str_tokens = list(decoded_str)
        return [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

    @timer.time("to_activations")
    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Activations with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        with timer.time("to_tokens"):
            tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        with timer.time("run_with_cache_until"):
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)
        return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens}


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw


# class SearchlessChessActionModel(TransformerLensLanguageModel):
#     def __init__(self, cfg: LanguageModelConfig):
#         # 不调用父类的 __init__ 方法，这样可以避免继承父类的 tokenizer 初始化
#         self.cfg = cfg
#         print(f"{cfg=}")
#         self.device = (
#             torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
#         )
        
#         # 如果用户指定了自定义的 checkpoint 路径，设置环境变量
#         if cfg.model_from_pretrained_path:
#             os.environ["SEARCHLESS_CHESS_PYTORCH_PATH"] = cfg.model_from_pretrained_path
        
#         # 使用 TransformerLens 的标准加载方式
#         print("Loading searchless_chess model using TransformerLens...")
#         # self.model = HookedTransformer.from_pretrained_no_processing(
#         #     'google/searchless-chess-270M',
#         #     dtype=torch.float32,
#         #     device=self.device,
#         # ).eval()
        
        
#         # 使用 SearchlessChessBehavioralCloningTokenizer
#         self.tokenizer = SearchlessChessBehavioralCloningTokenizer()

#     # 你可以根据需要覆写 tokenizer 相关的属性和方法
#     @property
#     def eos_token_id(self) -> int | None:
#         # 如果没有 tokenizer，可以直接返回 None 或者实现默认行为
#         return None

#     @property
#     def bos_token_id(self) -> int | None:
#         return None

#     @property
#     def pad_token_id(self) -> int | None:
#         return None
    
#     @torch.no_grad()
#     def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
#         assert len(raw["fen"]) == len(raw["move"]) == len(raw["move_type"]) == len(raw["meta"])
        
#         # 处理所有32个元素
#         texts = []
#         metas = []
#         for i in range(len(raw["fen"])):
#             fen = raw["fen"][i]
#             move = raw["move"][i]
#             # best_move = raw["best move"][i]
#             # random_move = raw["random move"][i]
#             input = f"{fen},{move}"
#             # input_random = f"{fen},{random_move}"
#             texts.append(input)
#             metas.append(raw["meta"][i])
        
#         preprocessed_data = {"text": texts}
#         preprocessed_data["meta"] = metas
        
#         return preprocessed_data
        
#     def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
#         fen = raw["fen"]
#         move = raw["move"]
#         # input_str = f"{fen},{move}"
#         # tokens = self.tokenizer.encode(input_str)
#         longfens = []
#         print(f"{len(fen)=}")
#         for i in range(len(fen)):
#             fen_i = fen[i]
#             move_i = move[i]
#             longfen = fen_to_longfen(fen_i, move_i)
#             longfens.append(longfen)
        
#         # 特殊分词：第77个字符到倒数第二个字符作为move token，其他字符分别作为token
#         str_tokens = []
#         for i, longfen in enumerate(longfens):
#             str_token = []
#             for j, char in enumerate(longfen):
#                 if j == 77:
#                     # move token
#                     str_token.append(longfen[77:-1])
#                     break
#                 elif j < 77:
#                     str_token.append(char)  
#             str_token.append(longfen[-1])
#             str_tokens.append(str_token)
        
#         return [
#             _match_str_tokens_to_input(longfen, str_tokens) for (longfen , str_tokens) in zip(longfens, str_tokens)
#         ]

#     @torch.no_grad()
#     def to_activations(
#         self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
#     ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]] | None]:
#         assert self.model is not None
#         layer_indices = _get_layer_indices_from_hook_points(hook_points)
#         inputs = raw["text"]
        
#         # Process single input (inputs length is 1)
#         # Convert input to tokens and get model output
#         tokens, _ = self.tokenizer(inputs)
#         log_softmax_output, cache = self.model.run_with_cache_until(tokens, names_filter=hook_points)

#         assert isinstance(log_softmax_output, torch.Tensor)

#         # Get activations for each hook point and tokens
#         activations = {hook_point: cache[hook_point] for hook_point in hook_points}
#         tokens = tokens.to(self.cfg.device)

#         # Return the activations and tokens
#         return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens}, None


class SearchlessChessBehavioralCloningModel(TransformerLensLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        # 不调用父类的 __init__ 方法，这样可以避免继承父类的 tokenizer 初始化
        self.cfg = cfg
        
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )
        
        # 如果用户指定了自定义的 checkpoint 路径，设置环境变量
        if cfg.model_from_pretrained_path:
            os.environ["SEARCHLESS_CHESS_PYTORCH_PATH"] = cfg.model_from_pretrained_path
        
        # 使用 TransformerLens 的标准加载方式
        print("Loading searchless_chess model using TransformerLens...")
        self.model = HookedTransformer.from_pretrained_no_processing(
            'google/searchless-chess-9M-behavioral-cloning',
            dtype=torch.float32,
            device=self.device,
        ).eval()
        
        
        # 使用 SearchlessChessBehavioralCloningTokenizer
        self.tokenizer = SearchlessChessBehavioralCloningTokenizer()

    # 你可以根据需要覆写 tokenizer 相关的属性和方法
    @property
    def eos_token_id(self) -> int | None:
        # 如果没有 tokenizer，可以直接返回 None 或者实现默认行为
        return None

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None
    
    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert len(raw["fen"]) == len(raw["meta"])
        
        # 处理所有32个元素
        texts = []
        metas = []
        for i in range(len(raw["fen"])):
            fen = raw["fen"][i]
            # move = raw["move"][i]
            # best_move = raw["best move"][i]
            # random_move = raw["random move"][i]
            input = fen
            # input_random = f"{fen},{random_move}"
            texts.append(input)
            metas.append(raw["meta"][i])
        
        preprocessed_data = {"text": texts}
        preprocessed_data["meta"] = metas
        
        return preprocessed_data
        
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        fen = raw["fen"]
        longfens = []
        for i in range(len(fen)):
            fen_i = fen[i]
            longfen = fen_to_longfen_behavioral_cloning(fen_i)
            longfens.append(longfen)
            
        # 每一个字符都作为一个token
        str_tokens = []
        for i, longfen in enumerate(longfens):
            str_token = []
            for j, char in enumerate(longfen):
                str_token.append(char)
            str_tokens.append(str_token)
        
        return [
            _match_str_tokens_to_input(longfen, str_tokens) for (longfen , str_tokens) in zip(longfens, str_tokens)
        ]

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = raw["text"]
        
        # Process single input (inputs length is 1)
        # Convert input to tokens and get model output
        tokens, _ = self.tokenizer(inputs)
        log_softmax_output, cache = self.model.run_with_cache_until(tokens, names_filter=hook_points)

        assert isinstance(log_softmax_output, torch.Tensor)

        # Get activations for each hook point and tokens
        activations = {hook_point: cache[hook_point] for hook_point in hook_points}
        if isinstance(tokens, list):
            if isinstance(tokens[0], torch.Tensor):
                tokens = torch.stack(tokens, dim=0)
            else:
                tokens = torch.tensor(tokens)
        tokens = tokens.to(self.cfg.device)
        return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens}



class LeelaChessModel(TransformerLensLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        # 不调用父类的 __init__ 方法，这样可以避免继承父类的 tokenizer 初始化
        self.cfg = cfg
        
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}") if cfg.device == "cuda" else torch.device(cfg.device)
        )

        if cfg.model_from_pretrained_path:
            os.environ["LEELA_PYTORCH_PATH"] = cfg.model_from_pretrained_path
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.cfg.model_name,
            dtype=torch.float32,
            device=self.device,
        ).eval()
        
        self.tokenizer = LeelaBoard()
        self.embed = LeelaEmbed(self.cfg.d_model)
        
    @property
    def eos_token_id(self) -> int | None:
        return None

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return None
    
    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert len(raw["fen"]) == len(raw["meta"])
        texts = []
        metas = []
        for i in range(len(raw["fen"])):
            fen = raw["fen"][i]
            # move = raw["move"][i]
            # best_move = raw["best move"][i]
            # random_move = raw["random move"][i]
            input = fen
            # input_random = f"{fen},{random_move}"
            texts.append(input)
            metas.append(raw["meta"][i])
        
        preprocessed_data = {"text": texts}
        preprocessed_data["meta"] = metas
        
        return preprocessed_data
        
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        fen = raw["fen"]
        longfens = []

        for i in range(len(fen)):
            fen_i = fen[i]
            longfen = fen_to_board_str(fen_i)
            longfens.append(longfen)
            
        # 每一个字符都作为一个token
        str_tokens = []
        for i, longfen in enumerate(longfens):
            str_token = []
            for j, char in enumerate(longfen):
                str_token.append(char)
            str_tokens.append(str_token)
        
        return [
            _match_str_tokens_to_input(longfen, str_tokens) for (longfen , str_tokens) in zip(longfens, str_tokens)
        ]

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = raw["text"]


        # Process single input (inputs length is 1)
        # Convert input to tokens and get model output  
        embeds = []
        for input in inputs:
            token = self.tokenizer(input)
            embed = self.embed(token)
            embed = embed.squeeze(0)  # 去掉第一个维度
            embeds.append(embed)
        # print(f"{hook_points = }")
        # outputs1, cache1 = self.model.run_with_cache(inputs, names_filter=hook_points)
        outputs, cache = self.model.run_with_cache_until(inputs, names_filter=hook_points)
        # print(cache.keys())
        logits = outputs[0]
        assert isinstance(logits, torch.Tensor)

        # Get activations for each hook point and tokens
        # activations1 = {hook_point: cache1[hook_point] for hook_point in hook_points}
        activations = {hook_point: cache[hook_point] for hook_point in hook_points}
        # print(f"{activations1 =}, {activations =}")
        
        if isinstance(embeds, list):
            if isinstance(embeds[0], torch.Tensor):
                embeds = torch.stack(embeds, dim=0)
            else:
                embeds = torch.tensor(embeds)
        # embeds = embeds.to(self.cfg.device)
        # print(f"{embeds.shape = }") #[2048, 64, 768]
        
        # 这边创建了一个全1的dummy_tokens，用于计数，保持lm_saes中的完整性
        batch_size, seq_len = embeds.shape[:2]
        dummy_tokens = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.cfg.device)

        # Return the activations and tokens
        return {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": dummy_tokens}



class LLaDALanguageModel(TransformerLensLanguageModel):
    cfg: LLaDAConfig  # Explicitly specify the type to avoid linter errors

    @property
    def mdm_mask_token_id(self) -> int:
        return self.cfg.mdm_mask_token_id

    @torch.no_grad()
    def _get_masked_tokens(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking to non-pad tokens based on mask_ratio. The random seed should be determined by the raw data.
        Args:
            tokens (torch.Tensor): The tokens to mask.
            pad_mask (torch.Tensor): The mask of the padding tokens.

        Returns:
            torch.Tensor: The tokens after masking. Pad tokens are included.
        """
        if self.cfg.mask_ratio <= 0:
            return tokens

        # Calculate random seeds based on non-pad tokens
        random_seeds = (tokens * ~pad_mask).sum(dim=1).tolist()
        masked_tokens = tokens.clone()

        # For each sequence in the batch
        for i, seed in enumerate(random_seeds):
            # Create random generator with deterministic seed
            random_generator = torch.Generator(device=tokens.device)
            random_generator.manual_seed(seed)

            # Generate random values for all positions
            random_values = torch.rand(tokens[i].shape, device=tokens[i].device, generator=random_generator)

            # Count non-pad tokens
            non_pad_mask = ~pad_mask[i]
            non_pad_count = non_pad_mask.sum()

            # Calculate number of tokens to mask
            num_to_mask = int(non_pad_count.item() * self.cfg.mask_ratio)
            if num_to_mask == 0:
                continue

            # Set random values for pad positions to infinity so they won't be selected
            random_values = torch.where(non_pad_mask, random_values, torch.full_like(random_values, float("inf")))

            # Get indices of positions to mask (lowest random values)
            mask_indices = torch.topk(random_values, k=num_to_mask, largest=False).indices

            # Apply masking
            masked_tokens[i][mask_indices] = self.mdm_mask_token_id

        return masked_tokens

    @torch.no_grad()
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        assert self.model is not None
        tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        pad_mask = tokens == self.pad_token_id
        masked_tokens = self._get_masked_tokens(tokens, pad_mask)
        raw["text"] = self.tokenizer.batch_decode(masked_tokens)
        pad_token = self.tokenizer.pad_token
        raw["text"] = [text.replace(pad_token, "") for text in raw["text"]]
        raw_meta = raw.get("meta", [])
        raw["meta"] = [
            (raw_meta[i] if i < len(raw_meta) else {}) | {"mask_ratio": self.cfg.mask_ratio}
            for i in range(len(raw["text"]))
        ]
        return raw

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        assert self.model is not None
        tokens = self.model.to_tokens(raw["text"], prepend_bos=self.cfg.prepend_bos)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for LLaDALanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        pad_mask = tokens == self.pad_token_id
        predicted_tokens = None
        if self.cfg.calculate_logits:
            logits, activations = self.model.run_with_cache(tokens, names_filter=hook_points)
            assert isinstance(logits, torch.Tensor)
            predicted_token_ids = logits.argmax(dim=-1)
            predicted_tokens = torch.where(pad_mask, -1, predicted_token_ids)
        else:
            _, activations = self.model.run_with_cache_until(tokens, names_filter=hook_points)
        predicted_tokens = {"predicted_tokens": predicted_tokens} if predicted_tokens is not None else {}
        return (
            {hook_point: activations[hook_point] for hook_point in hook_points} | {"tokens": tokens} | predicted_tokens
        )


class QwenVLLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attn else None,
        ).to(self.device)  # type: ignore

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            padding_side="left",
            max_pixels=1800 * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        assert self.tokenizer is not None, "tokenizer must be initialized"
        assert self.processor is not None, "processor must be initialized"
        inputs, _ = self.process_raw_data(raw)
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens: list[list[str]] = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]

        def split_number(n: int, m: float) -> list[int]:
            # split n into m parts, the parts are as even as possible
            assert m.is_integer()
            quotient = n // int(m)
            remainder = n % int(m)
            return [quotient + 1 if i < remainder else quotient for i in range(int(m))]

        batch_token_positions: list[list[Any]] = [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

        if "images" in raw and raw["images"] is not None:
            assert "image_grid_thw" in inputs
            assert "pixel_values" in inputs
            resized_shape_list = (inputs["image_grid_thw"][:, 1:] * 14).tolist()
            for str_tokens, images in zip(batch_str_tokens, raw["images"]):
                # str_tokens: list[str], tokens for each text in the batch
                # images: list[torch.Tensor], images for each text in the batch
                # resized_shape: [total_image_number, 2]
                # token_positions: list[Any], positions of the tokens in the input text
                start_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_start|>"]
                end_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_end|>"]
                assert len(start_id_list) == len(end_id_list)
                images_num = len(start_id_list)  # number of images in this text
                resized_shapes = resized_shape_list[:images_num]  # get the resized shapes for images in this text
                resized_shape_list = resized_shape_list[images_num:]

                for i, (start_id, end_id, image, resized_shape) in enumerate(
                    zip(start_id_list, end_id_list, images, resized_shapes)
                ):
                    resized_height, resized_width = int(resized_shape[0]), int(resized_shape[1])
                    original_height, original_width = image.shape[-2], image.shape[-1]
                    image_token_num = end_id - start_id - 1
                    assert image_token_num == resized_height * resized_width / 14 / 14 / 4

                    split_height = split_number(original_height, resized_height / 28)
                    split_width = split_number(original_width, resized_width / 28)

                    prefix_sum_height = [0] + list(accumulate(split_height))
                    prefix_sum_width = [0] + list(accumulate(split_width))

                    prefix_sum_height = [i / original_height for i in prefix_sum_height]
                    prefix_sum_width = [i / original_width for i in prefix_sum_width]
                    grid_coords = [
                        (
                            id // (resized_width // 28),
                            id % (resized_width // 28),
                        )
                        for id in range(image_token_num)
                    ]
                    original_coords = [
                        (
                            prefix_sum_width[grid_coords[id][1]],
                            prefix_sum_height[grid_coords[id][0]],
                            prefix_sum_width[grid_coords[id][1] + 1],
                            prefix_sum_height[grid_coords[id][0] + 1],
                        )
                        for id in range(image_token_num)
                    ]

                    batch_token_positions[i][start_id + 1 : end_id] = original_coords
                    for j in range(len(original_coords)):
                        batch_token_positions[i][start_id + 1 + j] = {
                            "key": "image",
                            "rect": original_coords[j],
                            "image_index": i,
                        }
        return batch_token_positions

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.process_raw_data(raw)[0].to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations

    def process_raw_data(
        self, raw: dict[str, Any], padding: str | bool = "max_length"
    ) -> tuple[BatchFeature, dict[str, Any]]:
        # raw is a dict with keys "text", "images", "videos", etc.
        IMAGE_PAD_TOKEN: str = raw.get("image_pad_token", "<image>")
        inputs = cast(BatchFeature, {})
        processed_raw = {}

        # use process_vision_info to resize images
        assert self.processor is not None, "processor must be initialized"
        if "images" in raw:
            processed_raw["images"] = raw["images"]

        processed_raw["text"] = [
            text.replace(IMAGE_PAD_TOKEN, f"<|vision_start|>{self.processor.image_token}<|vision_end|>")
            for text in raw["text"]
        ]
        inputs: BatchFeature = self.processor(
            text=processed_raw["text"],
            images=raw.get("images", None),
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=padding,
            truncation=True,
        )

        return inputs, processed_raw


class QwenLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
        # hidden_size is 3584 for Qwen2.5-7B
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            torch_dtype=cfg.dtype,
        ).to(self.device)  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            padding_side="left",
        )
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
        ).to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        inputs = self.tokenizer(
            raw["text"], return_tensors="pt", padding="max_length", max_length=self.cfg.max_length, truncation=True
        )
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        decoded_str = self.tokenizer.decode(input_ids)
        batch_str_tokens = list(decoded_str)
        return [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]
