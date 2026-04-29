from typing import List


__all__: List[str] = []


try:
    from .utils import (
        get_pos_from_square,
        has_piece_of_type,
        get_piece_type_pos,
        get_start_end_pos_from_move_uci,
        is_valid_move_uci,
    )

    __all__ += [
        "get_pos_from_square",
        "has_piece_of_type",
        "get_piece_type_pos",
        "get_start_end_pos_from_move_uci",
        "is_valid_move_uci",
    ]
except ModuleNotFoundError:
    pass


try:
    from .move import (
        get_move_from_model,
        get_wdl_from_model,
        get_m_from_model,
        get_value_from_model,
        get_move_from_policy_output,
        get_move_from_policy_output_with_prob,
        get_move_prob_and_rank,
        get_move_prob_and_rank_from_policy_output,
        get_value_from_output,
        get_top_3_moves_and_probs,
        get_move_from_policy_output_with_logit,
    )

    __all__ += [
        "get_move_from_model",
        "get_wdl_from_model",
        "get_m_from_model",
        "get_value_from_model",
        "get_move_from_policy_output",
        "get_move_from_policy_output_with_prob",
        "get_move_from_policy_output_with_logit",
        "get_move_prob_and_rank",
        "get_move_prob_and_rank_from_policy_output",
        "get_value_from_output",
        "get_top_3_moves_and_probs",
    ]
except ModuleNotFoundError:
    pass


try:
    from .feature import (
        get_feature_vector,
        get_feature_encoder_vector,
        FeatureSpec,
        FeatureSpecParams,
        FeatureType,
    )

    __all__ += [
        "get_feature_vector",
        "get_feature_encoder_vector",
        "FeatureSpec",
        "FeatureSpecParams",
        "FeatureType",
    ]
except ModuleNotFoundError:
    pass


try:
    from .activation_frequency import (
        feature_frequency_with_piece_type,
        feature_frequency_with_keypoint,
    )

    __all__ += [
        "feature_frequency_with_piece_type",
        "feature_frequency_with_keypoint",
    ]
except ModuleNotFoundError:
    pass


try:
    from .plot import (
        parse_fen_board,
        vals64_to_board,
        make_board_fig,
        plot_board_heatmap,
        N_POS,
        PIECE_SYMBOLS,
    )

    __all__ += [
        "parse_fen_board",
        "vals64_to_board",
        "make_board_fig",
        "plot_board_heatmap",
        "N_POS",
        "PIECE_SYMBOLS",
    ]
except ModuleNotFoundError:
    pass


from .onnx_to_torch_conversion import (
    BT4CleanLC0Model,
    T82CleanLC0Model,
    build_clean_model,
    conversion_support_status,
    convert_bt4_onnx_to_clean_state_dict,
    convert_onnx_to_clean_state_dict,
    convert_t82_onnx_to_clean_state_dict,
    get_conversion_spec,
    list_supported_models,
    load_clean_model,
    supports_onnx_conversion,
)

__all__ += [
    "BT4CleanLC0Model",
    "T82CleanLC0Model",
    "build_clean_model",
    "conversion_support_status",
    "convert_bt4_onnx_to_clean_state_dict",
    "convert_onnx_to_clean_state_dict",
    "convert_t82_onnx_to_clean_state_dict",
    "get_conversion_spec",
    "list_supported_models",
    "load_clean_model",
    "supports_onnx_conversion",
]
