"""
Circuit Interpretation Service Module

该模块提供所有与 circuit annotation 相关的业务逻辑函数。
"""
import uuid
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

from lm_saes.database import MongoClient


def create_circuit_annotation(
    client: MongoClient,
    sae_series: str,
    circuit_interpretation: str,
    sae_combo_id: str,
    features: List[Dict[str, Any]],
    edges: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建新的circuit标注
    
    Args:
        client: MongoDB客户端实例
        sae_series: SAE系列名称
        circuit_interpretation: 回路的整体解释
        sae_combo_id: SAE组合ID
        features: 特征列表，每个特征包含：
            - sae_name: SAE名称
            - sae_series: SAE系列
            - layer: 层号（模型中的实际层）
            - feature_index: 特征索引
            - feature_type: 特征类型 ("transcoder" 或 "lorsa")
            - interpretation: 该特征的解释（可选）
            - level: 可选的circuit层级（独立于layer，用于可视化）
            - feature_id: 可选的feature唯一标识符
        edges: 可选的边列表，每条边包含：
            - source_feature_id: 源feature的ID
            - target_feature_id: 目标feature的ID
            - weight: 边的权重
            - interpretation: 可选的边解释
        metadata: 可选的元数据字典
    
    Returns:
        创建的circuit标注信息（字典格式）
    
    Raises:
        HTTPException: 当参数无效或创建失败时
    """
    if not sae_combo_id:
        raise HTTPException(status_code=400, detail="sae_combo_id is required")
    
    if not isinstance(features, list) or len(features) == 0:
        raise HTTPException(status_code=400, detail="features must be a non-empty list")
    
    # 根据 sae_combo_id 自动修正 features 中的 sae_name
    try:
        from .constants import get_bt4_sae_combo
    except ImportError:
        from constants import get_bt4_sae_combo
    
    combo_cfg = get_bt4_sae_combo(sae_combo_id)
    
    # 修正每个 feature 的 sae_name
    corrected_features = []
    for feature in features:
        corrected_feature = feature.copy()
        layer = feature.get("layer")
        feature_type = feature.get("feature_type", "").lower()
        
        # 根据 feature_type 和 layer 生成正确的 sae_name
        if "lorsa" in feature_type:
            template = combo_cfg.get("lorsa_sae_name_template", "BT4_lorsa_L{layer}A")
            corrected_feature["sae_name"] = template.format(layer=layer)
        elif "transcoder" in feature_type or "cross layer transcoder" in feature_type:
            template = combo_cfg.get("tc_sae_name_template", "BT4_tc_L{layer}M")
            corrected_feature["sae_name"] = template.format(layer=layer)
        
        # 如果前端传递的 sae_name 与生成的 sae_name 不一致，记录警告
        original_sae_name = feature.get("sae_name", "")
        if original_sae_name and original_sae_name != corrected_feature["sae_name"]:
            print(f"[WARNING] SAE name corrected for feature: {original_sae_name} -> {corrected_feature['sae_name']} (combo_id={sae_combo_id}, layer={layer}, type={feature_type})")
        
        corrected_features.append(corrected_feature)
    
    # 生成唯一的circuit_id
    circuit_id = str(uuid.uuid4())
    
    # 添加调试日志
    print(f"[DEBUG] create_circuit_annotation: circuit_id={circuit_id}, sae_combo_id={sae_combo_id}")
    print(f"[DEBUG] create_circuit_annotation: corrected_features={corrected_features}")
    if edges:
        print(f"[DEBUG] create_circuit_annotation: edges={edges}")
    
    # 创建circuit标注（使用修正后的 features）
    success = client.create_circuit_annotation(
        circuit_id=circuit_id,
        circuit_interpretation=circuit_interpretation,
        sae_combo_id=sae_combo_id,
        features=corrected_features,
        edges=edges,
        metadata=metadata,
    )
    
    print(f"[DEBUG] create_circuit_annotation: success={success}")
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create circuit annotation")
    
    # 获取创建的circuit标注
    circuit = client.get_circuit_annotation(circuit_id)
    if circuit is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve created circuit annotation")
    
    # 转换为字典格式
    circuit_dict = circuit.model_dump()
    circuit_dict["created_at"] = circuit.created_at.isoformat()
    circuit_dict["updated_at"] = circuit.updated_at.isoformat()
    
    return circuit_dict


def get_circuits_by_feature(
    client: MongoClient,
    sae_series: str,
    sae_name: str,
    layer: int,
    feature_index: int,
    sae_series_param: Optional[str] = None,
    feature_type: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    获取包含指定特征的所有circuit标注
    
    Args:
        client: MongoDB客户端实例
        sae_series: 默认的SAE系列名称
        sae_name: SAE名称
        layer: 层号
        feature_index: 特征索引
        sae_series_param: SAE系列（可选，默认使用sae_series参数）
        feature_type: 可选的特征类型过滤器 ("transcoder" 或 "lorsa")
    
    Returns:
        包含该特征的所有circuit标注列表（字典格式）
    
    Raises:
        HTTPException: 当获取失败时
    """
    sae_series_actual = sae_series_param if sae_series_param is not None else sae_series
    
    # 添加调试日志
    print(f"[DEBUG] get_circuits_by_feature: sae_name={sae_name}, sae_series={sae_series_actual}, layer={layer}, feature_index={feature_index}, feature_type={feature_type}")
    
    circuits = client.get_circuits_by_feature(
        sae_name=sae_name,
        sae_series=sae_series_actual,
        layer=layer,
        feature_index=feature_index,
        feature_type=feature_type,
    )
    
    print(f"[DEBUG] get_circuits_by_feature: found {len(circuits)} circuits")
    
    circuit_list = []
    for circuit in circuits:
        circuit_dict = circuit.model_dump()
        circuit_dict["created_at"] = circuit.created_at.isoformat()
        circuit_dict["updated_at"] = circuit.updated_at.isoformat()
        circuit_list.append(circuit_dict)
    
    return {"circuits": circuit_list}


def get_circuit_annotation(
    client: MongoClient,
    circuit_id: str,
) -> Dict[str, Any]:
    """
    获取指定的circuit标注
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
    
    Returns:
        Circuit标注信息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或获取失败时
    """
    circuit = client.get_circuit_annotation(circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
    
    circuit_dict = circuit.model_dump()
    circuit_dict["created_at"] = circuit.created_at.isoformat()
    circuit_dict["updated_at"] = circuit.updated_at.isoformat()
    
    return circuit_dict


def list_circuit_annotations(
    client: MongoClient,
    sae_combo_id: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
) -> Dict[str, Any]:
    """
    列出所有circuit标注
    
    Args:
        client: MongoDB客户端实例
        sae_combo_id: 可选的SAE组合ID过滤器
        limit: 返回的最大数量
        skip: 跳过的数量（用于分页）
    
    Returns:
        Circuit标注列表和总数（字典格式）
    
    Raises:
        HTTPException: 当获取失败时
    """
    circuits = client.list_circuit_annotations(
        sae_combo_id=sae_combo_id,
        limit=limit,
        skip=skip,
    )
    
    circuit_list = []
    for circuit in circuits:
        circuit_dict = circuit.model_dump()
        circuit_dict["created_at"] = circuit.created_at.isoformat()
        circuit_dict["updated_at"] = circuit.updated_at.isoformat()
        circuit_list.append(circuit_dict)
    
    return {
        "circuits": circuit_list,
        "total_count": client.get_circuit_annotation_count(sae_combo_id=sae_combo_id),
    }


def update_circuit_interpretation(
    client: MongoClient,
    circuit_id: str,
    circuit_interpretation: str,
) -> Dict[str, str]:
    """
    更新circuit的整体解释
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
        circuit_interpretation: 新的解释文本
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或更新失败时
    """
    success = client.update_circuit_interpretation(
        circuit_id=circuit_id,
        circuit_interpretation=circuit_interpretation,
    )
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
    
    return {"message": "Circuit interpretation updated successfully"}


def add_feature_to_circuit(
    client: MongoClient,
    sae_series: str,
    circuit_id: str,
    sae_name: str,
    layer: int,
    feature_index: int,
    feature_type: str,
    sae_series_param: Optional[str] = None,
    interpretation: str = "",
) -> Dict[str, str]:
    """
    向circuit添加一个特征
    
    Args:
        client: MongoDB客户端实例
        sae_series: 默认的SAE系列名称
        circuit_id: Circuit标注的唯一ID
        sae_name: SAE名称（如果提供，会根据circuit的sae_combo_id自动修正）
        layer: 层号
        feature_index: 特征索引
        feature_type: 特征类型 ("transcoder" 或 "lorsa")
        sae_series_param: SAE系列（可选，默认使用sae_series参数）
        interpretation: 该特征的解释（可选）
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当参数无效、circuit不存在或添加失败时
    """
    if not all([layer is not None, feature_index is not None, feature_type]):
        raise HTTPException(
            status_code=400,
            detail="layer, feature_index, and feature_type are required"
        )
    
    # 获取circuit以获取sae_combo_id
    circuit = client.get_circuit_annotation(circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
    
    # 根据circuit的sae_combo_id自动生成正确的sae_name
    try:
        from .constants import get_bt4_sae_combo
    except ImportError:
        from constants import get_bt4_sae_combo
    
    combo_cfg = get_bt4_sae_combo(circuit.sae_combo_id)
    feature_type_lower = feature_type.lower()
    
    # 根据 feature_type 和 layer 生成正确的 sae_name
    if "lorsa" in feature_type_lower:
        template = combo_cfg.get("lorsa_sae_name_template", "BT4_lorsa_L{layer}A")
        corrected_sae_name = template.format(layer=layer)
    elif "transcoder" in feature_type_lower or "cross layer transcoder" in feature_type_lower:
        template = combo_cfg.get("tc_sae_name_template", "BT4_tc_L{layer}M")
        corrected_sae_name = template.format(layer=layer)
    else:
        corrected_sae_name = sae_name  # 如果类型未知，使用原始值
    
    # 如果前端传递的 sae_name 与生成的 sae_name 不一致，记录警告
    if sae_name and sae_name != corrected_sae_name:
        print(f"[WARNING] SAE name corrected when adding feature: {sae_name} -> {corrected_sae_name} (combo_id={circuit.sae_combo_id}, layer={layer}, type={feature_type})")
    
    sae_series_actual = sae_series_param if sae_series_param is not None else sae_series
    
    success = client.add_feature_to_circuit(
        circuit_id=circuit_id,
        sae_name=corrected_sae_name,
        sae_series=sae_series_actual,
        layer=layer,
        feature_index=feature_index,
        feature_type=feature_type,
        interpretation=interpretation,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or feature already exists"
        )
    
    return {"message": "Feature added to circuit successfully"}


def remove_feature_from_circuit(
    client: MongoClient,
    sae_series: str,
    circuit_id: str,
    sae_name: str,
    layer: int,
    feature_index: int,
    feature_type: str,
    sae_series_param: Optional[str] = None,
) -> Dict[str, str]:
    """
    从circuit中删除一个特征
    
    Args:
        client: MongoDB客户端实例
        sae_series: 默认的SAE系列名称
        circuit_id: Circuit标注的唯一ID
        sae_name: SAE名称
        layer: 层号
        feature_index: 特征索引
        feature_type: 特征类型 ("transcoder" 或 "lorsa")
        sae_series_param: SAE系列（可选，默认使用sae_series参数）
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当参数无效、circuit不存在或删除失败时
    """
    if not all([sae_name, layer is not None, feature_index is not None, feature_type]):
        raise HTTPException(
            status_code=400,
            detail="sae_name, layer, feature_index, and feature_type are required"
        )
    
    sae_series_actual = sae_series_param if sae_series_param is not None else sae_series
    
    success = client.remove_feature_from_circuit(
        circuit_id=circuit_id,
        sae_name=sae_name,
        sae_series=sae_series_actual,
        layer=layer,
        feature_index=feature_index,
        feature_type=feature_type,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or feature not in circuit"
        )
    
    return {"message": "Feature removed from circuit successfully"}


def update_feature_interpretation_in_circuit(
    client: MongoClient,
    sae_series: str,
    circuit_id: str,
    sae_name: str,
    layer: int,
    feature_index: int,
    feature_type: str,
    interpretation: str,
    sae_series_param: Optional[str] = None,
) -> Dict[str, str]:
    """
    更新circuit中某个特征的解释
    
    Args:
        client: MongoDB客户端实例
        sae_series: 默认的SAE系列名称
        circuit_id: Circuit标注的唯一ID
        sae_name: SAE名称
        layer: 层号
        feature_index: 特征索引
        feature_type: 特征类型 ("transcoder" 或 "lorsa")
        interpretation: 新的解释文本
        sae_series_param: SAE系列（可选，默认使用sae_series参数）
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当参数无效、circuit不存在或更新失败时
    """
    if not all([sae_name, layer is not None, feature_index is not None, feature_type]):
        raise HTTPException(
            status_code=400,
            detail="sae_name, layer, feature_index, and feature_type are required"
        )
    
    sae_series_actual = sae_series_param if sae_series_param is not None else sae_series
    
    success = client.update_feature_interpretation_in_circuit(
        circuit_id=circuit_id,
        sae_name=sae_name,
        sae_series=sae_series_actual,
        layer=layer,
        feature_index=feature_index,
        feature_type=feature_type,
        interpretation=interpretation,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or feature not in circuit"
        )
    
    return {"message": "Feature interpretation updated successfully"}


def delete_circuit_annotation(
    client: MongoClient,
    circuit_id: str,
) -> Dict[str, str]:
    """
    删除circuit标注
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或删除失败时
    """
    success = client.delete_circuit_annotation(circuit_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
    
    return {"message": "Circuit annotation deleted successfully"}


def add_edge_to_circuit(
    client: MongoClient,
    circuit_id: str,
    source_feature_id: str,
    target_feature_id: str,
    weight: float = 0.0,
    interpretation: Optional[str] = None,
) -> Dict[str, str]:
    """
    向circuit添加一条边
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
        source_feature_id: 源feature的ID
        target_feature_id: 目标feature的ID
        weight: 边的权重
        interpretation: 可选的边解释
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当参数无效、circuit不存在或添加失败时
    """
    try:
        success = client.add_edge_to_circuit(
            circuit_id=circuit_id,
            source_feature_id=source_feature_id,
            target_feature_id=target_feature_id,
            weight=weight,
            interpretation=interpretation,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or edge already exists"
        )
    
    return {"message": "Edge added to circuit successfully"}


def remove_edge_from_circuit(
    client: MongoClient,
    circuit_id: str,
    source_feature_id: str,
    target_feature_id: str,
) -> Dict[str, str]:
    """
    从circuit删除一条边
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
        source_feature_id: 源feature的ID
        target_feature_id: 目标feature的ID
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或删除失败时
    """
    success = client.remove_edge_from_circuit(
        circuit_id=circuit_id,
        source_feature_id=source_feature_id,
        target_feature_id=target_feature_id,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or edge not in circuit"
        )
    
    return {"message": "Edge removed from circuit successfully"}


def update_edge_weight(
    client: MongoClient,
    circuit_id: str,
    source_feature_id: str,
    target_feature_id: str,
    weight: float,
    interpretation: Optional[str] = None,
) -> Dict[str, str]:
    """
    更新circuit中边的权重
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
        source_feature_id: 源feature的ID
        target_feature_id: 目标feature的ID
        weight: 新的权重
        interpretation: 可选的新边解释
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或更新失败时
    """
    success = client.update_edge_weight(
        circuit_id=circuit_id,
        source_feature_id=source_feature_id,
        target_feature_id=target_feature_id,
        weight=weight,
        interpretation=interpretation,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or edge not in circuit"
        )
    
    return {"message": "Edge weight updated successfully"}


def set_feature_level(
    client: MongoClient,
    circuit_id: str,
    feature_id: str,
    level: int,
) -> Dict[str, str]:
    """
    设置circuit中feature的层级
    
    Args:
        client: MongoDB客户端实例
        circuit_id: Circuit标注的唯一ID
        feature_id: Feature的ID
        level: Circuit层级（独立于layer，用于可视化）
    
    Returns:
        成功消息（字典格式）
    
    Raises:
        HTTPException: 当circuit不存在或更新失败时
    """
    success = client.set_feature_level(
        circuit_id=circuit_id,
        feature_id=feature_id,
        level=level,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit annotation {circuit_id} not found or feature not in circuit"
        )
    
    return {"message": "Feature level updated successfully"}

