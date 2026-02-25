"""
Circuit Interpretation Service Module.

This module provides all business logic functions related to circuit annotations.
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

    if not sae_combo_id:
        raise HTTPException(status_code=400, detail="sae_combo_id is required")
    
    if not isinstance(features, list) or len(features) == 0:
        raise HTTPException(status_code=400, detail="features must be a non-empty list")
    
    try:
        from .constants import get_bt4_sae_combo
    except ImportError:
        from constants import get_bt4_sae_combo
    
    combo_cfg = get_bt4_sae_combo(sae_combo_id)
    
    corrected_features = []
    for feature in features:
        corrected_feature = feature.copy()
        layer = feature.get("layer")
        feature_type = feature.get("feature_type", "").lower()
        
        if "lorsa" in feature_type:
            template = combo_cfg.get("lorsa_sae_name_template", "BT4_lorsa_L{layer}A")
            corrected_feature["sae_name"] = template.format(layer=layer)
        elif "transcoder" in feature_type or "cross layer transcoder" in feature_type:
            template = combo_cfg.get("tc_sae_name_template", "BT4_tc_L{layer}M")
            corrected_feature["sae_name"] = template.format(layer=layer)
        
        original_sae_name = feature.get("sae_name", "")
        if original_sae_name and original_sae_name != corrected_feature["sae_name"]:
            print(f"[WARNING] SAE name corrected for feature: {original_sae_name} -> {corrected_feature['sae_name']} (combo_id={sae_combo_id}, layer={layer}, type={feature_type})")
        
        corrected_features.append(corrected_feature)
    
    circuit_id = str(uuid.uuid4())
    
    print(f"[DEBUG] create_circuit_annotation: circuit_id={circuit_id}, sae_combo_id={sae_combo_id}")
    print(f"[DEBUG] create_circuit_annotation: corrected_features={corrected_features}")
    if edges:
        print(f"[DEBUG] create_circuit_annotation: edges={edges}")
    
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
    
    circuit = client.get_circuit_annotation(circuit_id)
    if circuit is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve created circuit annotation")
    
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

    sae_series_actual = sae_series_param if sae_series_param is not None else sae_series
    
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
    if not all([layer is not None, feature_index is not None, feature_type]):
        raise HTTPException(
            status_code=400,
            detail="layer, feature_index, and feature_type are required"
        )
    
    circuit = client.get_circuit_annotation(circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit annotation {circuit_id} not found")
    
    try:
        from .constants import get_bt4_sae_combo
    except ImportError:
        from constants import get_bt4_sae_combo
    
    combo_cfg = get_bt4_sae_combo(circuit.sae_combo_id)
    feature_type_lower = feature_type.lower()
    
    if "lorsa" in feature_type_lower:
        template = combo_cfg.get("lorsa_sae_name_template", "BT4_lorsa_L{layer}A")
        corrected_sae_name = template.format(layer=layer)
    elif "transcoder" in feature_type_lower or "cross layer transcoder" in feature_type_lower:
        template = combo_cfg.get("tc_sae_name_template", "BT4_tc_L{layer}M")
        corrected_sae_name = template.format(layer=layer)
    else:
        corrected_sae_name = sae_name
    
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

