from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException

try:
    from .circuit_interpretation import (
        create_circuit_annotation as create_circuit_annotation_service,
        get_circuits_by_feature as get_circuits_by_feature_service,
        get_circuit_annotation as get_circuit_annotation_service,
        list_circuit_annotations as list_circuit_annotations_service,
        update_circuit_interpretation as update_circuit_interpretation_service,
        add_feature_to_circuit as add_feature_to_circuit_service,
        remove_feature_from_circuit as remove_feature_from_circuit_service,
        update_feature_interpretation_in_circuit as update_feature_interpretation_in_circuit_service,
        delete_circuit_annotation as delete_circuit_annotation_service,
        add_edge_to_circuit as add_edge_to_circuit_service,
        remove_edge_from_circuit as remove_edge_from_circuit_service,
        update_edge_weight as update_edge_weight_service,
        set_feature_level as set_feature_level_service,
    )
except ImportError:
    from circuit_interpretation import (  # type: ignore[no-redef]
        create_circuit_annotation as create_circuit_annotation_service,
        get_circuits_by_feature as get_circuits_by_feature_service,
        get_circuit_annotation as get_circuit_annotation_service,
        list_circuit_annotations as list_circuit_annotations_service,
        update_circuit_interpretation as update_circuit_interpretation_service,
        add_feature_to_circuit as add_feature_to_circuit_service,
        remove_feature_from_circuit as remove_feature_from_circuit_service,
        update_feature_interpretation_in_circuit as update_feature_interpretation_in_circuit_service,
        delete_circuit_annotation as delete_circuit_annotation_service,
        add_edge_to_circuit as add_edge_to_circuit_service,
        remove_edge_from_circuit as remove_edge_from_circuit_service,
        update_edge_weight as update_edge_weight_service,
        set_feature_level as set_feature_level_service,
    )


def get_circuit_annotations_router(client: Any, sae_series: str) -> APIRouter:
    """
    Create an APIRouter with all circuit annotation endpoints bound to the given client and sae_series.
    """
    router = APIRouter()

    @router.post("/circuit_annotations")
    def create_circuit_annotation(request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new circuit annotation.

        Args:
            request: contains the following fields:
                - circuit_interpretation: circuit interpretation
                - sae_combo_id: SAE combo ID
                - features: feature list, each feature contains:
                    - sae_name: SAE name
                    - sae_series: SAE series
                    - layer: layer number (actual layer in the model)
                    - feature_index: feature index
                    - feature_type: feature type ("transcoder" or "lorsa")
                    - interpretation: interpretation of the feature (optional)
                    - level: optional circuit level (independent of layer, for visualization)
                    - feature_id: optional feature unique identifier
                - edges: optional edge list, each edge contains:
                    - source_feature_id: source feature ID
                    - target_feature_id: target feature ID
                    - weight: edge weight
                    - interpretation: optional edge interpretation
                - metadata: optional metadata dictionary

        Returns:
            Created circuit annotation information.
        """
        try:
            return create_circuit_annotation_service(
                client=client,
                sae_series=sae_series,
                circuit_interpretation=request.get("circuit_interpretation", ""),
                sae_combo_id=request.get("sae_combo_id"),
                features=request.get("features", []),
                edges=request.get("edges"),
                metadata=request.get("metadata"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"create circuit annotation failed: {str(e)}",
            )

    @router.get("/circuit_annotations/by_feature")
    def get_circuits_by_feature(
        sae_name: str,
        sae_series_param: Optional[str] = None,
        layer: int = 0,
        feature_index: int = 0,
        feature_type: Optional[str] = None,
    ):
        """
        Get all circuit annotations containing the specified feature.

        Args:
            sae_name: SAE name
            sae_series_param: SAE series (optional, default uses global sae_series)
            layer: layer number
            feature_index: feature index
            feature_type: optional feature type filter ("transcoder" or "lorsa")

        Returns:
            List of all circuit annotations containing the specified feature.
        """
        try:
            return get_circuits_by_feature_service(
                client=client,
                sae_series=sae_series,  # global default value
                sae_name=sae_name,
                layer=layer,
                feature_index=feature_index,
                sae_series_param=sae_series_param,  # route parameter (may be None, service function will use default value)
                feature_type=feature_type,
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"get circuits by feature failed: {str(e)}",
            )

    @router.get("/circuit_annotations/{circuit_id}")
    def get_circuit_annotation(circuit_id: str):
        """
        Get the specified circuit annotation.

        Args:
            circuit_id: unique ID of the circuit annotation

        Returns:
            Circuit annotation information.
        """
        try:
            return get_circuit_annotation_service(
                client=client,
                circuit_id=circuit_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"get circuit annotation failed: {str(e)}",
            )

    @router.get("/circuit_annotations")
    def list_circuit_annotations(
        sae_combo_id: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ):
        """
        List all circuit annotations.

        Args:
            sae_combo_id: optional SAE combo ID filter
            limit: maximum number of items to return
            skip: number of items to skip (for pagination)

        Returns:
            List of circuit annotations.
        """
        try:
            return list_circuit_annotations_service(
                client=client,
                sae_combo_id=sae_combo_id,
                limit=limit,
                skip=skip,
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"list circuit annotations failed: {str(e)}",
            )

    @router.put("/circuit_annotations/{circuit_id}/interpretation")
    def update_circuit_interpretation(circuit_id: str, request: Dict[str, Any]):
        """
        Update the overall circuit interpretation.

        Args:
            circuit_id: unique ID of the circuit annotation
            request: contains:
                - circuit_interpretation: circuit interpretation

        Returns:
            Success message.
        """
        try:
            return update_circuit_interpretation_service(
                client=client,
                circuit_id=circuit_id,
                circuit_interpretation=request.get("circuit_interpretation", ""),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"update circuit interpretation failed: {str(e)}",
            )

    @router.post("/circuit_annotations/{circuit_id}/features")
    def add_feature_to_circuit(circuit_id: str, request: Dict[str, Any]):
        """
        Add a feature to the circuit.

        Args:
            circuit_id: unique ID of the circuit annotation
            request: contains:
                - sae_name: SAE name
                - sae_series: SAE series (optional, default uses global sae_series)
                - layer: layer number
                - feature_index: feature index
                - feature_type: feature type ("transcoder" or "lorsa")
                - interpretation: interpretation of the feature (optional)

        Returns:
            Success message.
        """
        try:
            return add_feature_to_circuit_service(
                client=client,
                sae_series=sae_series,
                circuit_id=circuit_id,
                sae_name=request.get("sae_name"),
                layer=request.get("layer"),
                feature_index=request.get("feature_index"),
                feature_type=request.get("feature_type"),
                sae_series_param=request.get("sae_series"),
                interpretation=request.get("interpretation", ""),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"add feature to circuit failed: {str(e)}",
            )

    @router.delete("/circuit_annotations/{circuit_id}/features")
    def remove_feature_from_circuit(circuit_id: str, request: Dict[str, Any]):
        """
        Remove a feature from the circuit.

        Args:
            circuit_id: unique ID of the circuit annotation
            request: contains:
                - sae_name: SAE name
                - sae_series: SAE series (optional, default uses global sae_series)
                - layer: layer number
                - feature_index: feature index
                - feature_type: feature type ("transcoder" or "lorsa")

        Returns:
            Success message.
        """
        try:
            return remove_feature_from_circuit_service(
                client=client,
                sae_series=sae_series,
                circuit_id=circuit_id,
                sae_name=request.get("sae_name"),
                layer=request.get("layer"),
                feature_index=request.get("feature_index"),
                feature_type=request.get("feature_type"),
                sae_series_param=request.get("sae_series"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"remove feature from circuit failed: {str(e)}",
            )

    @router.put("/circuit_annotations/{circuit_id}/features/interpretation")
    def update_feature_interpretation_in_circuit(circuit_id: str, request: Dict[str, Any]):
        """
        Update the interpretation of a specific feature within a circuit.
        """
        try:
            return update_feature_interpretation_in_circuit_service(
                client=client,
                sae_series=sae_series,
                circuit_id=circuit_id,
                sae_name=request.get("sae_name"),
                layer=request.get("layer"),
                feature_index=request.get("feature_index"),
                feature_type=request.get("feature_type"),
                interpretation=request.get("interpretation", ""),
                sae_series_param=request.get("sae_series"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"update feature interpretation failed: {str(e)}",
            )

    @router.delete("/circuit_annotations/{circuit_id}")
    def delete_circuit_annotation(circuit_id: str):
        """
        Delete a circuit annotation by ID.
        """
        try:
            return delete_circuit_annotation_service(
                client=client,
                circuit_id=circuit_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"delete circuit annotation failed: {str(e)}",
            )

    @router.post("/circuit_annotations/{circuit_id}/edges")
    def add_edge_to_circuit(circuit_id: str, request: Dict[str, Any]):
        """
        Add an edge between two features within a circuit.
        """
        try:
            return add_edge_to_circuit_service(
                client=client,
                circuit_id=circuit_id,
                source_feature_id=request.get("source_feature_id"),
                target_feature_id=request.get("target_feature_id"),
                weight=request.get("weight", 0.0),
                interpretation=request.get("interpretation"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"add edge to circuit failed: {str(e)}",
            )

    @router.delete("/circuit_annotations/{circuit_id}/edges")
    def remove_edge_from_circuit(circuit_id: str, request: Dict[str, Any]):
        """
        Remove an edge between two features within a circuit.
        """
        try:
            return remove_edge_from_circuit_service(
                client=client,
                circuit_id=circuit_id,
                source_feature_id=request.get("source_feature_id"),
                target_feature_id=request.get("target_feature_id"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"remove edge from circuit failed: {str(e)}",
            )

    @router.put("/circuit_annotations/{circuit_id}/edges")
    def update_edge_weight(circuit_id: str, request: Dict[str, Any]):
        """
        Update the weight (and optional interpretation) of an edge in a circuit.
        """
        try:
            return update_edge_weight_service(
                client=client,
                circuit_id=circuit_id,
                source_feature_id=request.get("source_feature_id"),
                target_feature_id=request.get("target_feature_id"),
                weight=request.get("weight"),
                interpretation=request.get("interpretation"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"update edge weight failed: {str(e)}",
            )

    @router.put("/circuit_annotations/{circuit_id}/features/{feature_id}/level")
    def set_feature_level(circuit_id: str, feature_id: str, request: Dict[str, Any]):
        """
        Set the visualization level for a feature within a circuit.
        """
        try:
            return set_feature_level_service(
                client=client,
                circuit_id=circuit_id,
                feature_id=feature_id,
                level=request.get("level"),
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"set feature level failed: {str(e)}",
            )

    return router

