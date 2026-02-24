import { Feature } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";

export const fetchFeature = async (
  analysisName: string,
  layer: number,
  featureId: number
): Promise<Feature | null> => {
  try {
    // Replace {} with layer in the analysis name
    const formattedAnalysisName = analysisName.replace("{}", layer.toString());
        
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${formattedAnalysisName}/features/${featureId}`,
      {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
      }
    );

    if (!response.ok) {
      console.warn(`Failed to fetch feature from ${formattedAnalysisName}: ${response.status} ${response.statusText}`);
      return null;
    }

    const arrayBuffer = await response.arrayBuffer();
    const decoded = decode(new Uint8Array(arrayBuffer)) as Record<string, unknown>;
    const camelCased = camelcaseKeys(decoded, {
      deep: true,
      stopPaths: ["context"],
    });

    return camelCased as Feature;
  } catch (error) {
    console.error(`Error fetching feature ${featureId} from layer ${layer}:`, error);
    return null;
  }
};

export const getDictionaryName = (metadata: any, layer: number, isLorsa: boolean): string => {
  if (isLorsa) {
    // Support the lorsa_analysis_name field
    const analysisName = metadata.lorsa_analysis_name;
    if (analysisName && analysisName.includes('BT4')) {
      // BT4 format: BT4_lorsa_L{layer}A
      return `BT4_lorsa_L${layer}A`;
    } else {
      return analysisName ? analysisName.replace("{}", layer.toString()) : `lc0-lorsa-L${layer}`;
    }
  } else {
    // Support the newer field name tc_analysis_name and keep clt_analysis_name for backward compatibility
    const analysisName = metadata.tc_analysis_name || metadata.clt_analysis_name;
    if (analysisName && analysisName.includes('BT4')) {
      // BT4 format: BT4_tc_L{layer}M
      return `BT4_tc_L${layer}M`;
    } else {
      return analysisName ? analysisName.replace("{}", layer.toString()) : `lc0_L${layer}M_16x_k30_lr2e-03_auxk_sparseadam`;
    }
  }
};

// Circuit Annotation API
export interface CircuitFeature {
  sae_name: string;
  sae_series: string;
  layer: number;
  feature_index: number;
  feature_type: "transcoder" | "lorsa";
  interpretation?: string;
  level?: number; // Circuit level (independent of layer, for visualization)
  feature_id?: string; // Unique identifier for this feature in the circuit
}

export interface CircuitEdge {
  source_feature_id: string;
  target_feature_id: string;
  weight: number;
  interpretation?: string;
}

export interface CircuitAnnotation {
  circuit_id: string;
  circuit_interpretation: string;
  sae_combo_id: string;
  features: CircuitFeature[];
  edges?: CircuitEdge[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

const API_BASE = import.meta.env.VITE_BACKEND_URL || "";

export const createCircuitAnnotation = async (
  circuitInterpretation: string,
  saeComboId: string,
  features: CircuitFeature[],
  edges?: CircuitEdge[],
  metadata?: Record<string, any>
): Promise<CircuitAnnotation> => {
  const response = await fetch(`${API_BASE}/circuit_annotations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      circuit_interpretation: circuitInterpretation,
      sae_combo_id: saeComboId,
      features,
      edges,
      metadata,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to create circuit annotation: ${response.status}`);
  }

  return response.json();
};

export const getCircuitAnnotation = async (circuitId: string): Promise<CircuitAnnotation> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to get circuit annotation: ${response.status}`);
  }

  return response.json();
};

export const listCircuitAnnotations = async (
  saeComboId?: string,
  limit: number = 100,
  skip: number = 0
): Promise<{ circuits: CircuitAnnotation[]; total_count: number }> => {
  const params = new URLSearchParams();
  if (saeComboId) params.append("sae_combo_id", saeComboId);
  params.append("limit", limit.toString());
  params.append("skip", skip.toString());

  const response = await fetch(`${API_BASE}/circuit_annotations?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to list circuit annotations: ${response.status}`);
  }

  return response.json();
};

export const updateCircuitInterpretation = async (
  circuitId: string,
  circuitInterpretation: string
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/interpretation`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      circuit_interpretation: circuitInterpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update circuit interpretation: ${response.status}`);
  }
};

export const addFeatureToCircuit = async (
  circuitId: string,
  feature: CircuitFeature
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(feature),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to add feature to circuit: ${response.status}`);
  }
};

export const removeFeatureFromCircuit = async (
  circuitId: string,
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType: "transcoder" | "lorsa"
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sae_name: saeName,
      sae_series: saeSeries,
      layer,
      feature_index: featureIndex,
      feature_type: featureType,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to remove feature from circuit: ${response.status}`);
  }
};

export const updateFeatureInterpretationInCircuit = async (
  circuitId: string,
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType: "transcoder" | "lorsa",
  interpretation: string
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features/interpretation`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sae_name: saeName,
      sae_series: saeSeries,
      layer,
      feature_index: featureIndex,
      feature_type: featureType,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update feature interpretation: ${response.status}`);
  }
};

export const getCircuitsByFeature = async (
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType?: "transcoder" | "lorsa"
): Promise<{ circuits: CircuitAnnotation[] }> => {
  const params = new URLSearchParams();
  params.append("sae_name", saeName);
  params.append("sae_series", saeSeries);
  params.append("layer", layer.toString());
  params.append("feature_index", featureIndex.toString());
  if (featureType) params.append("feature_type", featureType);

  const response = await fetch(`${API_BASE}/circuit_annotations/by_feature?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to get circuits by feature: ${response.status}`);
  }

  return response.json();
};

export const deleteCircuitAnnotation = async (circuitId: string): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to delete circuit annotation: ${response.status}`);
  }
};

// Edge management APIs
export const addEdgeToCircuit = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string,
  weight: number = 0.0,
  interpretation?: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
      weight,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to add edge: ${response.status}`);
  }

  return response.json();
};

export const removeEdgeFromCircuit = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to remove edge: ${response.status}`);
  }

  return response.json();
};

export const updateEdgeWeight = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string,
  weight: number,
  interpretation?: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
      weight,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update edge weight: ${response.status}`);
  }

  return response.json();
};

export const setFeatureLevel = async (
  circuitId: string,
  featureId: string,
  level: number
): Promise<{ message: string }> => {
  const response = await fetch(
    `${API_BASE}/circuit_annotations/${circuitId}/features/${featureId}/level`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        level,
      }),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to set feature level: ${response.status}`);
  }

  return response.json();
}; 