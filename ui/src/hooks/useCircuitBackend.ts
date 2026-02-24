/**
 * Hook for circuit-related backend API calls.
 * Centralizes fetch logic for analyze_fen_all_positions and related endpoints.
 */

import { useCallback } from "react";
import { NodeActivationData, normalizeZPattern } from "@/utils/activationUtils";

const getBackendUrl = (): string => import.meta.env.VITE_BACKEND_URL;

export interface UseCircuitBackendOptions {
  /** Called when loading state changes (e.g. setLoadingAllPositions) */
  setLoadingAllPositions?: (loading: boolean) => void;
  /** Graph data to resolve node metadata (nodeType, clerp) for the result. May be null when no graph loaded. */
  linkGraphData?: { nodes?: Array<{ nodeId: string; feature_type?: string; clerp?: string }> } | null;
}

/**
 * Fetch activation data for all positions from backend.
 * Merges activations across positions by taking max absolute value per cell.
 * z_pattern is not included (semantically tied to a single query position).
 */
export const fetchAllPositionsFromBackend = async (
  dictionary: string,
  featureIndex: number,
  fen: string,
  nodeMetadata?: { nodeType?: string; clerp?: string }
): Promise<NodeActivationData | null> => {
  try {
    const response = await fetch(
      `${getBackendUrl()}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ fen: fen.trim() }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();

    const mergedActivations = new Array(64).fill(0);
    if (data.positions && Array.isArray(data.positions)) {
      for (const posData of data.positions) {
        if (posData.activations && Array.isArray(posData.activations) && posData.activations.length === 64) {
          for (let i = 0; i < 64; i++) {
            const newValue = posData.activations[i];
            if (Math.abs(newValue) > Math.abs(mergedActivations[i])) {
              mergedActivations[i] = newValue;
            }
          }
        }
      }
    }

    return {
      activations: mergedActivations,
      zPatternIndices: undefined,
      zPatternValues: undefined,
      nodeType: nodeMetadata?.nodeType,
      clerp: nodeMetadata?.clerp,
    };
  } catch (error) {
    console.error("fetchAllPositionsFromBackend failed:", error);
    return null;
  }
};

export interface ZPatternResult {
  zPatternIndices?: number[][];
  zPatternValues?: number[];
}

/**
 * Fetch z_pattern for a Lorsa feature at a specific query position (single-position display).
 * Calls analyze_fen_all_positions and extracts z_pattern for the given queryPos.
 */
export const fetchZPatternForPosFromBackend = async (
  dictionary: string,
  featureIndex: number,
  fen: string,
  queryPos: number,
  signal?: AbortSignal
): Promise<ZPatternResult | null> => {
  try {
    const response = await fetch(
      `${getBackendUrl()}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ fen: fen.trim() }),
        signal,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const positions = data?.positions;
    if (!Array.isArray(positions)) return null;

    const posData = positions.find((p: { position?: number }) => Number(p?.position) === queryPos);
    if (!posData) return null;

    const { zPatternIndices, zPatternValues } = normalizeZPattern(
      (posData as { z_pattern_indices?: unknown }).z_pattern_indices,
      (posData as { z_pattern_values?: unknown }).z_pattern_values
    );

    return { zPatternIndices, zPatternValues };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    console.error("fetchZPatternForPosFromBackend failed:", error);
    return null;
  }
};

/**
 * Hook that provides fetchAllPositionsFromBackend with loading state and node metadata.
 */
export const useCircuitBackend = (options: UseCircuitBackendOptions = {}) => {
  const { setLoadingAllPositions, linkGraphData } = options;

  const fetchAllPositions = useCallback(
    async (
      nodeId: string,
      fen: string,
      dictionary: string,
      featureIndex: number
    ): Promise<NodeActivationData | null> => {
      setLoadingAllPositions?.(true);
      try {
        const currentNode = linkGraphData?.nodes?.find((n: any) => n.nodeId === nodeId);
        const nodeMetadata = currentNode
          ? { nodeType: currentNode.feature_type, clerp: (currentNode as any)?.clerp }
          : undefined;

        return await fetchAllPositionsFromBackend(dictionary, featureIndex, fen, nodeMetadata);
      } finally {
        setLoadingAllPositions?.(false);
      }
    },
    [setLoadingAllPositions, linkGraphData]
  );

  return {
    fetchAllPositionsFromBackend: fetchAllPositions,
    fetchZPatternForPosFromBackend,
  };
};
