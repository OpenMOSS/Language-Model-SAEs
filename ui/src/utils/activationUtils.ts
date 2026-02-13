/**
 * Activation data utility functions
 * Used for parsing node IDs and finding activation records in circuit JSON data
 */

export interface ParsedNodeId {
  rawLayer: number;
  layerForActivation: number;
  featureOrHead: number;
  ctxIdx: number;
}

export interface NodeActivationData {
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  nodeType?: string;
  clerp?: string;
}

/**
 * Parse node ID (format: rawLayer_featureOrHead_ctxIdx)
 */
export const parseNodeId = (nodeId: string): ParsedNodeId => {
  const parts = nodeId.split('_');
  const rawLayer = Number(parts[0]) || 0;
  const featureOrHead = Number(parts[1]) || 0;
  const ctxIdx = Number(parts[2]) || 0;
  const layerForActivation = Math.floor(rawLayer / 2);
  return { rawLayer, layerForActivation, featureOrHead, ctxIdx };
};

/**
 * Find nodes array in JSON data structure
 */
export const findNodesArray = (jsonData: any): any[] => {
  if (jsonData.nodes && Array.isArray(jsonData.nodes)) {
    return jsonData.nodes;
  }
  if (Array.isArray(jsonData)) {
    return jsonData;
  }
  
  const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
  for (const key of possibleArrayKeys) {
    if (Array.isArray(jsonData[key])) {
      return jsonData[key];
    }
  }
  
  // Try to find any array in values
  const values = Object.values(jsonData);
  const arrayValue = values.find(v => Array.isArray(v)) as any[] | undefined;
  return arrayValue || [];
};

/**
 * Find candidate activation records in JSON data
 */
export const findCandidateRecords = (jsonData: any): any[] => {
  const candidateRecords: any[] = [];
  
  const pushCandidateArrays = (obj: any) => {
    if (!obj) return;
    if (Array.isArray(obj)) {
      for (const item of obj) {
        if (item && typeof item === 'object') {
          const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
          const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
          const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
          if (hasActivationShape || hasZShape || hasIndexKey) {
            candidateRecords.push(item);
          }
        }
      }
    } else if (typeof obj === 'object') {
      for (const v of Object.values(obj)) {
        pushCandidateArrays(v);
      }
    }
  };
  
  pushCandidateArrays(jsonData);
  return candidateRecords;
};

/**
 * Match activation record based on parsed node ID and feature type
 */
export const matchActivationRecord = (
  rec: any,
  parsed: ParsedNodeId,
  featureType?: string
): boolean => {
  const recLayer = Number(rec?.layer);
  const recPos = Number(rec?.position);
  const recHead = rec?.head_idx;
  const recFeatIdx = rec?.feature_idx;

  const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;
  const posOk = !Number.isNaN(recPos) && recPos === parsed.ctxIdx;

  let indexOk = false;
  if (featureType) {
    const t = featureType.toLowerCase();
    if (t === 'lorsa') {
      indexOk = recHead === parsed.featureOrHead;
    } else if (t === 'cross layer transcoder') {
      indexOk = recFeatIdx === parsed.featureOrHead;
    } else {
      indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
    }
  } else {
    indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
  }

  return layerOk && posOk && indexOk;
};

/**
 * Match activation record ignoring position (for all-positions merge).
 * Same as matchActivationRecord but without posOk check.
 */
export const matchActivationRecordIgnorePosition = (
  rec: any,
  parsed: ParsedNodeId,
  featureType?: string
): boolean => {
  const recLayer = Number(rec?.layer);
  const recHead = rec?.head_idx;
  const recFeatIdx = rec?.feature_idx;

  const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;

  let indexOk = false;
  if (featureType) {
    const t = featureType.toLowerCase();
    if (t === 'lorsa') {
      indexOk = recHead === parsed.featureOrHead;
    } else if (t === 'cross layer transcoder') {
      indexOk = recFeatIdx === parsed.featureOrHead;
    } else {
      indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
    }
  } else {
    indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
  }

  return layerOk && indexOk;
};

/**
 * Get merged activation data for all positions from JSON (multi-file mode fallback).
 * Merges activations across all positions by taking max absolute value per cell.
 * z_pattern is not merged (semantically tied to a single query position).
 */
export const getAllPositionsActivationDataFromJson = (
  jsonData: any,
  nodeId: string
): NodeActivationData | null => {
  if (!nodeId || !jsonData) {
    return null;
  }

  const parsed = parseNodeId(nodeId);
  const nodesToSearch = findNodesArray(jsonData);
  const featureTypeForNode = nodesToSearch.find((n: any) => n?.node_id === nodeId)?.feature_type;
  const candidateRecords = findCandidateRecords(jsonData);

  const matchedRecords = candidateRecords.filter((rec: any) =>
    matchActivationRecordIgnorePosition(rec, parsed, featureTypeForNode)
  );

  if (matchedRecords.length === 0) {
    return null;
  }

  const mergedActivations = new Array(64).fill(0);
  for (const rec of matchedRecords) {
    if (rec.activations && Array.isArray(rec.activations) && rec.activations.length === 64) {
      for (let i = 0; i < 64; i++) {
        const currentValue = mergedActivations[i];
        const newValue = rec.activations[i];
        if (Math.abs(newValue) > Math.abs(currentValue)) {
          mergedActivations[i] = newValue;
        }
      }
    }
  }

  const nodeMeta = nodesToSearch.find((n: any) => n?.node_id === nodeId);
  return {
    activations: mergedActivations,
    zPatternIndices: undefined,
    zPatternValues: undefined,
    nodeType: featureTypeForNode,
    clerp: nodeMeta?.clerp,
  };
};

/**
 * Normalize z-pattern data
 * Keeps consistent with CustomFenInput processing logic
 */
export const normalizeZPattern = (
  zPatternIndicesRaw: unknown,
  zPatternValuesRaw: unknown
): { zPatternIndices?: number[][]; zPatternValues?: number[] } => {
  if (!Array.isArray(zPatternIndicesRaw) || !Array.isArray(zPatternValuesRaw)) {
    return {};
  }

  const zPatternValues = zPatternValuesRaw as number[];
  const zPatternIndices =
    zPatternIndicesRaw.length > 0 && Array.isArray(zPatternIndicesRaw[0])
      ? (zPatternIndicesRaw as number[][])
      : ([zPatternIndicesRaw as number[]] as number[][]);

  return { zPatternIndices, zPatternValues };
};
