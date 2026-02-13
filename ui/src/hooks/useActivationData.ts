/**
 * Custom hook for managing node activation data
 * Handles fetching and caching activation data from JSON or backend
 */

import { useCallback, useMemo } from "react";
import {
  parseNodeId,
  findNodesArray,
  findCandidateRecords,
  matchActivationRecord,
  normalizeZPattern,
  NodeActivationData,
} from "@/utils/activationUtils";

interface UseActivationDataOptions {
  originalCircuitJson: any;
  updateCounter: number;
}

export const useActivationData = ({
  originalCircuitJson,
  updateCounter,
}: UseActivationDataOptions) => {
  /**
   * Get node activation data from original circuit JSON
   */
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    if (!nodeId || !originalCircuitJson) {
      console.log('❌ Missing required parameters:', { 
        nodeId, 
        hasOriginalCircuitJson: !!originalCircuitJson 
      });
      return { 
        activations: undefined, 
        zPatternIndices: undefined, 
        zPatternValues: undefined 
      };
    }
    
    console.log(`🔍 Finding activation data for node ${nodeId}...`);
    
    const parsed = parseNodeId(nodeId);

    // 1) Priority: direct match in nodes array (if node object has inline activations/zPattern* fields)
    const nodesToSearch = findNodesArray(originalCircuitJson);

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find(node => node?.node_id === nodeId);
      if (exactMatch) {
        const inlineActs = exactMatch.activations;
        const { zPatternIndices: inlineZIdx, zPatternValues: inlineZVal } = normalizeZPattern(
          exactMatch.zPatternIndices,
          exactMatch.zPatternValues
        );
        console.log('✅ Node inline fields check:', {
          hasInlineActivations: !!inlineActs,
          hasInlineZIdx: !!inlineZIdx,
          hasInlineZVal: !!inlineZVal,
        });
        if (inlineActs || (inlineZIdx && inlineZVal)) {
          return {
            activations: inlineActs,
            zPatternIndices: inlineZIdx,
            zPatternValues: inlineZVal,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // 2) Match based on node_id parsing in activation record collection (layer/position/head_idx/feature_idx)
    const candidateRecords = findCandidateRecords(originalCircuitJson);
    console.log('🧭 Candidate records count:', candidateRecords.length);

    // Determine feature_type from nodes
    let featureTypeForNode: string | undefined = undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find(n => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find(rec => 
      matchActivationRecord(rec, parsed, featureTypeForNode)
    );
    
    if (matched) {
      console.log('✅ Matched activation record via parsing:', {
        nodeId,
        layerForActivation: parsed.layerForActivation,
        ctxIdx: parsed.ctxIdx,
        featureOrHead: parsed.featureOrHead,
        featureTypeForNode,
      });
      return {
        activations: matched.activations,
        ...normalizeZPattern(matched.zPatternIndices, matched.zPatternValues),
        nodeType: featureTypeForNode,
        clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
      };
    }

    // 3) Fallback: try fuzzy match on node_id prefix
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter(node => 
        node?.node_id && node.node_id.includes(nodeId.split('_')[0])
      );
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        console.log('🔍 Using fuzzy matched node:', {
          node_id: firstMatch.node_id,
          hasActivations: !!firstMatch.activations,
        });
        return {
          activations: firstMatch.activations,
          ...normalizeZPattern(firstMatch.zPatternIndices, firstMatch.zPatternValues),
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    console.log('❌ No matching node/record found');
    return { 
      activations: undefined, 
      zPatternIndices: undefined, 
      zPatternValues: undefined 
    };
  }, [originalCircuitJson, updateCounter]);

  /**
   * Get node activation data from specific JSON data
   */
  const getNodeActivationDataFromJson = useCallback((jsonData: any, nodeId: string | null): NodeActivationData => {
    if (!nodeId || !jsonData) {
      return { 
        activations: undefined, 
        zPatternIndices: undefined, 
        zPatternValues: undefined 
      };
    }
    
    const parsed = parseNodeId(nodeId);
    const nodesToSearch = findNodesArray(jsonData);

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find((node: any) => node?.node_id === nodeId);
      if (exactMatch) {
        if (exactMatch.activations || (exactMatch.zPatternIndices && exactMatch.zPatternValues)) {
          return {
            activations: exactMatch.activations,
            ...normalizeZPattern(exactMatch.zPatternIndices, exactMatch.zPatternValues),
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // Deep scan activation record collection
    const candidateRecords = findCandidateRecords(jsonData);

    // Determine feature_type from nodes
    let featureTypeForNode: string | undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find((n: any) => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find((rec: any) => 
      matchActivationRecord(rec, parsed, featureTypeForNode)
    );

    if (matched) {
      const clerp = (nodesToSearch.find((n: any) => n?.node_id === nodeId) || {}).clerp;
      return {
        activations: matched.activations,
        ...normalizeZPattern(matched.zPatternIndices, matched.zPatternValues),
        nodeType: featureTypeForNode,
        clerp,
      };
    }

    // Fuzzy match
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter((node: any) => 
        node?.node_id && node.node_id.includes(nodeId.split('_')[0])
      );
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        return {
          activations: firstMatch.activations,
          ...normalizeZPattern(firstMatch.zPatternIndices, firstMatch.zPatternValues),
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    return { 
      activations: undefined, 
      zPatternIndices: undefined, 
      zPatternValues: undefined 
    };
  }, []);

  return {
    getNodeActivationData,
    getNodeActivationDataFromJson,
  };
};
