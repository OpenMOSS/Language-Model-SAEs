/**
 * Custom hook for getting dictionary names based on layer and feature type
 */

import { useCallback } from "react";

const KNOWN_ANALYSIS_NAMES = [
  "BT4_tc_k30_e16",
  "BT4_lorsa_k30_e16",
  "BT4_tc_k64_e32",
  "BT4_lorsa_k64_e32",
  "BT4_tc_k128_e64",
  "BT4_lorsa_k128_e64",
  "BT4_tc_k256_e128",
  "BT4_lorsa_k256_e128",
  "BT4_tc",  // k128_e128 (default combo, no suffix)
  "BT4_lorsa",  // k128_e128 (default combo, no suffix)
];

interface UseDictionaryNameOptions {
  linkGraphData: any;
}

/**
 * Get SAE name template (without layer number)
 */
export const getSaeNameTemplate = (
  _layerIdx: number,
  isLorsa: boolean,
  metadata: any
): string => {
  if (isLorsa) {
    const lorsaAnalysisName = metadata?.lorsa_analysis_name;
    if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
      if (lorsaAnalysisName === "BT4_lorsa") {
        return "BT4_lorsa_L{}A";
      } else {
        const suffix = lorsaAnalysisName.replace("BT4_lorsa_", "");
        return `BT4_lorsa_L{}A_${suffix}`;
      }
    }
    return "BT4_lorsa_L{}A"; // Default
  } else {
    const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
    if (tcAnalysisName && typeof tcAnalysisName === 'string') {
      if (tcAnalysisName === "BT4_tc") {
        return "BT4_tc_L{}M";
      } else {
        const suffix = tcAnalysisName.replace("BT4_tc_", "");
        return `BT4_tc_L{}M_${suffix}`;
      }
    }
    return "BT4_tc_L{}M"; // Default
  }
};

export const useDictionaryName = ({ linkGraphData }: UseDictionaryNameOptions) => {
  /**
   * Get dictionary name based on layer index and feature type
   */
  const getDictionaryName = useCallback((layerIdx: number, isLorsa: boolean): string => {
    const metadata = linkGraphData?.metadata || {};
    let dictionary: string;
    let usingDefault = false;
    
    if (isLorsa) {
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
        const isKnown = KNOWN_ANALYSIS_NAMES.includes(lorsaAnalysisName);
        
        if (isKnown) {
          if (lorsaAnalysisName === "BT4_lorsa") {
            dictionary = `BT4_lorsa_L${layerIdx}A`;
          } else {
            const suffix = lorsaAnalysisName.replace("BT4_lorsa_", "");
            dictionary = `BT4_lorsa_L${layerIdx}A_${suffix}`;
          }
        } else {
          console.warn(`⚠️ LoRSA analysis_name "${lorsaAnalysisName}" is not in known combinations, using default combo k128_e128`);
          dictionary = `BT4_lorsa_L${layerIdx}A`;
          usingDefault = true;
        }
      } else {
        dictionary = `BT4_lorsa_L${layerIdx}A`;
        usingDefault = true;
      }
    } else {
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      if (tcAnalysisName && typeof tcAnalysisName === 'string') {
        const isKnown = KNOWN_ANALYSIS_NAMES.includes(tcAnalysisName);
        
        if (isKnown) {
          if (tcAnalysisName === "BT4_tc") {
            dictionary = `BT4_tc_L${layerIdx}M`;
          } else {
            const suffix = tcAnalysisName.replace("BT4_tc_", "");
            dictionary = `BT4_tc_L${layerIdx}M_${suffix}`;
          }
        } else {
          console.warn(`⚠️ TC analysis_name "${tcAnalysisName}" is not in known combinations, using default combo k128_e128`);
          dictionary = `BT4_tc_L${layerIdx}M`;
          usingDefault = true;
        }
      } else {
        dictionary = `BT4_tc_L${layerIdx}M`;
        usingDefault = true;
      }
    }
    
    if (usingDefault) {
      console.warn(`⚠️ Using default combo k128_e128 dictionary name: ${dictionary}`);
    }
    
    return dictionary;
  }, [linkGraphData]);

  /**
   * Get SAE name for CircuitInterpretation component
   */
  const getSaeNameForCircuit = useCallback((layer: number, isLorsa: boolean) => {
    console.log('[CircuitVisualization] getSaeNameForCircuit called:', { layer, isLorsa });
    try {
      const metadata = linkGraphData?.metadata || {};
      const template = getSaeNameTemplate(layer, isLorsa, metadata);
      console.log('[CircuitVisualization] Template result:', template);
      const result = template.replace('{}', layer.toString());
      console.log('[CircuitVisualization] Final SAE name:', result);
      return result;
    } catch (error) {
      console.error('[CircuitVisualization] Error in getSaeNameForCircuit:', error);
      // Return default value
      const fallback = isLorsa ? `BT4_lorsa_L${layer}A` : `BT4_tc_L${layer}M`;
      console.log('[CircuitVisualization] Using fallback:', fallback);
      return fallback;
    }
  }, [linkGraphData]);

  return {
    getDictionaryName,
    getSaeNameForCircuit,
  };
};
