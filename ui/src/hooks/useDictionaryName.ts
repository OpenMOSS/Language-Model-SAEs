/**
 * Custom hook for getting dictionary names based on layer and feature type
 */

import { useCallback } from "react";
import {
  buildBt4DictionaryFromAnalysisName,
  buildBt4DictionaryName,
  buildBt4FeatureName,
  isValidBt4AnalysisName,
} from "@/utils/bt4Sae";

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
  const analysisName = isLorsa
    ? metadata?.lorsa_analysis_name
    : metadata?.tc_analysis_name || metadata?.clt_analysis_name;
  if (typeof analysisName === "string" && analysisName.includes("{}")) {
    return analysisName;
  }

  return buildBt4DictionaryFromAnalysisName(
    analysisName,
    0,
    isLorsa ? "lorsa" : "tc",
  ).replace("L0", "L{}");
};

export const useDictionaryName = ({ linkGraphData }: UseDictionaryNameOptions) => {
  /**
   * Get dictionary name based on layer index and feature type
   */
  const getDictionaryName = useCallback((layerIdx: number, isLorsa: boolean): string => {
    const metadata = linkGraphData?.metadata || {};
    const analysisName = isLorsa
      ? metadata?.lorsa_analysis_name
      : metadata?.tc_analysis_name || metadata?.clt_analysis_name;
    if (isLorsa) {
      if (typeof analysisName === "string" && !isValidBt4AnalysisName(analysisName) && !analysisName.includes("{}")) {
        console.warn(`⚠️ Unexpected Lorsa analysis_name "${analysisName}", falling back to default BT4 combo.`);
      }
    } else if (typeof analysisName === "string" && !isValidBt4AnalysisName(analysisName) && !analysisName.includes("{}")) {
      console.warn(`⚠️ Unexpected TC analysis_name "${analysisName}", falling back to default BT4 combo.`);
    }

    return buildBt4DictionaryFromAnalysisName(
      analysisName,
      layerIdx,
      isLorsa ? "lorsa" : "tc",
    );
  }, [linkGraphData]);

  /**
   * Get SAE name for CircuitInterpretation component
   */
  const getSaeNameForCircuit = useCallback((layer: number, isLorsa: boolean) => {
    try {
      const metadata = linkGraphData?.metadata || {};
      const template = getSaeNameTemplate(layer, isLorsa, metadata);
      if (template.includes("{}")) {
        return template.replace("{}", layer.toString());
      }
      return buildBt4FeatureName(layer, 0, isLorsa ? "lorsa" : "tc").replace(/#0$/, "");
    } catch (error) {
      console.error('[CircuitVisualization] Error in getSaeNameForCircuit:', error);
      return buildBt4DictionaryName(layer, isLorsa ? "lorsa" : "tc");
    }
  }, [linkGraphData]);

  return {
    getDictionaryName,
    getSaeNameForCircuit,
  };
};
