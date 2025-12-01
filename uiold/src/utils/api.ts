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
    // 支持lorsa_analysis_name字段
    const analysisName = metadata.lorsa_analysis_name;
    if (analysisName && analysisName.includes('BT4')) {
      // BT4格式: BT4_lorsa_L{layer}A
      return `BT4_lorsa_L${layer}A`;
    } else {
      return analysisName ? analysisName.replace("{}", layer.toString()) : `lc0-lorsa-L${layer}`;
    }
  } else {
    // 支持新的字段名tc_analysis_name，向后兼容clt_analysis_name
    const analysisName = metadata.tc_analysis_name || metadata.clt_analysis_name;
    if (analysisName && analysisName.includes('BT4')) {
      // BT4格式: BT4_tc_L{layer}M
      return `BT4_tc_L${layer}M`;
    } else {
      return analysisName ? analysisName.replace("{}", layer.toString()) : `lc0_L${layer}M_16x_k30_lr2e-03_auxk_sparseadam`;
    }
  }
}; 