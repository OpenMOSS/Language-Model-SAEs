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
    return metadata.lorsa_analysis_name.replace("{}", layer.toString());
  } else {
    return metadata.clt_analysis_name.replace("{}", layer.toString());
  }
}; 