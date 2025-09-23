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

export interface CacheFeatureSpec {
  feature_id: number;
  layer: number;
  is_lorsa: boolean;
  analysis_name?: string;
}

export const cacheFeatures = async (
  dictionaryName: string,
  features: CacheFeatureSpec[],
  outputDir: string,
): Promise<{ saved: number; output_dir: string }> => {
  const response = await fetch(
    `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${encodeURIComponent(dictionaryName)}/cache_features`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, output_dir: outputDir }),
    }
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to cache features: ${response.status} ${response.statusText} - ${text}`);
  }
  return response.json();
};