const BT4_COMBO_RE = /^k_(\d+)_e_(\d+)$/;
const BT4_ANALYSIS_RE = /^BT4_(tc|lorsa)(?:_(k\d+_e\d+))?$/;
const BT4_FEATURE_RE = /^BT4_(tc|lorsa)_L(\d+)(M|A)(?:_(k\d+_e\d+))?#(\d+)$/;

export const DEFAULT_BT4_SAE_COMBO_ID = "k_30_e_16";

export type Bt4FeatureType = "tc" | "lorsa" | "transcoder";

export const normalizeBt4FeatureType = (featureType: Bt4FeatureType): "tc" | "lorsa" =>
  featureType === "lorsa" ? "lorsa" : "tc";

export const bt4ComboIdToSuffix = (comboId?: string | null): string => {
  const match = (comboId ?? DEFAULT_BT4_SAE_COMBO_ID).match(BT4_COMBO_RE);
  if (!match) {
    return "k30_e16";
  }
  return `k${match[1]}_e${match[2]}`;
};

export const buildBt4AnalysisName = (
  featureType: Bt4FeatureType,
  comboId?: string | null,
): string => `BT4_${normalizeBt4FeatureType(featureType)}_${bt4ComboIdToSuffix(comboId)}`;

export const buildBt4DictionaryName = (
  layer: number,
  featureType: Bt4FeatureType,
  comboId?: string | null,
): string => {
  const normalizedType = normalizeBt4FeatureType(featureType);
  const componentSuffix = normalizedType === "lorsa" ? "A" : "M";
  return `BT4_${normalizedType}_L${layer}${componentSuffix}_${bt4ComboIdToSuffix(comboId)}`;
};

export const buildBt4FeatureName = (
  layer: number,
  featureIndex: number,
  featureType: Bt4FeatureType,
  comboId?: string | null,
): string => `${buildBt4DictionaryName(layer, featureType, comboId)}#${featureIndex}`;

export const buildBt4AnalysisTemplate = (
  featureType: Bt4FeatureType,
  comboId?: string | null,
): string => {
  const normalizedType = normalizeBt4FeatureType(featureType);
  const componentSuffix = normalizedType === "lorsa" ? "A" : "M";
  return `BT4_${normalizedType}_L{}${componentSuffix}_${bt4ComboIdToSuffix(comboId)}`;
};

export const isValidBt4AnalysisName = (analysisName?: string | null): boolean =>
  !!analysisName && BT4_ANALYSIS_RE.test(analysisName);

export const buildBt4DictionaryFromAnalysisName = (
  analysisName: string | undefined | null,
  layer: number,
  featureType: Bt4FeatureType,
): string => {
  if (analysisName?.includes("{}")) {
    return analysisName.replace("{}", layer.toString());
  }

  const normalizedType = normalizeBt4FeatureType(featureType);
  const match = analysisName?.match(BT4_ANALYSIS_RE);
  if (match && match[1] === normalizedType) {
    const comboSuffix = match[2] ?? bt4ComboIdToSuffix(DEFAULT_BT4_SAE_COMBO_ID);
    const componentSuffix = normalizedType === "lorsa" ? "A" : "M";
    return `BT4_${normalizedType}_L${layer}${componentSuffix}_${comboSuffix}`;
  }

  return buildBt4DictionaryName(layer, normalizedType, DEFAULT_BT4_SAE_COMBO_ID);
};

export const parseBt4FeatureName = (
  featureName: string,
): { layerIdx: number; featureIdx: number; featureType: "tc" | "lorsa"; comboSuffix: string | null } | null => {
  const match = featureName.match(BT4_FEATURE_RE);
  if (!match) {
    return null;
  }

  return {
    featureType: match[1] as "tc" | "lorsa",
    layerIdx: Number.parseInt(match[2], 10),
    featureIdx: Number.parseInt(match[5], 10),
    comboSuffix: match[4] ?? null,
  };
};
