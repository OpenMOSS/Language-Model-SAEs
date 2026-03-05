import { useCallback, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { ChessBoard } from "@/components/chess/chess-board";
import { FeatureSchema, type Feature } from "@/types/feature";
import { FeatureInterpretationCard } from "@/components/feature/interpret";
import camelcaseKeys from "camelcase-keys";
import { decode } from "@msgpack/msgpack";

interface ActivatedFeature {
  feature_index: number;
  activation_value: number;
}

interface FeatureActivationData {
  attn_features?: ActivatedFeature[];
  mlp_features?: ActivatedFeature[];
}

interface TopActivationSample {
  fen: string;
  activationStrength: number;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  contextId?: number;
  sampleIndex?: number;
}

interface DecoderWeightsUmapResponse {
  embedding: number[][];
  feature_ids: number[];
}

interface UmapPoint {
  featureIndex: number;
  x: number;
  y: number;
  activationValue?: number;
  isActive: boolean;
}

interface PositionFeatureUmapCardProps {
  fen: string;
  layer: number;
  componentType: "attn" | "mlp";
  position: number;
  modelName?: string;
  saeComboId?: string;
}

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const parseTopActivationData = (camelData: any): TopActivationSample[] => {
  const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
  const allSamples: any[] = [];

  for (const group of sampleGroups) {
    if (group.samples && Array.isArray(group.samples)) {
      allSamples.push(...group.samples);
    }
  }

  const chessSamples: TopActivationSample[] = [];

  for (const sample of allSamples) {
    if (sample.text) {
      const lines = sample.text.split("\n");

      for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed.includes("/")) {
          const parts = trimmed.split(/\s+/);

          if (parts.length >= 6) {
            const [boardPart, activeColor] = parts;
            const boardRows = boardPart.split("/");

            if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
              let isValidBoard = true;
              let totalSquares = 0;

              for (const row of boardRows) {
                if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                  isValidBoard = false;
                  break;
                }

                let rowSquares = 0;
                for (const char of row) {
                  if (/\d/.test(char)) {
                    rowSquares += parseInt(char);
                  } else {
                    rowSquares += 1;
                  }
                }
                totalSquares += rowSquares;
              }

              if (isValidBoard && totalSquares === 64) {
                let activationsArray: number[] | undefined = undefined;
                let maxActivation = 0;

                if (
                  sample.featureActsIndices &&
                  sample.featureActsValues &&
                  Array.isArray(sample.featureActsIndices) &&
                  Array.isArray(sample.featureActsValues)
                ) {
                  activationsArray = new Array(64).fill(0);

                  for (
                    let i = 0;
                    i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length);
                    i++
                  ) {
                    const index = sample.featureActsIndices[i];
                    const value = sample.featureActsValues[i];

                    if (index >= 0 && index < 64) {
                      activationsArray[index] = value;
                      if (Math.abs(value) > Math.abs(maxActivation)) {
                        maxActivation = value;
                      }
                    }
                  }
                }

                chessSamples.push({
                  fen: trimmed,
                  activationStrength: maxActivation,
                  activations: activationsArray,
                  zPatternIndices: sample.zPatternIndices,
                  zPatternValues: sample.zPatternValues,
                  contextId: sample.contextIdx || sample.context_idx,
                  sampleIndex: sample.sampleIndex || 0,
                });

                break;
              }
            }
          }
        }
      }
    }
  }

  return chessSamples.sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength));
};

export const PositionFeatureUmapCard = ({
  fen,
  layer,
  componentType,
  position,
  modelName = "lc0/BT4-1024x15x32h",
  saeComboId,
}: PositionFeatureUmapCardProps) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [umapPoints, setUmapPoints] = useState<UmapPoint[]>([]);
  const [selectedPoint, setSelectedPoint] = useState<UmapPoint | null>(null);
  const [detailFeature, setDetailFeature] = useState<Feature | null>(null);
  const [topActivation, setTopActivation] = useState<TopActivationSample | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const getDictionaryName = useCallback((): string => {
    const suffix = componentType === "attn" ? "A" : "M";
    const baseDict = `BT4_${componentType === "attn" ? "lorsa" : "tc"}_L${layer}${suffix}`;
    if (saeComboId && saeComboId !== "k_128_e_128") {
      const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, "k$1_e$2");
      return `${baseDict}_${comboParts}`;
    }
    return baseDict;
  }, [componentType, layer, saeComboId]);

  const loadUmapAndFeatures = useCallback(async () => {
    if (!fen?.trim()) {
      setError("FEN must not be empty");
      return;
    }
    if (position < 0 || position > 63) {
      setError("Position must be between 0 and 63");
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedPoint(null);
    setDetailFeature(null);
    setTopActivation(null);

    try {
      const dictionary = getDictionaryName();

      const [umapRes, featuresRes] = await Promise.all([
        fetch(
          `${BACKEND_URL}/dictionaries/${encodeURIComponent(dictionary)}/decoder-weights-umap`,
          {
            method: "GET",
            headers: {
              Accept: "application/json",
            },
          }
        ),
        fetch(`${BACKEND_URL}/activation/get_features_at_position`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            fen: fen.trim(),
            layer,
            pos: position,
            component_type: componentType,
            model_name: modelName,
            sae_combo_id: saeComboId,
          }),
        }),
      ]);

      if (!umapRes.ok) {
        throw new Error(await umapRes.text());
      }
      if (!featuresRes.ok) {
        throw new Error(await featuresRes.text());
      }

      const umapData = (await umapRes.json()) as DecoderWeightsUmapResponse;
      const featureData = (await featuresRes.json()) as FeatureActivationData;

      const activeFeatures: ActivatedFeature[] =
        componentType === "attn" ? featureData.attn_features ?? [] : featureData.mlp_features ?? [];

      const activationMap = new Map<number, number>();
      for (const f of activeFeatures) {
        activationMap.set(f.feature_index, f.activation_value);
      }

      const { embedding, feature_ids } = umapData;
      if (!Array.isArray(embedding) || !Array.isArray(feature_ids) || embedding.length !== feature_ids.length) {
        throw new Error("Invalid UMAP data returned from backend");
      }

      const points: UmapPoint[] = embedding.map((coord, idx) => {
        const featureIndex = feature_ids[idx];
        const activationValue = activationMap.get(featureIndex);
        return {
          featureIndex,
          x: coord[0] ?? 0,
          y: coord[1] ?? 0,
          activationValue,
          isActive: activationValue !== undefined,
        };
      });

      setUmapPoints(points);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setUmapPoints([]);
    } finally {
      setLoading(false);
    }
  }, [fen, layer, position, componentType, modelName, saeComboId, getDictionaryName]);

  const loadFeatureDetail = useCallback(
    async (featureIndex: number) => {
      setLoadingDetail(true);
      setDetailFeature(null);
      setTopActivation(null);
      try {
        const dictionary = getDictionaryName();
        const res = await fetch(
          `${BACKEND_URL}/dictionaries/${encodeURIComponent(dictionary)}/features/${featureIndex}`,
          {
            method: "GET",
            headers: {
              Accept: "application/x-msgpack",
            },
          }
        );

        if (!res.ok) {
          throw new Error(await res.text());
        }

        const arrayBuffer = await res.arrayBuffer();
        const decoded = decode(new Uint8Array(arrayBuffer)) as Record<string, unknown>;
        const camel = camelcaseKeys(decoded, {
          deep: true,
          stopPaths: ["sample_groups.samples.context"],
        });

        const feature = FeatureSchema.parse(camel) as Feature;
        setDetailFeature(feature);

        const chessSamples = parseTopActivationData(camel);
        setTopActivation(chessSamples.length > 0 ? chessSamples[0] : null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setDetailFeature(null);
        setTopActivation(null);
      } finally {
        setLoadingDetail(false);
      }
    },
    [getDictionaryName]
  );

  const { scaledPoints } = useMemo(() => {
    if (umapPoints.length === 0) {
      return {
        scaledPoints: [] as (UmapPoint & { sx: number; sy: number })[],
      };
    }

    let localMinX = Number.POSITIVE_INFINITY;
    let localMaxX = Number.NEGATIVE_INFINITY;
    let localMinY = Number.POSITIVE_INFINITY;
    let localMaxY = Number.NEGATIVE_INFINITY;

    for (const p of umapPoints) {
      if (p.x < localMinX) localMinX = p.x;
      if (p.x > localMaxX) localMaxX = p.x;
      if (p.y < localMinY) localMinY = p.y;
      if (p.y > localMaxY) localMaxY = p.y;
    }

    // Use a slightly larger canvas to visually spread points out more.
    const padding = 32;
    const width = 480 - padding * 2;
    const height = 480 - padding * 2;
    const dx = localMaxX - localMinX || 1;
    const dy = localMaxY - localMinY || 1;

    // Draw inactive points first, then active ones so that active features
    // always appear visually on top of the layer.
    const orderedPoints = [...umapPoints].sort((a, b) => {
      if (a.isActive === b.isActive) return 0;
      return a.isActive ? 1 : -1;
    });

    const sp = orderedPoints.map((p) => ({
      ...p,
      sx: padding + ((p.x - localMinX) / dx) * width,
      sy: padding + ((localMaxY - p.y) / dy) * height,
    }));

    return {
      scaledPoints: sp,
    };
  }, [umapPoints]);

  const activeCount = useMemo(
    () => umapPoints.filter((p) => p.isActive).length,
    [umapPoints]
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Feature UMAP for Position
          <div className="text-sm font-normal mt-2 text-gray-600">
            FEN: <code className="bg-gray-100 px-2 py-1 rounded text-xs break-all">{fen}</code>
            <br />
            Layer: {layer} | Component: {componentType === "attn" ? "Attention (Lorsa)" : "MLP (Transcoder)"} | Position:{" "}
            {position}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={loadUmapAndFeatures} disabled={loading || !fen.trim()}>
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Loading UMAP...
                </>
              ) : (
                "Load Umap"
              )}
            </Button>
            {umapPoints.length > 0 && (
              <div className="text-sm text-gray-600">
                Total features: {umapPoints.length}
                {", "}
                activated at position {position}: {activeCount}
              </div>
            )}
          </div>

          {error && (
            <div className="text-red-500 font-bold text-sm bg-red-50 border border-red-200 rounded px-3 py-2 max-w-xl">
              {error}
            </div>
          )}

          {umapPoints.length === 0 && !loading && !error && (
            <div className="text-sm text-gray-500">
              Click <span className="font-semibold">Load Umap</span> to compute the decoder-weights UMAP for the current
              dictionary and highlight features activated at position {position}.
            </div>
          )}

          {umapPoints.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
              <div>
                <div className="mb-2 text-sm text-gray-600">
                  UMAP over all SAE decoder features (blue dots are features activated at position {position}, gray are
                  inactive). Click a dot to inspect its top activation and interpretation.
                </div>
                <svg
                  viewBox="0 0 480 480"
                  className="w-full h-96 border rounded bg-white"
                  role="img"
                  aria-label="UMAP embedding of features"
                >
                  <rect x={0} y={0} width={480} height={480} fill="#ffffff" />
                  {scaledPoints.map((p) => {
                    const isSelected = selectedPoint?.featureIndex === p.featureIndex;
                    const fill = p.isActive ? "#2563EB" : "rgba(148, 163, 184, 0.4)";
                    const radius = p.isActive ? 3 : 2;
                    const stroke = isSelected ? "#F97316" : p.isActive ? "#1D4ED8" : "rgba(148, 163, 184, 0.8)";
                    const strokeWidth = isSelected ? 2 : p.isActive ? 1.5 : 1;
                    return (
                      <circle
                        key={p.featureIndex}
                        cx={p.sx}
                        cy={p.sy}
                        r={radius}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        className="cursor-pointer transition-transform hover:scale-110"
                        onClick={() => {
                          setSelectedPoint(p);
                          void loadFeatureDetail(p.featureIndex);
                        }}
                      >
                        <title>
                          {`Feature #${p.featureIndex}${
                            p.activationValue !== undefined ? ` | act=${p.activationValue.toFixed(4)}` : ""
                          }`}
                        </title>
                      </circle>
                    );
                  })}
                </svg>
                <div className="mt-2 text-xs text-gray-500">
                  X/Y axes are arbitrary UMAP dimensions over decoder weights; nearby points tend to have similar decoder
                  directions.
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-base mb-2">Selected feature details</h3>
                  {selectedPoint ? (
                    <div className="text-sm text-gray-700">
                      <div>
                        Feature <span className="font-mono font-semibold">#{selectedPoint.featureIndex}</span>
                      </div>
                      <div>
                        Activation at position {position}:{" "}
                        {selectedPoint.activationValue !== undefined ? (
                          <span
                            className={
                              selectedPoint.activationValue > 0 ? "text-green-600 font-medium" : "text-red-600 font-medium"
                            }
                          >
                            {selectedPoint.activationValue > 0 ? "+" : ""}
                            {selectedPoint.activationValue.toFixed(6)}
                          </span>
                        ) : (
                          <span className="text-gray-500">inactive at this position</span>
                        )}
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        Dictionary: <code className="bg-gray-100 px-1 py-0.5 rounded text-[10px]">{getDictionaryName()}</code>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500">
                      Click any point in the UMAP on the left to view its top activation and interpretation here.
                    </div>
                  )}
                </div>

                {loadingDetail && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading top activation and interpretation...
                  </div>
                )}

                {topActivation && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">
                        Top activation sample — Feature #{selectedPoint?.featureIndex}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-xs text-gray-600 mb-2">
                        Maximum activation value: {topActivation.activationStrength.toFixed(3)}
                      </div>
                      <div className="flex justify-center">
                        <div className="origin-center" style={{ transform: "scale(0.8)" }}>
                          <ChessBoard
                            fen={topActivation.fen}
                            size="small"
                            showCoordinates={true}
                            activations={topActivation.activations}
                            zPatternIndices={topActivation.zPatternIndices}
                            zPatternValues={topActivation.zPatternValues}
                            sampleIndex={topActivation.sampleIndex}
                            analysisName={
                              topActivation.contextId != null ? `Context ${topActivation.contextId}` : undefined
                            }
                            flip_activation={topActivation.fen.includes(" b ")}
                            autoFlipWhenBlack={true}
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {detailFeature && (
                  <FeatureInterpretationCard feature={detailFeature} title="Interpretation" />
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

