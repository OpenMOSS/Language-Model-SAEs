import { useState, useCallback, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAsyncFn } from "react-use";
import { ChessBoard } from "@/components/chess/chess-board";
import { Link } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Label } from "@/components/ui/label";
import { GetFeatureFromFen } from "@/components/feature/get-feature-from-fen";

interface PosFeatureCardProps {
  fen: string;
  layer: number;
  positions: number[] | string;
  componentType: "attn" | "mlp";
  modelName?: string;
  saeComboId?: string;
  onFeatureSelect?: (selection: { featureIndex: number; position: number } | null) => void;
}

interface ActivatedFeature {
  feature_index: number;
  activation_value: number;
}

interface PositionFeatures {
  position: number;
  features: ActivatedFeature[];
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

interface SteeringNode {
  pos: number;
  feature: number;
  steering_scale: number;
}

interface SteeringMoveDiff {
  uci: string;
  diff: number;
  original_logit: number;
  modified_logit: number;
  prob_diff?: number;
  original_prob?: number;
  modified_prob?: number;
  idx?: number;
}

interface MultiSteeringResult {
  promoting_moves: SteeringMoveDiff[];
  inhibiting_moves: SteeringMoveDiff[];
  top_moves_by_prob?: SteeringMoveDiff[];
  statistics: {
    total_legal_moves: number;
    avg_logit_diff: number;
    max_logit_diff: number;
    min_logit_diff: number;
    avg_prob_diff?: number;
    max_prob_diff?: number;
    min_prob_diff?: number;
    original_value?: number;
    modified_value?: number;
    value_diff?: number;
  };
  ablation_info?: {
    feature_type?: string;
    layer?: number;
    nodes?: Array<{
      pos: number;
      feature: number;
      steering_scale: number;
      activation_value?: number;
    }>;
  };
}

interface AnalyzeFenResponse {
  feature_acts_indices?: number[];
  feature_acts_values?: number[];
  z_pattern_indices?: number[] | number[][];
  z_pattern_values?: number[];
}

interface FenActivationData {
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
}

export const _testParsePositionsInput = (positionsInput: number[] | string): number[] => {
  if (Array.isArray(positionsInput)) {
    return positionsInput;
  }

  const result: number[] = [];
  const parts = positionsInput.split(',');

  for (const part of parts) {
    const trimmed = part.trim();
    if (trimmed.includes('-')) {
      const [startStr, endStr] = trimmed.split('-');
      const start = parseInt(startStr.trim(), 10);
      const end = parseInt(endStr.trim(), 10);

      if (!isNaN(start) && !isNaN(end) && start <= end) {
        for (let i = start; i <= end; i++) {
          if (!result.includes(i)) {
            result.push(i);
          }
        }
      }
    } else {
      const num = parseInt(trimmed, 10);
      if (!isNaN(num) && !result.includes(num)) {
        result.push(num);
      }
    }
  }

  return result
    .filter(pos => pos >= 0 && pos <= 63)
    .sort((a, b) => a - b);
};

if (typeof window !== 'undefined') {
  console.log('Testing parsing logic:');
  console.log('  "0-2,3" ->', _testParsePositionsInput("0-2,3"));
  console.log('  "0-7" ->', _testParsePositionsInput("0-7"));
  console.log('  "0,2,4" ->', _testParsePositionsInput("0,2,4"));
  console.log('  "8-10,15" ->', _testParsePositionsInput("8-10,15"));
  console.log('  [0,1,2] ->', _testParsePositionsInput([0,1,2]));
}

export const PosFeatureCard = ({
  fen,
  layer,
  positions,
  componentType,
  modelName = "lc0/BT4-1024x15x32h",
  saeComboId,
  onFeatureSelect,
}: PosFeatureCardProps) => {
  console.log('PosFeatureCard received props:', { fen, layer, positions, componentType });
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<number | null>(null);
  const [topActivations, setTopActivations] = useState<TopActivationSample[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);
  const [selectedFeaturePos, setSelectedFeaturePos] = useState<number | null>(null);
  const [steeringNodes, setSteeringNodes] = useState<SteeringNode[]>([]);
  const [defaultSteeringScale, setDefaultSteeringScale] = useState<number>(2.0);
  const [autoSteerThreshold, setAutoSteerThreshold] = useState<number>(0);

  useEffect(() => {
    console.log('positions prop changed:', positions);
  }, [positions]);

  const parseScaleInput = useCallback((raw: string): number => {
    const v = parseFloat(raw);
    return Number.isFinite(v) ? v : 1;
  }, []);

const parsePositionsInput = useCallback((positionsInput: number[] | string): number[] => {
  if (Array.isArray(positionsInput)) {
    return positionsInput;
  }

  console.log('Starting to parse string:', positionsInput);

  const result: number[] = [];
  const parts = positionsInput.split(',');

  console.log('Split parts:', parts);

  for (const part of parts) {
    const trimmed = part.trim();
    console.log('Processing part:', trimmed);

    if (trimmed.includes('-')) {
      const rangeParts = trimmed.split('-');
      console.log('Range parts:', rangeParts);

      if (rangeParts.length === 2) {
        const startStr = rangeParts[0].trim();
        const endStr = rangeParts[1].trim();
        const start = parseInt(startStr, 10);
        const end = parseInt(endStr, 10);

        console.log('Parsed range:', { startStr, endStr, start, end });

        if (!isNaN(start) && !isNaN(end) && start <= end) {
          console.log('Starting to add range:', { start, end });
          for (let i = start; i <= end; i++) {
            if (!result.includes(i)) {
              result.push(i);
              console.log('Added to result:', i);
            }
          }
        } else {
          console.log('Range is invalid:', { start, end });
        }
      }
    } else {
      const num = parseInt(trimmed, 10);
      console.log('Parsed single number:', { trimmed, num });

      if (!isNaN(num) && !result.includes(num)) {
        result.push(num);
        console.log('Added single number to result:', num);
      }
    }
  }

  console.log('Result before filtering:', result);

  const filtered = result
    .filter(pos => pos >= 0 && pos <= 63)
    .sort((a, b) => a - b);

  console.log('Final result:', filtered);
  return filtered;
}, []);

  const parsedPositions = useMemo(() => {
    const result = parsePositionsInput(positions);
    console.log('Parsed positions input:', {
      input: positions,
      inputType: typeof positions,
      isArray: Array.isArray(positions),
      output: result,
      outputLength: result.length
    });
    console.log('parsedPositions details:', result);
    return result;
  }, [positions, parsePositionsInput]);

  const isSinglePosition = parsedPositions.length === 1;
  const singlePosition = isSinglePosition ? parsedPositions[0] : null;

  const [featuresState, fetchFeatures] = useAsyncFn(async () => {
    if (!fen || parsedPositions.length === 0) {
      return null;
    }

    const positionFeatures: PositionFeatures[] = [];

    for (const pos of parsedPositions) {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/activation/get_features_at_position`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            fen: fen.trim(),
            layer: layer,
            pos: pos,
            component_type: componentType,
            model_name: modelName,
            sae_combo_id: saeComboId,
          }),
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const data: FeatureActivationData = await response.json();
      const features = componentType === "attn" ? data.attn_features : data.mlp_features;

      if (features && Array.isArray(features)) {
        const sortedFeatures = [...features].sort((a, b) => a.feature_index - b.feature_index);

        positionFeatures.push({
          position: pos,
          features: sortedFeatures,
        });
      } else {
        positionFeatures.push({
          position: pos,
          features: [],
        });
      }
    }

    return positionFeatures;
  }, [fen, layer, parsedPositions, componentType, modelName, saeComboId]);

  const backendFeatureType = componentType === "attn" ? "lorsa" : "transcoder";

  const parseAnalyzeFen = useCallback((data: AnalyzeFenResponse): FenActivationData => {
    let activations: number[] | undefined = undefined;
    if (Array.isArray(data.feature_acts_indices) && Array.isArray(data.feature_acts_values)) {
      activations = new Array(64).fill(0);
      const indices = data.feature_acts_indices;
      const values = data.feature_acts_values;
      for (let i = 0; i < Math.min(indices.length, values.length); i++) {
        const idx = indices[i];
        const v = values[i];
        if (typeof idx === "number" && idx >= 0 && idx < 64 && typeof v === "number") {
          activations[idx] = v;
        }
      }
    }

    let zPatternIndices: number[][] | undefined = undefined;
    let zPatternValues: number[] | undefined = undefined;
    if (data.z_pattern_indices && data.z_pattern_values) {
      const raw = data.z_pattern_indices;
      zPatternIndices = Array.isArray(raw) && Array.isArray(raw[0]) ? (raw as number[][]) : [raw as number[]];
      zPatternValues = data.z_pattern_values;
    }

    return { activations, zPatternIndices, zPatternValues };
  }, []);

  const addSteeringNode = useCallback(
    (pos: number, feature: number) => {
      setSteeringNodes((prev) => {
        if (prev.some((n) => n.pos === pos && n.feature === feature)) {
          return prev;
        }
        return [...prev, { pos, feature, steering_scale: defaultSteeringScale }];
      });
    },
    [defaultSteeringScale]
  );

  const addAllSteeringNodesAtPosition = useCallback(
    (pos: number, features: ActivatedFeature[]) => {
      if (!features || features.length === 0) return;
      setSteeringNodes((prev) => {
        const existing = new Set(prev.filter((n) => n.pos === pos).map((n) => n.feature));
        const next: SteeringNode[] = [...prev];
        for (const f of features) {
          if (!existing.has(f.feature_index)) {
            next.push({ pos, feature: f.feature_index, steering_scale: defaultSteeringScale });
            existing.add(f.feature_index);
          }
        }
        return next;
      });
    },
    [defaultSteeringScale]
  );

  const applySteeringScaleToAllNodes = useCallback((scale: number) => {
    setSteeringNodes((prev) => prev.map((n) => ({ ...n, steering_scale: scale })));
  }, []);

  const updateSteeringScale = useCallback((pos: number, feature: number, scale: number) => {
    setSteeringNodes((prev) =>
      prev.map((n) => (n.pos === pos && n.feature === feature ? { ...n, steering_scale: scale } : n))
    );
  }, []);

  const removeSteeringNode = useCallback((pos: number, feature: number) => {
    setSteeringNodes((prev) => prev.filter((n) => !(n.pos === pos && n.feature === feature)));
  }, []);

  const clearSteeringNodes = useCallback(() => {
    setSteeringNodes([]);
  }, []);

  const [steeringState, runMultiSteering] = useAsyncFn(async () => {
    if (!fen?.trim()) {
      throw new Error("FEN cannot be empty");
    }
    if (steeringNodes.length === 0) {
      throw new Error("Please add at least one feature to steer");
    }

    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/steering_analysis/multi`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        fen: fen.trim(),
        feature_type: backendFeatureType,
        layer,
        nodes: steeringNodes.map((n) => ({
          pos: n.pos,
          feature: n.feature,
          steering_scale: n.steering_scale,
        })),
        metadata: {
          model_name: modelName,
          sae_combo_id: saeComboId,
          source: "position_feature_page",
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }
    return (await response.json()) as MultiSteeringResult;
  }, [fen, steeringNodes, backendFeatureType, layer, modelName, saeComboId]);

  const getDictionaryName = useCallback(() => {
    const lorsaSuffix = componentType === "attn" ? "A" : "M";
    const baseDict = `BT4_${componentType === "attn" ? "lorsa" : "tc"}_L${layer}${lorsaSuffix}`;
    if (saeComboId && saeComboId !== "k_128_e_128") {
      const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, "k$1_e$2");
      return `${baseDict}_${comboParts}`;
    }
    return baseDict;
  }, [componentType, layer, saeComboId]);

  const [fenActivationState, fetchFenActivationForSelectedFeature] = useAsyncFn(async () => {
    if (!fen?.trim()) {
      throw new Error("FEN cannot be empty");
    }
    if (selectedFeatureIndex === null) {
      return null;
    }

    const dictionary = getDictionaryName();
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${selectedFeatureIndex}/analyze_fen`,
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

    const data = (await response.json()) as AnalyzeFenResponse;
    return parseAnalyzeFen(data);
  }, [fen, selectedFeatureIndex, getDictionaryName, parseAnalyzeFen]);

  const [autoSteerState, runAutoSteerAllPositions] = useAsyncFn(async () => {
    if (!fen?.trim()) {
      throw new Error("FEN cannot be empty");
    }
    if (selectedFeatureIndex === null) {
      throw new Error("Please select a feature in the list above");
    }

    const dictionary = getDictionaryName();
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${selectedFeatureIndex}/analyze_fen`,
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

    const data = (await response.json()) as AnalyzeFenResponse;
    const parsed = parseAnalyzeFen(data);
    const dense = parsed.activations ?? new Array(64).fill(0);

    const activePositions: number[] = [];
    for (let pos = 0; pos < 64; pos++) {
      const v = dense[pos] ?? 0;
      if (Math.abs(v) > autoSteerThreshold) activePositions.push(pos);
    }

    const nodes: SteeringNode[] = activePositions.map((pos) => ({
      pos,
      feature: selectedFeatureIndex,
      steering_scale: defaultSteeringScale,
    }));

    setSteeringNodes(nodes);

    return { activePositions, count: nodes.length };
  }, [fen, selectedFeatureIndex, autoSteerThreshold, defaultSteeringScale, getDictionaryName, parseAnalyzeFen]);

  const sortedPromotingMoves = useMemo(() => {
    const moves = steeringState.value?.promoting_moves ?? [];
    return [...moves].sort((a, b) => {
      const ap = a.prob_diff;
      const bp = b.prob_diff;
      if (typeof ap === "number" && typeof bp === "number") return bp - ap;
      if (typeof ap === "number") return -1;
      if (typeof bp === "number") return 1;
      return b.diff - a.diff;
    });
  }, [steeringState.value]);

  const sortedInhibitingMoves = useMemo(() => {
    const moves = steeringState.value?.inhibiting_moves ?? [];
    return [...moves].sort((a, b) => {
      const ap = a.prob_diff;
      const bp = b.prob_diff;
      if (typeof ap === "number" && typeof bp === "number") return ap - bp;
      if (typeof ap === "number") return -1;
      if (typeof bp === "number") return 1;
      return a.diff - b.diff;
    });
  }, [steeringState.value]);

  const sortedTopMovesByProb = useMemo(() => {
    const moves = steeringState.value?.top_moves_by_prob ?? [];
    return [...moves].sort((a, b) => {
      const am = typeof a.modified_prob === "number" ? a.modified_prob : a.original_prob;
      const bm = typeof b.modified_prob === "number" ? b.modified_prob : b.original_prob;
      if (typeof am === "number" && typeof bm === "number") return bm - am;
      if (typeof am === "number") return -1;
      if (typeof bm === "number") return 1;
      return b.diff - a.diff;
    });
  }, [steeringState.value]);

  const fetchTopActivationsForFeature = useCallback(
    async (featureIndex: number, isLorsa: boolean) => {
      setLoadingTopActivations(true);
      try {
        const lorsaSuffix = componentType === "attn" ? "A" : "M";
        const dictionary = `BT4_${componentType === "attn" ? "lorsa" : "tc"}_L${layer}${lorsaSuffix}`;

        let fullDictionary = dictionary;
        if (saeComboId && saeComboId !== "k_128_e_128") {
          const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, "k$1_e$2");
          fullDictionary = `${dictionary}_${comboParts}`;
        }

        console.log("Get Top Activation data:", {
          layer,
          featureIndex,
          dictionary: fullDictionary,
          isLorsa,
        });

        const response = await fetch(
          `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${fullDictionary}/features/${featureIndex}`,
          {
            method: "GET",
            headers: {
              Accept: "application/x-msgpack",
            },
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const decoded = await import("@msgpack/msgpack").then((module) =>
          module.decode(new Uint8Array(arrayBuffer))
        );
        const camelcaseKeys = await import("camelcase-keys").then((module) => module.default);

        const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
          deep: true,
          stopPaths: ["sample_groups.samples.context"],
        }) as any;

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

        const topSamples = chessSamples
          .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
          .slice(0, 8);

        setTopActivations(topSamples);
      } catch (error) {
        console.error("Failed to get Top Activation data:", error);
        setTopActivations([]);
      } finally {
        setLoadingTopActivations(false);
      }
    },
    [layer, componentType, saeComboId]
  );

  useEffect(() => {
    if (selectedFeatureIndex !== null) {
      fetchTopActivationsForFeature(selectedFeatureIndex, componentType === "attn");
    } else {
      setTopActivations([]);
    }
  }, [selectedFeatureIndex, fetchTopActivationsForFeature, componentType]);

  useEffect(() => {
    if (selectedFeatureIndex !== null) {
      fetchFenActivationForSelectedFeature();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFeatureIndex, fen, fetchFenActivationForSelectedFeature]);

  return (
    <Card key={`pos-feature-card-${JSON.stringify(positions)}`} className="w-full">
      <CardHeader>
        <CardTitle>
          <div className="text-sm font-normal mt-2 text-gray-600">
            FEN: <code className="bg-gray-100 px-2 py-1 rounded text-xs">{fen}</code>
            <br />
            Layer: {layer} | Positions: {Array.isArray(positions) ? positions.join(", ") : positions} | Component: {componentType === "attn" ? "Attention" : "MLP"}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {!isSinglePosition && (
            <Button
              onClick={() => fetchFeatures()}
              disabled={featuresState.loading || !fen || parsedPositions.length === 0}
              className="w-full"
            >
              {featuresState.loading ? "Loading..." : "Get activated Features"}
            </Button>
          )}

          {featuresState.error && (
            <div className="text-red-500 font-bold text-center p-4 bg-red-50 rounded">
              Error: {featuresState.error instanceof Error ? featuresState.error.message : String(featuresState.error)}
            </div>
          )}

          {isSinglePosition && singlePosition !== null ? (
            <GetFeatureFromFen
              fen={fen}
              layer={layer}
              position={singlePosition}
              componentType={componentType}
              modelName={modelName}
              saeComboId={saeComboId}
              actionTypes={["add_to_steer"]}
              showTopActivations={true}
              showFenActivations={true}
              wrapInCard={false}
              onFeatureAction={(action) => {
                if (action.type === "add_to_steer") {
                  addSteeringNode(singlePosition, action.featureIndex);
                }
              }}
            />
          ) : (
            featuresState.value && featuresState.value.length > 0 && (
              <div className="space-y-4">
                <h3 className="font-semibold">
                  Display activated Features by positions ({featuresState.value.length} positions)
                </h3>
                <div className="text-sm text-gray-600">
                  Parsed positions: [{parsedPositions.join(', ')}]
                </div>
                {featuresState.value.map((posFeatures, index) => {
                  console.log(`Rendering position ${index}:`, posFeatures.position, posFeatures.features.length);
                  return (
                    <div key={posFeatures.position} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between gap-3 mb-2">
                        <h4 className="font-medium text-gray-800">
                          Position {posFeatures.position}
                          {posFeatures.features.length > 0 && (
                            <span className="text-sm text-gray-500 ml-2">
                              ({posFeatures.features.length} activated Features)
                            </span>
                          )}
                        </h4>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={posFeatures.features.length === 0}
                          onClick={() => addAllSteeringNodesAtPosition(posFeatures.position, posFeatures.features)}
                          title="Add all activated Features at this position to Multi Steering (will automatically deduplicate)"
                        >
                          Add all Steer at this position
                        </Button>
                      </div>
                      {posFeatures.features.length > 0 ? (
                        <div className="max-h-64 overflow-y-auto space-y-2">
                          {posFeatures.features.map((feature, index) => (
                            <div
                              key={`${posFeatures.position}-${feature.feature_index}`}
                              className={`p-3 border rounded cursor-pointer transition-colors ${
                                selectedFeatureIndex === feature.feature_index
                                  ? "bg-blue-100 border-blue-500"
                                  : "bg-white hover:bg-gray-50"
                              }`}
                              onClick={() => {
                                const next =
                                  selectedFeatureIndex === feature.feature_index ? null : feature.feature_index;
                                setSelectedFeatureIndex(next);
                                setSelectedFeaturePos(next === null ? null : posFeatures.position);
                                if (onFeatureSelect) {
                                  onFeatureSelect(
                                    next === null ? null : { featureIndex: next, position: posFeatures.position }
                                  );
                                }
                              }}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <span className="font-mono text-sm">#{index + 1}</span>
                                  <span className="font-bold">Feature #{feature.feature_index}</span>
                                  <span
                                    className={`text-sm ${
                                      feature.activation_value > 0 ? "text-green-600" : "text-red-600"
                                    }`}
                                  >
                                    {feature.activation_value > 0 ? "+" : ""}
                                    {feature.activation_value.toFixed(4)}
                                  </span>
                                </div>
                                <div className="flex items-center gap-3">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      addSteeringNode(posFeatures.position, feature.feature_index);
                                    }}
                                  >
                                    Steer
                                  </Button>
                                  <Link
                                    to={`/features?dictionary=${encodeURIComponent(
                                      getDictionaryName()
                                    )}&featureIndex=${feature.feature_index}`}
                                    target="_blank"
                                    className="text-blue-600 hover:text-blue-800 text-sm"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    View details →
                                  </Link>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-gray-500 text-sm italic">
                          No activated Features at this position
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )
          )}

          <div className="border rounded-lg p-4 bg-gray-50">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="font-semibold">Steering (multiple features, multiple positions)</h3>
                <p className="text-xs text-gray-600 mt-1">
                  Component: {componentType === "attn" ? "Attention (LoRSA)" : "MLP (Transcoder)"}, Layer: {layer}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={clearSteeringNodes}
                  disabled={steeringNodes.length === 0 || steeringState.loading}
                >
                  Clear
                </Button>
                <Button onClick={runMultiSteering} disabled={steeringState.loading || steeringNodes.length === 0}>
                  {steeringState.loading ? "Steering..." : "Run Steering"}
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-3 mb-3">
              <Label htmlFor="default-steering-scale" className="text-sm">
                Default steering_scale
              </Label>
              <Input
                id="default-steering-scale"
                type="number"
                step="0.1"
                className="w-32 bg-white"
                value={defaultSteeringScale}
                onChange={(e) => setDefaultSteeringScale(parseScaleInput(e.target.value))}
              />
              <Button
                variant="outline"
                onClick={() => applySteeringScaleToAllNodes(defaultSteeringScale)}
                disabled={steeringNodes.length === 0 || steeringState.loading}
              >
                Apply to all
              </Button>
              <span className="text-xs text-gray-500">Newly added features will use this default value (can be negative, indicating reverse steering)</span>
            </div>

            <div className="flex flex-wrap items-center gap-3 mb-3">
              <Label htmlFor="auto-steer-threshold" className="text-sm">
                Automatic select threshold act &gt;
              </Label>
              <Input
                id="auto-steer-threshold"
                type="number"
                step="0.000001"
                className="w-40 bg-white"
                value={autoSteerThreshold}
                onChange={(e) => setAutoSteerThreshold(parseFloat(e.target.value) || 0)}
              />
              <Button
                variant="outline"
                onClick={() => runAutoSteerAllPositions()}
                disabled={autoSteerState.loading || selectedFeatureIndex === null || !fen?.trim()}
              >
                {autoSteerState.loading ? "Processing..." : "One-click: add Steer at all activated positions (replace)"}
              </Button>

              {autoSteerState.value && (
                <span className="text-xs text-gray-600">
                  Added {autoSteerState.value.count} positions
                </span>
              )}
            </div>

            {autoSteerState.error && (
              <div className="text-red-600 text-sm mb-3">
                {autoSteerState.error instanceof Error ? autoSteerState.error.message : String(autoSteerState.error)}
              </div>
            )}

            {steeringState.error && (
              <div className="text-red-600 text-sm mb-3">
                {steeringState.error instanceof Error ? steeringState.error.message : String(steeringState.error)}
              </div>
            )}

            {steeringNodes.length > 0 ? (
              <div className="mb-4">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Pos</TableHead>
                      <TableHead>Feature</TableHead>
                      <TableHead>Scale</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {steeringNodes.map((n) => (
                      <TableRow key={`${n.pos}-${n.feature}`}>
                        <TableCell>{n.pos}</TableCell>
                        <TableCell className="font-mono">{n.feature}</TableCell>
                        <TableCell className="w-40">
                          <Input
                            type="number"
                            step="0.1"
                            className="bg-white"
                            value={n.steering_scale}
                            onChange={(e) => updateSteeringScale(n.pos, n.feature, parseScaleInput(e.target.value))}
                          />
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => removeSteeringNode(n.pos, n.feature)}
                            disabled={steeringState.loading}
                          >
                            Remove
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : null}

            {steeringState.value && (
              <div className="space-y-4">
                <div className="text-sm text-gray-700 space-y-1">
                  <div>
                    <span className="font-medium">Total legal moves:</span> {steeringState.value.statistics.total_legal_moves}{" "}
                    <span className="font-medium ml-4">avg_logit_diff:</span>{" "}
                    {steeringState.value.statistics.avg_logit_diff.toFixed(6)}
                    {typeof steeringState.value.statistics.avg_prob_diff === "number" && (
                      <>
                        <span className="font-medium ml-4">avg_prob_diff:</span>{" "}
                        {steeringState.value.statistics.avg_prob_diff.toFixed(6)}
                      </>
                    )}
                  </div>
                  <div>
                    <span className="font-medium">Original Value:</span> {steeringState.value.statistics.original_value?.toFixed(6) || "N/A"}{" "}
                    <span className="font-medium ml-4">Modified Value:</span>{" "}
                    {steeringState.value.statistics.modified_value?.toFixed(6) || "N/A"}{" "}
                    <span className="font-medium ml-4">Value difference:</span>{" "}
                    <span className={(steeringState.value.statistics.value_diff ?? 0) >= 0 ? "text-green-600" : "text-red-600"}>
                      {((steeringState.value.statistics.value_diff ?? 0) >= 0 ? "+" : "") + (steeringState.value.statistics.value_diff?.toFixed(6) ?? "N/A")}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white border rounded p-3">
                    <h4 className="font-semibold mb-2 text-sm">Promoting (Top 5 with highest probability increase)</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>UCI</TableHead>
                          <TableHead className="text-right">Δprob</TableHead>
                          <TableHead className="text-right">Δlogit</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedPromotingMoves.map((m) => (
                          <TableRow key={`p-${m.uci}`}>
                            <TableCell className="font-mono">{m.uci}</TableCell>
                            <TableCell className="text-right">
                              {typeof m.prob_diff === "number" ? m.prob_diff.toFixed(6) : "-"}
                            </TableCell>
                            <TableCell className="text-right">{m.diff.toFixed(6)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  <div className="bg-white border rounded p-3">
                    <h4 className="font-semibold mb-2 text-sm">Inhibiting (Top 5 with highest probability decrease)</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>UCI</TableHead>
                          <TableHead className="text-right">Δprob</TableHead>
                          <TableHead className="text-right">Δlogit</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedInhibitingMoves.map((m) => (
                          <TableRow key={`i-${m.uci}`}>
                            <TableCell className="font-mono">{m.uci}</TableCell>
                            <TableCell className="text-right">
                              {typeof m.prob_diff === "number" ? m.prob_diff.toFixed(6) : "-"}
                            </TableCell>
                            <TableCell className="text-right">{m.diff.toFixed(6)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>

                {steeringState.value.top_moves_by_prob && steeringState.value.top_moves_by_prob.length > 0 && (
                  <div className="bg-white border rounded p-3">
                    <h4 className="font-semibold mb-2 text-sm">Top Moves by Prob (Top 10)</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>UCI</TableHead>
                          <TableHead className="text-right">orig_prob</TableHead>
                          <TableHead className="text-right">mod_prob</TableHead>
                          <TableHead className="text-right">Δprob</TableHead>
                          <TableHead className="text-right">Δlogit</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedTopMovesByProb.map((m) => (
                          <TableRow key={`t-${m.uci}`}>
                            <TableCell className="font-mono">{m.uci}</TableCell>
                            <TableCell className="text-right">
                              {typeof m.original_prob === "number" ? m.original_prob.toFixed(6) : "-"}
                            </TableCell>
                            <TableCell className="text-right">
                              {typeof m.modified_prob === "number" ? m.modified_prob.toFixed(6) : "-"}
                            </TableCell>
                            <TableCell className="text-right">
                              {typeof m.prob_diff === "number" ? m.prob_diff.toFixed(6) : "-"}
                            </TableCell>
                            <TableCell className="text-right">{m.diff.toFixed(6)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>
            )}
          </div>

          {!isSinglePosition && selectedFeatureIndex !== null && (
            <Tabs defaultValue="fen-activations" className="w-full">
              <TabsList>
                <TabsTrigger value="fen-activations">Current FEN (64 positions activated)</TabsTrigger>
                <TabsTrigger value="top-activations">Top Activations</TabsTrigger>
              </TabsList>
              <TabsContent value="fen-activations" className="space-y-4">
                {fenActivationState.error && (
                  <div className="text-red-500 font-bold text-center p-4 bg-red-50 rounded">
                    Error:{" "}
                    {fenActivationState.error instanceof Error
                      ? fenActivationState.error.message
                      : String(fenActivationState.error)}
                  </div>
                )}
                {fenActivationState.loading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                      <p className="text-gray-600">Getting 64 positions activated for current FEN...</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-center">
                    <ChessBoard
                      fen={fen}
                      size="medium"
                      showCoordinates={true}
                      activations={fenActivationState.value?.activations}
                      zPatternIndices={fenActivationState.value?.zPatternIndices}
                      zPatternValues={fenActivationState.value?.zPatternValues}
                      analysisName={`Feature #${selectedFeatureIndex}${selectedFeaturePos !== null ? ` | pos ${selectedFeaturePos}` : ""}`}
                      autoFlipWhenBlack={true}
                      flip_activation={fen.includes(" b ")}
                      showSelfPlay={true}
                    />
                  </div>
                )}
              </TabsContent>
              <TabsContent value="top-activations" className="space-y-4">
                {loadingTopActivations ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                      <p className="text-gray-600">Getting Top Activation data...</p>
                    </div>
                  </div>
                ) : topActivations.length > 0 ? (
                  <div>
                    <h4 className="font-semibold mb-4">
                      Feature #{selectedFeatureIndex} Top Activations
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {topActivations.map((sample, index) => (
                        <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                          <div className="text-center mb-2">
                            <div className="text-sm font-medium text-gray-700">Top #{index + 1}</div>
                            <div className="text-xs text-gray-500">
                              Maximum activation value: {sample.activationStrength.toFixed(3)}
                            </div>
                          </div>
                          <ChessBoard
                            fen={sample.fen}
                            size="small"
                            showCoordinates={false}
                            activations={sample.activations}
                            zPatternIndices={sample.zPatternIndices}
                            zPatternValues={sample.zPatternValues}
                            sampleIndex={sample.sampleIndex}
                            analysisName={`Context ${sample.contextId}`}
                            flip_activation={sample.fen.includes(" b ")}
                            autoFlipWhenBlack={true}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No activation samples found with chessboard</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
