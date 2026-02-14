import { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAsyncFn } from "react-use";
import { ChessBoard } from "@/components/chess/chess-board";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2 } from "lucide-react";
import { Link } from "react-router-dom";

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

interface GetFeatureFromFenProps {
  fen: string;
  layer: number;
  position: number;
  componentType: "attn" | "mlp";
  modelName?: string;
  saeComboId?: string;
  onFeatureAction?: (action: {
    type: "add_to_source" | "set_as_target" | "add_to_steer" | "view_details";
    featureIndex: number;
    activationValue: number;
  }) => void;
  actionTypes?: Array<"add_to_source" | "set_as_target" | "add_to_steer">; // 指定要显示的操作按钮，默认显示所有
  actionButtonLabels?: {
    add_to_source?: string;
    set_as_target?: string;
    add_to_steer?: string;
  }; // 自定义按钮文本
  showTopActivations?: boolean;
  showFenActivations?: boolean;
  /** When false, render only inner content (no Card wrapper). Use when parent already provides the card and title. */
  wrapInCard?: boolean;
  className?: string;
}

export const GetFeatureFromFen = ({
  fen,
  layer,
  position,
  componentType,
  modelName = "lc0/BT4-1024x15x32h",
  saeComboId,
  onFeatureAction,
  actionTypes = ["add_to_source", "set_as_target", "add_to_steer"], // 默认显示所有按钮
  actionButtonLabels = {}, // 默认使用内置文本
  showTopActivations = true,
  showFenActivations = true,
  wrapInCard = true,
  className,
}: GetFeatureFromFenProps) => {
  // 默认按钮文本
  const defaultLabels = {
    add_to_source: "加入Source",
    set_as_target: "设为Target",
    add_to_steer: "加入Steer",
  };
  
  // 合并自定义标签和默认标签
  const buttonLabels = {
    add_to_source: actionButtonLabels?.add_to_source || defaultLabels.add_to_source,
    set_as_target: actionButtonLabels?.set_as_target || defaultLabels.set_as_target,
    add_to_steer: actionButtonLabels?.add_to_steer || defaultLabels.add_to_steer,
  };
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<number | null>(null);
  const [topActivations, setTopActivations] = useState<TopActivationSample[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);
  // 存储每个 feature 的 top activation（只存储第一个用于预览）
  const [featureTopActivations, setFeatureTopActivations] = useState<Map<number, TopActivationSample | null>>(new Map());
  const [loadingFeatureTopActivations, setLoadingFeatureTopActivations] = useState<Set<number>>(new Set());

  // 获取激活的features
  const [featuresState, fetchFeatures] = useAsyncFn(async () => {
    if (!fen || position < 0 || position > 63) {
      return null;
    }

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
          pos: position,
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
      // 按激活值降序排序
      return [...features].sort((a, b) => b.activation_value - a.activation_value);
    }

    return [];
  }, [fen, layer, position, componentType, modelName, saeComboId]);

  // 构建字典名
  const getDictionaryName = useCallback(() => {
    const lorsaSuffix = componentType === "attn" ? "A" : "M";
    const baseDict = `BT4_${componentType === "attn" ? "lorsa" : "tc"}_L${layer}${lorsaSuffix}`;
    if (saeComboId && saeComboId !== "k_128_e_128") {
      const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, "k$1_e$2");
      return `${baseDict}_${comboParts}`;
    }
    return baseDict;
  }, [componentType, layer, saeComboId]);

  // 解析analyze_fen响应
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

  // 获取当前FEN下feature的64格激活值
  const [fenActivationState, fetchFenActivationForSelectedFeature] = useAsyncFn(async () => {
    if (!fen?.trim()) {
      throw new Error("FEN 不能为空");
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

  // 辅助函数：解析 top activation 数据
  const parseTopActivationData = useCallback((camelData: any): TopActivationSample[] => {
    const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
    const allSamples: any[] = [];

    for (const group of sampleGroups) {
      if (group.samples && Array.isArray(group.samples)) {
        allSamples.push(...group.samples);
      }
    }

    // 查找包含FEN的样本并提取激活值
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
                // 验证FEN格式
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
                  // 处理稀疏激活数据
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

    // 按最大激活值排序
    return chessSamples.sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength));
  }, []);

  // 获取指定feature的top activation（用于选中后的详细显示）
  const fetchTopActivationsForFeature = useCallback(
    async (featureIndex: number) => {
      setLoadingTopActivations(true);
      try {
        const dictionary = getDictionaryName();

        const response = await fetch(
          `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
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

        const chessSamples = parseTopActivationData(camelData);
        const topSamples = chessSamples.slice(0, 8);
        setTopActivations(topSamples);
      } catch (error) {
        console.error("❌ 获取Top Activation数据失败:", error);
        setTopActivations([]);
      } finally {
        setLoadingTopActivations(false);
      }
    },
    [getDictionaryName, parseTopActivationData]
  );

  // 获取单个 feature 的第一个 top activation（用于表格预览）
  const fetchFirstTopActivationForFeature = useCallback(
    async (featureIndex: number) => {
      // 如果已经有缓存，跳过
      if (featureTopActivations.has(featureIndex)) {
        return;
      }

      // 如果正在加载，跳过
      if (loadingFeatureTopActivations.has(featureIndex)) {
        return;
      }

      setLoadingFeatureTopActivations((prev) => new Set(prev).add(featureIndex));

      try {
        const dictionary = getDictionaryName();

        const response = await fetch(
          `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
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

        const chessSamples = parseTopActivationData(camelData);
        const firstTopActivation = chessSamples.length > 0 ? chessSamples[0] : null;

        setFeatureTopActivations((prev) => {
          const newMap = new Map(prev);
          newMap.set(featureIndex, firstTopActivation);
          return newMap;
        });
      } catch (error) {
        console.error(`❌ 获取 Feature #${featureIndex} 的 Top Activation 失败:`, error);
        setFeatureTopActivations((prev) => {
          const newMap = new Map(prev);
          newMap.set(featureIndex, null);
          return newMap;
        });
      } finally {
        setLoadingFeatureTopActivations((prev) => {
          const newSet = new Set(prev);
          newSet.delete(featureIndex);
          return newSet;
        });
      }
    },
    [getDictionaryName, parseTopActivationData, featureTopActivations, loadingFeatureTopActivations]
  );

  // 当选择的feature改变时，获取top activation和FEN激活值
  useEffect(() => {
    if (selectedFeatureIndex !== null) {
      if (showTopActivations) {
        fetchTopActivationsForFeature(selectedFeatureIndex);
      }
      if (showFenActivations) {
        fetchFenActivationForSelectedFeature();
      }
    } else {
      setTopActivations([]);
    }
  }, [selectedFeatureIndex, showTopActivations, showFenActivations, fetchTopActivationsForFeature, fetchFenActivationForSelectedFeature]);

  const features = featuresState.value || [];

  const content = (
    <div className="space-y-4">
          {/* 刷新按钮 */}
          <Button
            onClick={() => fetchFeatures()}
            disabled={featuresState.loading || !fen || position < 0 || position > 63}
            className="w-full"
          >
            {featuresState.loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                加载中...
              </>
            ) : (
              "获取激活的 Features"
            )}
          </Button>

          {/* 错误显示 */}
          {featuresState.error && (
            <div className="text-red-500 font-bold text-center p-4 bg-red-50 rounded">
              错误: {featuresState.error instanceof Error ? featuresState.error.message : String(featuresState.error)}
            </div>
          )}

          {/* Features列表 */}
          {features.length > 0 && (
            <div className="space-y-4">
              <h3 className="font-semibold">
                激活的 Features ({features.length} 个)
              </h3>
              <div className="max-h-96 overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Feature Index</TableHead>
                      <TableHead>Activation Value</TableHead>
                      <TableHead>Top Activation</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {features.slice(0, 50).map((feature, index) => (
                      <TableRow
                        key={feature.feature_index}
                        className={`cursor-pointer ${
                          selectedFeatureIndex === feature.feature_index
                            ? "bg-blue-50 border-blue-500"
                            : "hover:bg-gray-50"
                        }`}
                        onClick={() => {
                          const next =
                            selectedFeatureIndex === feature.feature_index ? null : feature.feature_index;
                          setSelectedFeatureIndex(next);
                        }}
                      >
                        <TableCell>{index + 1}</TableCell>
                        <TableCell className="font-mono">#{feature.feature_index}</TableCell>
                        <TableCell>
                          <span
                            className={
                              feature.activation_value > 0 ? "text-green-600" : "text-red-600"
                            }
                          >
                            {feature.activation_value > 0 ? "+" : ""}
                            {feature.activation_value.toFixed(6)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Link
                            to={`/features?dictionary=${encodeURIComponent(getDictionaryName())}&featureIndex=${feature.feature_index}`}
                            onClick={(e) => e.stopPropagation()}
                            className="text-blue-600 hover:text-blue-800 underline text-sm"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            查看详情
                          </Link>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {onFeatureAction && (
                              <>
                                {actionTypes.includes("add_to_source") && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onFeatureAction({
                                        type: "add_to_source",
                                        featureIndex: feature.feature_index,
                                        activationValue: feature.activation_value,
                                      });
                                    }}
                                  >
                                    {buttonLabels.add_to_source}
                                  </Button>
                                )}
                                {actionTypes.includes("set_as_target") && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onFeatureAction({
                                        type: "set_as_target",
                                        featureIndex: feature.feature_index,
                                        activationValue: feature.activation_value,
                                      });
                                    }}
                                  >
                                    {buttonLabels.set_as_target}
                                  </Button>
                                )}
                                {actionTypes.includes("add_to_steer") && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onFeatureAction({
                                        type: "add_to_steer",
                                        featureIndex: feature.feature_index,
                                        activationValue: feature.activation_value,
                                      });
                                    }}
                                  >
                                    {buttonLabels.add_to_steer}
                                  </Button>
                                )}
                              </>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {features.length > 50 && (
                  <p className="text-sm text-gray-500 mt-2">
                    显示前50个features，总共{features.length}个
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Top Activations 预览区域 */}
          {features.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">
                  Top Activations 预览
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    // 批量加载所有features的top activation（约30个）
                    for (const feature of features) {
                      if (!featureTopActivations.has(feature.feature_index) && 
                          !loadingFeatureTopActivations.has(feature.feature_index)) {
                        await fetchFirstTopActivationForFeature(feature.feature_index);
                        // 添加小延迟避免同时发送太多请求
                        await new Promise(resolve => setTimeout(resolve, 100));
                      }
                    }
                  }}
                  disabled={loadingFeatureTopActivations.size > 0}
                >
                  {loadingFeatureTopActivations.size > 0 ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      加载中...
                    </>
                  ) : (
                    `加载所有Features的Top Activation (${features.length}个)`
                  )}
                </Button>
              </div>
              <div className="w-full max-w-full min-w-0 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                {features.map((feature) => (
                  <div
                    key={feature.feature_index}
                    className="min-w-0 border rounded-lg p-2 bg-gray-50 hover:bg-gray-100 transition-colors flex flex-col items-center"
                  >
                    <div className="text-center mb-1 w-full">
                      <div className="text-sm font-medium text-gray-700">
                        Feature #{feature.feature_index}
                      </div>
                      <div className="text-xs text-gray-600 mt-0.5">
                        <span
                          className={
                            feature.activation_value > 0 ? "text-green-600" : "text-red-600"
                          }
                        >
                          {feature.activation_value > 0 ? "+" : ""}
                          {feature.activation_value.toFixed(4)}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-center mb-1 w-full max-w-full overflow-hidden">
                      {loadingFeatureTopActivations.has(feature.feature_index) ? (
                        <div className="flex flex-col items-center justify-center py-4">
                          <Loader2 className="w-6 h-6 animate-spin text-gray-400 mb-2" />
                          <span className="text-xs text-gray-400">加载中...</span>
                        </div>
                      ) : featureTopActivations.has(feature.feature_index) ? (
                        (() => {
                          const topActivation = featureTopActivations.get(feature.feature_index);
                          if (topActivation && topActivation.fen) {
                            return (
                              <Link
                                to={`/features?dictionary=${encodeURIComponent(getDictionaryName())}&featureIndex=${feature.feature_index}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="cursor-pointer block max-w-full"
                                title="点击在Feature页面查看完整详情"
                              >
                                <div className="max-w-full overflow-hidden flex justify-center" style={{ maxWidth: "100%" }}>
                                  <div className="origin-center shrink-0" style={{ transform: "scale(0.65)" }}>
                                    <ChessBoard
                                      fen={topActivation.fen}
                                      size="small"
                                      showCoordinates={true}
                                      activations={topActivation.activations}
                                      zPatternIndices={topActivation.zPatternIndices}
                                      zPatternValues={topActivation.zPatternValues}
                                      flip_activation={topActivation.fen.includes(" b ")}
                                      autoFlipWhenBlack={true}
                                    />
                                  </div>
                                </div>
                              </Link>
                            );
                          } else {
                            return (
                              <div className="text-center py-4">
                                <span className="text-xs text-gray-400">无数据</span>
                              </div>
                            );
                          }
                        })()
                      ) : (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            fetchFirstTopActivationForFeature(feature.feature_index);
                          }}
                          className="text-xs"
                        >
                          加载
                        </Button>
                      )}
                    </div>
                    <div className="text-center">
                      <Link
                        to={`/features?dictionary=${encodeURIComponent(getDictionaryName())}&featureIndex=${feature.feature_index}`}
                        onClick={(e) => e.stopPropagation()}
                        className="text-blue-600 hover:text-blue-800 underline text-xs"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        查看详情
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-sm text-gray-500 text-center">
                显示所有 {features.length} 个features的Top Activation预览，点击棋盘可在Feature页面查看完整详情
              </p>
            </div>
          )}

          {/* 选中的Feature详情 - Top Activations和FEN激活值 */}
          {selectedFeatureIndex !== null && (showTopActivations || showFenActivations) && (
            <Tabs defaultValue="fen-activations" className="w-full">
              <TabsList>
                {showFenActivations && <TabsTrigger value="fen-activations">当前FEN（64格激活）</TabsTrigger>}
                {showTopActivations && <TabsTrigger value="top-activations">Top Activations</TabsTrigger>}
              </TabsList>
              
              {showFenActivations && (
                <TabsContent value="fen-activations" className="space-y-4">
                  {fenActivationState.error && (
                    <div className="text-red-500 font-bold text-center p-4 bg-red-50 rounded">
                      错误:{" "}
                      {fenActivationState.error instanceof Error
                        ? fenActivationState.error.message
                        : String(fenActivationState.error)}
                    </div>
                  )}
                  {fenActivationState.loading ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">正在获取当前FEN的64格激活...</p>
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
                        analysisName={`Feature #${selectedFeatureIndex} | pos ${position}`}
                        autoFlipWhenBlack={true}
                        flip_activation={fen.includes(" b ")}
                        showSelfPlay={true}
                      />
                    </div>
                  )}
                </TabsContent>
              )}

              {showTopActivations && (
                <TabsContent value="top-activations" className="space-y-4">
                  {loadingTopActivations ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
                        <p className="text-gray-600">正在获取Top Activation数据...</p>
                      </div>
                    </div>
                  ) : topActivations.length > 0 ? (
                    <div>
                      <h4 className="font-semibold mb-4">
                        Feature #{selectedFeatureIndex} 的Top Activations
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {topActivations.map((sample, index) => (
                          <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                            <div className="text-center mb-2">
                              <div className="text-sm font-medium text-gray-700">Top #{index + 1}</div>
                              <div className="text-xs text-gray-500">
                                最大激活值: {sample.activationStrength.toFixed(3)}
                              </div>
                            </div>
                            <div className="max-w-full overflow-hidden flex justify-center">
                              <div className="origin-center shrink-0" style={{ transform: "scale(0.75)" }}>
                                <ChessBoard
                                  fen={sample.fen}
                                  size="small"
                                  showCoordinates={true}
                                  activations={sample.activations}
                                  zPatternIndices={sample.zPatternIndices}
                                  zPatternValues={sample.zPatternValues}
                                  sampleIndex={sample.sampleIndex}
                                  analysisName={`Context ${sample.contextId}`}
                                  flip_activation={sample.fen.includes(" b ")}
                                  autoFlipWhenBlack={true}
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <p>未找到包含棋盘的激活样本</p>
                    </div>
                  )}
                </TabsContent>
              )}
            </Tabs>
          )}
        </div>
  );

  if (!wrapInCard) {
    return <div className={className}>{content}</div>;
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          位置 Feature 分析
          <div className="text-sm font-normal mt-2 text-gray-600">
            FEN: <code className="bg-gray-100 px-2 py-1 rounded text-xs">{fen}</code>
            <br />
            层: {layer} | 位置: {position} | 组件: {componentType === "attn" ? "Attention (LoRSA)" : "MLP (Transcoder)"}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {content}
      </CardContent>
    </Card>
  );
};
