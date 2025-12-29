import { useState, useEffect, useCallback } from "react";
import { AppNavbar } from "@/components/app/navbar";
import { PosFeatureCard } from "@/components/feature/pos-feature-card";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { ChessBoard } from "@/components/chess/chess-board";

const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";

export const PositionFeaturePage = () => {
  const [fen, setFen] = useState<string>("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const [layer, setLayer] = useState<number>(0);
  const [positionsInput, setPositionsInput] = useState<string>("0");
  const [positions, setPositions] = useState<number[]>([0]);
  const [componentType, setComponentType] = useState<"attn" | "mlp">("attn");
  const [saeComboId, setSaeComboId] = useState<string | undefined>(undefined);
  const [selectedFeature, setSelectedFeature] = useState<{ featureIndex: number; position: number } | null>(null);
  const [allPosActivation, setAllPosActivation] = useState<number[] | undefined>(undefined);
  const [allPosZPatternIndices, setAllPosZPatternIndices] = useState<number[][] | undefined>(undefined);
  const [allPosZPatternValues, setAllPosZPatternValues] = useState<number[] | undefined>(undefined);
  const [loadingAllPos, setLoadingAllPos] = useState(false);
  const [allPosError, setAllPosError] = useState<string | null>(null);

  // 从 localStorage 读取 sae_combo_id
  useEffect(() => {
    const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
    if (stored) {
      setSaeComboId(stored);
    }
  }, []);

  // 监听 localStorage 变化（当 SaeComboLoader 更新时）
  useEffect(() => {
    const handleStorageChange = () => {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        setSaeComboId(stored);
      } else {
        setSaeComboId(undefined);
      }
    };

    window.addEventListener("storage", handleStorageChange);
    // 也监听同页面的更新（通过自定义事件）
    const interval = setInterval(() => {
      const stored = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored !== saeComboId) {
        setSaeComboId(stored || undefined);
      }
    }, 500);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(interval);
    };
  }, [saeComboId]);

  // 解析位置输入（支持逗号分隔的多个位置）
  const handlePositionsChange = (value: string) => {
    setPositionsInput(value);
    const parsed = value
      .split(",")
      .map((s) => parseInt(s.trim()))
      .filter((n) => !isNaN(n) && n >= 0 && n < 64);
    setPositions(parsed.length > 0 ? parsed : [0]);
  };

  const getDictionaryName = useCallback((): string => {
    const suffix = componentType === "attn" ? "A" : "M";
    const baseDict = `BT4_${componentType === "attn" ? "lorsa" : "tc"}_L${layer}${suffix}`;
    if (saeComboId && saeComboId !== "k_128_e_128") {
      const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, "k$1_e$2");
      return `${baseDict}_${comboParts}`;
    }
    return baseDict;
  }, [componentType, layer, saeComboId]);

  const fetchAllPositionsForFeature = useCallback(
    async (featureIndex: number) => {
      setLoadingAllPos(true);
      setAllPosError(null);
      setAllPosActivation(undefined);
      setAllPosZPatternIndices(undefined);
      setAllPosZPatternValues(undefined);
      try {
        const dictionary = getDictionaryName();
        const response = await fetch(
          `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
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

        const data = (await response.json()) as {
          positions?: Array<{
            position: number;
            // 后端历史上返回过两种口径：
            // - `activations`: number[64]（只在 activations[position] 非零）
            // - 或者直接给标量（某些版本/接口可能这样）
            activations: number[] | number;
            z_pattern_indices?: number[][] | null;
            z_pattern_values?: number[] | null;
          }>;
        };

        // 关键修正：
        // “全位置激活”不应跨 position 做 max-abs merge。
        // 正确做法是：对每个 position 取该 position 的标量激活值，并填回 64 格。
        const mergedActivations = new Array(64).fill(0);

        const positionsArr = data.positions && Array.isArray(data.positions) ? data.positions : [];
        for (const posData of positionsArr) {
          const pos = posData.position;
          if (pos < 0 || pos >= 64) continue;

          if (Array.isArray(posData.activations)) {
            // 借鉴 FeaturesPage 的逻辑：把返回的 64 格数组里“非零项”直接填回对应格子
            // （不依赖 pos 字段做任何聚合/平均；更鲁棒）
            if (posData.activations.length === 64) {
              for (let i = 0; i < 64; i++) {
                const v = posData.activations[i] ?? 0;
                if (v !== 0) {
                  if (mergedActivations[i] !== 0 && mergedActivations[i] !== v) {
                    // 理论上同一个格子不应从不同 posData 得到不同值；若发生，提示但不做 max/avg
                    console.warn("conflicting activation value for square", { i, prev: mergedActivations[i], next: v });
                  }
                  mergedActivations[i] = v;
                }
              }
            } else {
              // 兜底：如果返回不是 64 长，就取第一个元素当标量回填到 position
              const scalar = posData.activations[0] ?? 0;
              mergedActivations[pos] = scalar;
            }
          } else if (typeof posData.activations === "number") {
            // 标量口径：直接填回 position
            mergedActivations[pos] = posData.activations;
          }
        }

        // Z-Pattern：更合理的展示是“选中 position”对应的 query->key 模式，
        // 而不是跨 position 合并（合并会混淆多个 query 位置）。
        const selectedPos = selectedFeature?.position;
        const selectedPosData =
          selectedPos !== undefined ? positionsArr.find((p) => p.position === selectedPos) : undefined;

        setAllPosActivation(mergedActivations);
        setAllPosZPatternIndices(
          selectedPosData?.z_pattern_indices && Array.isArray(selectedPosData.z_pattern_indices)
            ? selectedPosData.z_pattern_indices
            : undefined
        );
        setAllPosZPatternValues(
          selectedPosData?.z_pattern_values && Array.isArray(selectedPosData.z_pattern_values)
            ? selectedPosData.z_pattern_values
            : undefined
        );
      } catch (e) {
        setAllPosError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoadingAllPos(false);
      }
    },
    [fen, getDictionaryName, selectedFeature?.position]
  );

  useEffect(() => {
    if (selectedFeature?.featureIndex !== undefined && fen.trim()) {
      fetchAllPositionsForFeature(selectedFeature.featureIndex);
    } else {
      setAllPosActivation(undefined);
      setAllPosZPatternIndices(undefined);
      setAllPosZPatternValues(undefined);
      setAllPosError(null);
      setLoadingAllPos(false);
    }
  }, [selectedFeature, fen, fetchAllPositionsForFeature]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">位置 Feature 分析</h1>
          <p className="text-gray-600 mt-2">
            分析特定棋盘位置的激活特征，按位置分组显示每个位置激活的 features。
          </p>
        </div>

        {/* SAE Combo Loader */}
        <div className="mb-6">
          <SaeComboLoader />
        </div>

        {/* Position Feature Analysis Section */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>位置 Feature 分析配置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div className="space-y-2">
                <Label htmlFor="fen-input">FEN 字符串</Label>
                <Input
                  id="fen-input"
                  value={fen}
                  onChange={(e) => setFen(e.target.value)}
                  placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="layer-input">层 (Layer)</Label>
                <Input
                  id="layer-input"
                  type="number"
                  min="0"
                  max="14"
                  value={layer}
                  onChange={(e) => setLayer(parseInt(e.target.value) || 0)}
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="positions-input">位置 (0-63，逗号分隔)</Label>
                <Input
                  id="positions-input"
                  value={positionsInput}
                  onChange={(e) => handlePositionsChange(e.target.value)}
                  placeholder="0,1,2"
                  className="bg-white"
                />
                <p className="text-xs text-gray-500">
                  当前选择: {positions.join(", ")}
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="component-type">组件类型</Label>
                <Select
                  value={componentType}
                  onValueChange={(value: "attn" | "mlp") => setComponentType(value)}
                >
                  <SelectTrigger id="component-type" className="bg-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="attn">Attention</SelectItem>
                    <SelectItem value="mlp">MLP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>棋盘</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center">
              <ChessBoard
                fen={fen}
                size="medium"
                showCoordinates={true}
                flip_activation={fen.includes(" b ")}
                autoFlipWhenBlack={true}
                showSelfPlay={true}
              />
            </div>
          </CardContent>
        </Card>

        {selectedFeature && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>
                选中 Feature 全位置激活与 Z-Pattern
                <div className="text-sm font-normal mt-2 text-gray-600">
                  Feature #{selectedFeature.featureIndex} | 来自位置 {selectedFeature.position} | 字典{" "}
                  <code className="bg-gray-100 px-2 py-1 rounded text-xs">{getDictionaryName()}</code>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {allPosError && <div className="text-red-600 font-medium mb-3">{allPosError}</div>}
              {loadingAllPos ? (
                <div className="text-gray-600">加载中...</div>
              ) : (
                <div className="flex justify-center">
                  <ChessBoard
                    fen={fen}
                    size="medium"
                    showCoordinates={true}
                    activations={allPosActivation}
                    zPatternIndices={allPosZPatternIndices}
                    zPatternValues={allPosZPatternValues}
                    flip_activation={fen.includes(" b ")}
                    autoFlipWhenBlack={true}
                    analysisName={`Feature #${selectedFeature.featureIndex}`}
                    showSelfPlay={true}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* PosFeatureCard Component */}
        {fen && positions.length > 0 && (
          <div className="mb-6">
            <PosFeatureCard
              fen={fen}
              layer={layer}
              positions={positions}
              componentType={componentType}
              modelName="lc0/BT4-1024x15x32h"
              saeComboId={saeComboId}
              onFeatureSelect={setSelectedFeature}
            />
          </div>
        )}
      </div>
    </div>
  );
};