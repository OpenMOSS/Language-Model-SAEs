import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChessBoard } from "@/components/chess/chess-board";
import { useAsyncFn } from "react-use";

interface CustomFenInputProps {
  dictionary: string | null;
  featureIndex: number;
  disabled?: boolean;
  className?: string;
}

interface FenAnalysisResult {
  fen: string;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
}

export const CustomFenInput = ({ 
  dictionary, 
  featureIndex, 
  disabled = false,
  className = ""
}: CustomFenInputProps) => {
  const [customFen, setCustomFen] = useState<string>("");
  const [fenAnalysisLoading, setFenAnalysisLoading] = useState<boolean>(false);
  const [fenAnalysisError, setFenAnalysisError] = useState<string | null>(null);
  const [fenAnalysisResult, setFenAnalysisResult] = useState<FenAnalysisResult | null>(null);

  // 当切换 dictionary 或 feature 时，清空自定义 FEN 分析结果
  useEffect(() => {
    setFenAnalysisResult(null);
    setFenAnalysisError(null);
    setCustomFen("");
  }, [dictionary, featureIndex]);

  // 分析自定义FEN的函数
  const analyzeCustomFen = useAsyncFn(async (fen: string, dict: string | null, featIndex: number) => {
    if (!fen || !fen.trim()) {
      setFenAnalysisError("请输入有效的FEN字符串");
      return;
    }
    
    if (!dict) {
      setFenAnalysisError("请先选择一个字典");
      return;
    }

    setFenAnalysisLoading(true);
    setFenAnalysisError(null);
    setFenAnalysisResult(null);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dict}/features/${featIndex}/analyze_fen`,
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

      const data = await response.json();
      
      // 解析返回的数据，构建激活值数组
      let activations: number[] | undefined = undefined;
      if (data.feature_acts_indices && data.feature_acts_values) {
        activations = new Array(64).fill(0);
        const indices = data.feature_acts_indices;
        const values = data.feature_acts_values;
        
        for (let i = 0; i < Math.min(indices.length, values.length); i++) {
          const index = indices[i];
          const value = values[i];
          if (index >= 0 && index < 64) {
            activations[index] = value;
          }
        }
      }

      // 处理 z pattern 数据
      let zPatternIndices: number[][] | undefined = undefined;
      let zPatternValues: number[] | undefined = undefined;
      if (data.z_pattern_indices && data.z_pattern_values) {
        const zpIdxRaw = data.z_pattern_indices;
        zPatternIndices = Array.isArray(zpIdxRaw) && Array.isArray(zpIdxRaw[0]) ? zpIdxRaw : [zpIdxRaw];
        zPatternValues = data.z_pattern_values;
      }

      setFenAnalysisResult({
        fen: fen.trim(),
        activations,
        zPatternIndices,
        zPatternValues,
      });
    } catch (error) {
      setFenAnalysisError(error instanceof Error ? error.message : "分析FEN时出错");
      console.error("分析FEN错误:", error);
    } finally {
      setFenAnalysisLoading(false);
    }
  }, []);

  const handleAnalyze = useCallback(() => {
    if (customFen.trim() && dictionary && !fenAnalysisLoading) {
      analyzeCustomFen[1](customFen.trim(), dictionary, featureIndex);
    }
  }, [customFen, dictionary, featureIndex, fenAnalysisLoading, analyzeCustomFen]);

  return (
    <div className={className}>
      <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-center flex-1">
            自定义FEN分析
          </h3>
        </div>

        {/* 输入区域 */}
        <div className="mb-4 space-y-2">
          <Input
            disabled={disabled || fenAnalysisLoading}
            className="bg-white"
            type="text"
            placeholder="输入FEN字符串，例如: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            value={customFen}
            onChange={(e) => setCustomFen(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleAnalyze();
              }
            }}
          />
          <Button
            disabled={disabled || fenAnalysisLoading || !customFen.trim() || !dictionary}
            onClick={handleAnalyze}
            className="w-full"
          >
            {fenAnalysisLoading ? "分析中..." : "分析FEN"}
          </Button>
        </div>

        {/* 错误显示 */}
        {fenAnalysisError && (
          <div className="text-red-500 font-bold text-center mb-4">
            FEN分析错误: {fenAnalysisError}
          </div>
        )}

        {/* 结果显示 */}
        {fenAnalysisResult && (
          <div className="space-y-2">
            <div className="text-center mb-2">
              <div className="text-sm text-gray-600 mb-2">
                FEN: <code className="bg-gray-100 px-2 py-1 rounded">{fenAnalysisResult.fen}</code>
              </div>
            </div>
            <div className="flex justify-center">
              <ChessBoard
                fen={fenAnalysisResult.fen}
                activations={fenAnalysisResult.activations}
                zPatternIndices={fenAnalysisResult.zPatternIndices}
                zPatternValues={fenAnalysisResult.zPatternValues}
                autoFlipWhenBlack={true}
                flip_activation={fenAnalysisResult.fen.includes(' b ')}
                showSelfPlay={true}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
