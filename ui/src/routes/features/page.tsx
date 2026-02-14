import { AppNavbar } from "@/components/app/navbar";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Combobox } from "@/components/ui/combobox";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount, useDebounce } from "react-use";
import { z } from "zod";

const FeatureCard = lazy(() => import("@/components/feature/feature-card").then(module => ({ default: module.FeatureCard })));
import { ChessBoard } from "@/components/chess/chess-board";
import { CustomFenInput } from "@/components/feature/custom-fen-input";

let boardCounter = 0;


export const FeaturesPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  
  // Use local state for features
  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);
  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [currentFeature, setCurrentFeature] = useState<any>(null);
  const [featureLoading, setFeatureLoading] = useState<boolean>(false);
  const [featureError, setFeatureError] = useState<string | null>(null);

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [analysesState, fetchAnalyses] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/analyses`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [inputValue, setInputValue] = useState<string>("0");
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);

  // Debounce the input value to avoid excessive updates
  useDebounce(
    () => {
      const parsed = parseInt(inputValue);
      if (!isNaN(parsed) && parsed !== featureIndex) {
        setFeatureIndex(parsed);
      }
    },
    300,
    [inputValue]
  );

  const handleFeatureIndexChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  }, []);

  const [featureState, fetchFeature] = useAsyncFn(
    async (
      dictionary: string | null,
      featureIndex: number | string = "random",
      analysisName: string | null = null
    ) => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

       setLoadingRandomFeature(featureIndex === "random");
      setFeatureLoading(true);
      setFeatureError(null);

      try {
        const feature = await fetch(
          `${
            import.meta.env.VITE_BACKEND_URL
          }/dictionaries/${dictionary}/features/${featureIndex}${analysisName ? `?feature_analysis_name=${analysisName}` : ""}`,
          {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
          }
        )
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(await res.text());
          }
          return res;
        })
        .then(async (res) => await res.arrayBuffer())
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .then((res) => decode(new Uint8Array(res)) as any)
        .then((res) =>
          camelcaseKeys(res, {
            deep: true,
              stopPaths: ["sample_groups.samples.context"],
          })
        )
        .then((res) => FeatureSchema.parse(res));
      
      setFeatureIndex(feature.featureIndex);
      setSelectedAnalysis(feature.analysisName);
        setCurrentFeature(feature);
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
        analysis: feature.analysisName,
      });
      return feature;
      } catch (error) {
        setFeatureError(error instanceof Error ? error.message : 'Failed to load feature');
        throw error;
      } finally {
        setFeatureLoading(false);
      }
    }
  );


  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      const dict = searchParams.get("dictionary")!;
      const analysisParam = searchParams.get("analysis");
      setSelectedDictionary(dict);

      fetchAnalyses(dict).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analysisParam || analyses[0]);
        }
      });

      if (searchParams.get("featureIndex")) {
        setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
        fetchFeature(dict, searchParams.get("featureIndex")!, analysisParam || null);
      }
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });

      fetchFeature(dictionariesState.value[0], "random", selectedAnalysis);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      setSelectedAnalysis(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);


  // Memoize dictionary options for Combobox
  const dictionaryOptions = useMemo(() => {
    if (!dictionariesState.value) return [];
    return dictionariesState.value.map((dict) => ({
      value: dict,
      label: dict,
    }));
  }, [dictionariesState.value]);

  // Memoize sections calculation
  const sections = useMemo(() => [
    {
      title: "Histogram",
      id: "Histogram",
    },
    {
      title: "Decoder Norms",
      id: "DecoderNorms",
    },
    {
      title: "Similarity Matrix",
      id: "DecoderSimilarityMatrix",
    },
    {
      title: "Inner Product Matrix",
      id: "DecoderInnerProductMatrix",
    },
    {
      title: "Logits",
      id: "Logits",
    },
    {
      title: "Top Activation",
      id: "Activation",
    },
  ].filter((section) => (currentFeature && currentFeature.logits != null) || section.id !== "Logits"), [currentFeature]);

  // 渲染棋盘示例（已迁移到组件，返回空）
  const renderChessBoardExample = () => {
    return null;
  };


  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Combobox
            disabled={dictionariesState.loading || featureLoading}
            value={selectedDictionary || null}
            onChange={(value) => {
              setSelectedDictionary(value);
            }}
            options={dictionaryOptions}
            placeholder="选择字典..."
            commandPlaceholder="搜索字典..."
            emptyIndicator="未找到匹配的字典"
            className="w-full"
          />
          <Button
            disabled={dictionariesState.loading || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Select analysis:</span>
          <Select
            disabled={analysesState.loading || !selectedDictionary || featureLoading}
            value={selectedAnalysis || undefined}
            onValueChange={setSelectedAnalysis}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select an analysis" />
            </SelectTrigger>
            <SelectContent>
              {analysesState.value?.map((analysis, i) => (
                <SelectItem key={i} value={analysis}>
                  {analysis}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={analysesState.loading || !selectedDictionary || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis);
            }}
          >
            Apply
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Choose a specific feature:</span>
          <Input
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            id="feature-input"
            className="bg-white"
            type="number"
            value={inputValue}
            onChange={handleFeatureIndexChange}
          />
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            onClick={async () => await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis)}
          >
            Go
          </Button>
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Show Random Feature
          </Button>
                    </div>

        {/* 自定义FEN分析区域 */}
        <div className="container w-full max-w-6xl mx-auto mb-8">
          <CustomFenInput
            dictionary={selectedDictionary}
            featureIndex={featureIndex}
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
          />
        </div>

        {featureLoading && !loadingRandomFeature && (
          <div>
            Loading Feature <span className="font-bold">#{featureIndex}</span>...
          </div>
        )}
        {featureLoading && loadingRandomFeature && <div>Loading Random Living Feature...</div>}
        {featureError && <div className="text-red-500 font-bold">Error: {featureError}</div>}
        
            {/* 检查是否包含象棋相关数据 */}
        {!featureLoading && currentFeature && (() => {
          const feature = currentFeature;
          
          console.log('🎯 Feature #' + feature.featureIndex + ' (激活次数: ' + feature.actTimes + ')');
          
          // 检查可能的样本数据字段
          const possibleSampleFields = ['sampleGroups', 'samples', 'sample_groups', 'sample_groups_', 'samples_'];
          let actualSampleGroups: any[] = [];
          
          for (const field of possibleSampleFields) {
            if (feature[field]) {
              if (Array.isArray(feature[field]) && feature[field].length > 0) {
                actualSampleGroups = feature[field];
                break;
              }
            }
          }
          
          // 如果没有找到样本组，尝试直接查找samples字段
          if (!actualSampleGroups.length && feature.samples) {
            actualSampleGroups = [{ samples: feature.samples }];
          }
          
          // 收集所有包含FEN的样本及其信息
          interface ChessSample {
            fen: string;
            sample: any;
            groupIndex: number;
            sampleIndex: number;
            lineIndex: number;
            activationsArray?: number[];
            zPatternIndices?: number[][];
            zPatternValues?: number[];
            maxActivation: number;
            analysisName: string;
            contextId?: number;
            activeColor: string;
          }
          
          const chessSamples: ChessSample[] = [];
          let totalSamples = 0;
          let samplesWithText = 0;
          let samplesWithActivations = 0;
          let debugInfo: string[] = [];
          
          // 遍历所有样本组，收集包含FEN的样本
          for (const [groupIndex, group] of actualSampleGroups.entries()) {
            console.log(`🔍 检查样本组 ${groupIndex}:`, {
              analysisName: group.analysisName,
              samplesCount: group.samples?.length || 0
            });
            
            totalSamples += group.samples.length;
                
            for (const [sampleIndex, sample] of (group.samples || []).entries()) {
              if (sample.text) samplesWithText++;
              
              // 检查是否有激活值数据
              const hasActivations = (sample as any).featureActsIndices && (sample as any).featureActsValues && 
                                    (sample as any).featureActsIndices.length > 0 && (sample as any).featureActsValues.length > 0;
              if (hasActivations) samplesWithActivations++;
              
              // 详细调试信息
              if (sampleIndex < 3) { // 只显示前3个样本的详细信息
                console.log(`🔍 样本 ${sampleIndex} 详细信息:`, {
                  hasText: !!sample.text,
                  textLength: sample.text?.length || 0,
                  textPreview: sample.text?.substring(0, 100) || '无文本',
                  hasFeatureActsIndices: !!(sample as any).featureActsIndices,
                  featureActsIndicesLength: (sample as any).featureActsIndices?.length || 0,
                  hasFeatureActsValues: !!(sample as any).featureActsValues,
                  featureActsValuesLength: (sample as any).featureActsValues?.length || 0,
                  hasZPatternIndices: !!(sample as any).zPatternIndices,
                  zPatternIndicesLength: (sample as any).zPatternIndices?.length || 0,
                  hasZPatternValues: !!(sample as any).zPatternValues,
                  zPatternValuesLength: (sample as any).zPatternValues?.length || 0
                });
              }
                  
              if (sample.text) {
                const lines = sample.text.split('\n');
                
                for (const [lineIndex, line] of lines.entries()) {
                  const trimmed = line.trim();
                  
                  // 检查是否包含FEN格式 - 更宽松的检测
                  // 只要包含8个斜杠分隔的部分，且每部分都是有效的棋子或数字组合
                  if (trimmed.includes('/')) {
                    const parts = trimmed.split(/\s+/);
                    
                    if (parts.length >= 6) {
                      const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
                      const boardRows = boardPart.split('/');
                      
                      if (boardRows.length === 8) {
                        let isValidBoard = true;
                        let totalSquares = 0;
                        
                        // 检查每一行是否符合FEN格式
                        for (const row of boardRows) {
                          if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                            isValidBoard = false;
                            break;
                          }
                          
                          // 计算每行的格子数
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
                        
                        // 验证总格子数为64，且行棋方有效
                        if (isValidBoard && totalSquares === 64 && /^[wb]$/.test(activeColor)) {
                          console.log(`✅ 找到有效的FEN字符串: "${trimmed}"`);
                          console.log(`棋盘行:`, boardRows);
                          console.log(`总格子数:`, totalSquares);
                          
                          // 构建激活值数组并计算最大激活值
                          let activationsArray: number[] | undefined = undefined;
                          let maxActivation = 0;
                          
                          if ((sample as any).featureActsIndices && (sample as any).featureActsValues) {
                            activationsArray = new Array(64).fill(0);
                            const indices = (sample as any).featureActsIndices;
                            const values = (sample as any).featureActsValues;
                            
                            for (let i = 0; i < Math.min(indices.length, values.length); i++) {
                              const index = indices[i];
                              const value = values[i];
                              if (index >= 0 && index < 64) {
                                activationsArray[index] = value;
                                // 计算最大激活值（绝对值）
                                if (Math.abs(value) > Math.abs(maxActivation)) {
                                  maxActivation = value;
                                }
                              }
                            }
                            
                            console.log(`🎯 构建激活值数组: ${activationsArray.filter(v => v !== 0).length} 个非零值, 最大激活值: ${maxActivation}`);
                          }

                          // 构建Z Pattern数组并打印调试信息
                          let zPatternIndices: number[][] | undefined = undefined;
                          let zPatternValues: number[] | undefined = undefined;
                          if ((sample as any).zPatternIndices && (sample as any).zPatternValues) {
                            const zpIdxRaw = (sample as any).zPatternIndices;
                            zPatternIndices = Array.isArray(zpIdxRaw) && Array.isArray(zpIdxRaw[0]) ? zpIdxRaw : [zpIdxRaw];
                            zPatternValues = (sample as any).zPatternValues;

                            try {
                              const len = zPatternValues?.length || 0;
                              console.log(`z_pattern_values shape: (${len},)`);
                              // 打印全部索引和值
                              console.log('z all idx:', zPatternIndices);
                              console.log('z all val:', zPatternValues);
                              // 识别格式并打印全部 pairs
                              const looksLikePairList = Array.isArray(zPatternIndices?.[0]) && (zPatternIndices?.[0] as number[]).length === 2;
                              if (looksLikePairList) {
                                const pairsAll = (zPatternIndices || []).map((p, i) => ([p, (zPatternValues || [])[i]]));
                                console.log('z pairs (all):', pairsAll);
                              } else {
                                const pairs: Array<[number[], number]> = [];
                                for (let t = 0; t < (zPatternIndices || []).length; t++) {
                                  const srcs = zPatternIndices?.[t] || [];
                                  const v = (zPatternValues || [])[t];
                                  for (const s of srcs) {
                                    pairs.push([[s, t], v]);
                                  }
                                }
                                console.log('z pairs (all 展开为[source,target]):', pairs);
                              }
                            } catch (e) {
                              console.warn('打印z_pattern调试信息时出错:', e);
                            }
                          }

                          // 收集样本信息
                          chessSamples.push({
                            fen: trimmed,
                            sample,
                            groupIndex,
                            sampleIndex,
                            lineIndex,
                            activationsArray,
                            zPatternIndices,
                            zPatternValues,
                            maxActivation,
                            analysisName: group.analysisName || '',
                            contextId: (sample as any).context_idx,
                            activeColor
                          });
                          
                          break; // 找到一个有效FEN就跳出行循环
                        } else {
                          debugInfo.push(`FEN验证失败: ${trimmed.substring(0, 30)}...`);
                        }
                      } else {
                        debugInfo.push(`行数错误: ${boardRows.length}行`);
                      }
                    } else {
                      debugInfo.push(`部分不足: ${parts.length}个`);
                    }
                  }
                }
              }
            }
          }
          
          // Sort by maximum activation value from largest to smallest
          chessSamples.sort((a, b) => Math.abs(b.maxActivation) - Math.abs(a.maxActivation));
          
          console.log('Analysis completed: found ' + chessSamples.length + ' FEN samples');
          console.log('Sorted maximum activation values:', chessSamples.slice(0, 10).map(s => s.maxActivation.toFixed(3)));
          
          // Generate chessboard components
          const chessBoards: JSX.Element[] = chessSamples.map((chessSample, index) => (
            <ChessBoard
              key={`chess-${chessSample.groupIndex}-${chessSample.sampleIndex}-${chessSample.lineIndex}-${index}`}
              fen={chessSample.fen}
              activations={chessSample.activationsArray}
              zPatternIndices={chessSample.zPatternIndices}
              zPatternValues={chessSample.zPatternValues}
              sampleIndex={chessSample.sampleIndex}
              analysisName={chessSample.analysisName}
              contextId={chessSample.contextId}
              autoFlipWhenBlack={true}
              flip_activation={Boolean(chessSample.activeColor === 'b')}
              showSelfPlay={true}
            />
          ));
          
          const validFENFound = chessSamples.length;
          
          if (chessBoards.length > 0) {
            console.log(`Generated ${chessBoards.length} chessboards`);
            
            return (
              <div className="w-full max-w-6xl mx-auto mb-8">
                <div className="text-center mb-6">
                  <h3 className="text-xl font-bold mb-2">Feature #{feature.featureIndex} Chessboard Visualization</h3>
                  <div className="text-sm text-gray-600">
                    Found {validFENFound} samples containing FEN (sorted by maximum activation value)
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {chessBoards}
                </div>
              </div>
            );
          } else {
            // If no chessboard is found, display debug information
            return (
              <div className="w-full max-w-6xl mx-auto mb-8">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-yellow-800 mb-4">⚠️ No chessboard visualization found</h3>
                  <div className="text-sm text-yellow-700 mb-4">
                    <strong>Feature #{feature.featureIndex}</strong> (Activation times: {feature.actTimes})
                  </div>
                  
                  <div className="text-xs text-yellow-600 mb-4">
                    <strong>Data statistics:</strong><br/>
                    • Total samples: {totalSamples}<br/>
                    • Found FEN: {validFENFound}
                  </div>
                  
                  <div className="text-xs text-yellow-500 mt-4">
                    💡 Possible reasons:<br/>
                    1. Sample does not contain FEN formatted text<br/>
                    2. Sample data structure does not match expectations
                  </div>
                </div>
              </div>
            );
          }
        })()}
            
        {!featureLoading && currentFeature && (
            <div className="flex gap-12 w-full">
              <Suspense fallback={<div>Loading Feature Card...</div>}>
              <FeatureCard feature={currentFeature} />
              </Suspense>
              <SectionNavigator sections={sections} />
          </div>
        )}
      </div>
    </div>
  );
};
