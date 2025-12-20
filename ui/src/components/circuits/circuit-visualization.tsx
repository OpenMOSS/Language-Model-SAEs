import { useState, useCallback, useEffect, useMemo, Suspense, lazy } from "react";
import { Link } from "react-router-dom";
import { useCircuitState } from "@/contexts/AppStateContext";
import { LinkGraphContainer } from "./link-graph-container";
import { NodeConnections } from "./node-connections";
import { transformCircuitData, CircuitJsonData } from "./link-graph/utils";
import { Node } from "./link-graph/types";
import { Feature } from "@/types/feature";
import { FeatureCard } from "@/components/feature/feature-card";
import { ChessBoard } from "@/components/chess/chess-board";
import React from "react";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
import { useModelLoadingStatus } from "@/components/shared/model-loading-status";
// 使用 React.lazy 动态导入，避免初始化错误导致整个应用崩溃
const CircuitInterpretation = lazy(() => {
  console.log('[CircuitVisualization] Lazy loading CircuitInterpretation...');
  return import("./circuit-interpretation").then(module => {
    console.log('[CircuitVisualization] CircuitInterpretation loaded successfully');
    return { default: module.CircuitInterpretation };
  }).catch(error => {
    console.error('[CircuitVisualization] Failed to load CircuitInterpretation:', error);
    // 返回一个错误占位组件
    return {
      default: () => (
        <div className="fixed inset-0 bg-red-100 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-4 rounded shadow-lg">
            <p className="text-red-600">Failed to load Circuit Interpretation component</p>
            <pre className="text-xs mt-2 overflow-auto max-h-40">{error.message}</pre>
          </div>
        </div>
      )
    };
  });
});

// 定义节点激活数据的类型
interface NodeActivationData {
  activations?: number[];
  zPatternIndices?: any;
  zPatternValues?: number[];
  nodeType?: string;
  clerp?: string;
}

export const CircuitVisualization = () => {
  const {
    circuitData: linkGraphData,
    isLoading,
    error,
    clickedId,
    hoveredId,
    pinnedIds,
    hiddenIds,
    setCircuitData: setLinkGraphData,
    setLoading,
    setError,
    setClickedId,
    setHoveredId,
    setPinnedIds,
    setHiddenIds,
  } = useCircuitState();

  const { isLoaded: isSaeLoaded } = useModelLoadingStatus();

  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [originalCircuitJson, setOriginalCircuitJson] = useState<any>(null); // 存储原始JSON数据（单图或合并后的）
  const [editingClerp, setEditingClerp] = useState<string>(''); // 当前编辑的clerp
  const [isSaving, setIsSaving] = useState(false); // 保存状态
  const [originalFileName, setOriginalFileName] = useState<string>(''); // 原始文件名（单文件时）
  const [updateCounter, setUpdateCounter] = useState(0); // 用于强制更新的计数器
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // 是否有未保存的更改
  const [saveHistory, setSaveHistory] = useState<string[]>([]); // 保存历史记录
  const [topActivations, setTopActivations] = useState<any[]>([]); // Top Activation 数据
  const [loadingTopActivations, setLoadingTopActivations] = useState(false); // 加载状态
  const [tokenPredictions, setTokenPredictions] = useState<any>(null); // Token Predictions 数据
  const [loadingTokenPredictions, setLoadingTokenPredictions] = useState(false); // 加载状态
  const [steeringScale, setSteeringScale] = useState<number>(0); // steering 放大系数
  const [steeringScaleInput, setSteeringScaleInput] = useState<string>('0'); // 文本输入，用于支持暂存 "-"
  const [denseNodes, setDenseNodes] = useState<Set<string>>(new Set()); // Dense节点集合
  const [denseThreshold, setDenseThreshold] = useState<string>(''); // Dense阈值（空字符串表示无限大）
  const [checkingDenseFeatures, setCheckingDenseFeatures] = useState(false); // 是否正在检查dense features
  const [syncingToBackend, setSyncingToBackend] = useState(false); // 是否正在同步到后端
  const [syncingFromBackend, setSyncingFromBackend] = useState(false); // 是否正在从后端同步
  
  // Graph Feature Diffing 相关状态
  const [perturbedFen, setPerturbedFen] = useState<string>(''); // Perturbed FEN输入
  const [isComparingFens, setIsComparingFens] = useState(false); // 是否正在比较
  const [inactiveNodes, setInactiveNodes] = useState<Set<string>>(new Set()); // 未激活节点集合
  const [diffingLogs, setDiffingLogs] = useState<Array<{timestamp: number; message: string}>>([]); // 比较日志
  const [showDiffingLogs, setShowDiffingLogs] = useState(false); // 是否显示日志

  // 子图功能相关状态
  const [showSubgraph, setShowSubgraph] = useState(false); // 是否显示子图模式
  const [subgraphData, setSubgraphData] = useState<any>(null); // 子图数据
  const [subgraphRootNodeId, setSubgraphRootNodeId] = useState<string | null>(null); // 子图根节点ID

  // Feature 激活显示模式：单个位置 vs 所有位置
  const [showAllPositions, setShowAllPositions] = useState(false); // 是否显示所有位置的激活
  const [allPositionsActivationData, setAllPositionsActivationData] = useState<NodeActivationData | null>(null); // 所有位置的合并激活数据

  // 多图支持：存放多份原始 JSON 及其文件名
  const [multiOriginalJsons, setMultiOriginalJsons] = useState<{ json: CircuitJsonData; fileName: string }[]>([]);

  // Circuit 标注相关状态
  const [showCircuitInterpretation, setShowCircuitInterpretation] = useState(false);
  const [selectedNodeForCircuit, setSelectedNodeForCircuit] = useState<{
    nodeId: string;
    layer: number;
    feature: number;
    feature_type: string;
  } | null>(null);

  // 为"各自独有"的节点/边分配的颜色表（最多4个图）
  const UNIQUE_GRAPH_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B"]; // 蓝、橙、绿、红

  // 将多个图的 JSON 合并为一个 LinkGraphData（节点按 node_id 合并，边按(source,target)合并）
  const mergeGraphs = useCallback((jsons: CircuitJsonData[], fileNames?: string[]) => {
    // 先将每个 JSON 转换为 LinkGraphData
    const graphs = jsons.map(j => transformCircuitData(j));

    // 合并 metadata（简单策略：拼接 prompt_tokens 并标注来源数量）
    const mergedMetadata: any = {
      ...(graphs[0]?.metadata || {}),
      prompt_tokens: graphs.map((g, i) => `[#${i + 1}] ` + (g?.metadata?.prompt_tokens?.join(' ') || '')).filter(Boolean),
      sourceFileNames: fileNames && fileNames.length ? fileNames : undefined,
    };

    // 合并节点
    type NodeAccum = {
      base: any; // 任意一个来源的节点作为基准（保留 feature_type 等）
      presentIn: number[]; // 出现于哪些图的索引
    };

    const nodeMap = new Map<string, NodeAccum>();

    graphs.forEach((g, gi) => {
      g.nodes.forEach((n: any) => {
        const key = n.nodeId;
        if (!nodeMap.has(key)) {
          nodeMap.set(key, { base: { ...n }, presentIn: [gi] });
        } else {
          const acc = nodeMap.get(key)!;
          // 合并可选字段（以非空为准）
          acc.base.localClerp = acc.base.localClerp ?? n.localClerp;
          acc.base.remoteClerp = acc.base.remoteClerp ?? n.remoteClerp;
          // 累加来源
          if (!acc.presentIn.includes(gi)) acc.presentIn.push(gi);
        }
      });
    });

    // 为节点设置颜色：
    // - 若 presentIn.length > 1（多个图共有）：使用 transformCircuitData 原有的 feature_type 颜色（acc.base.nodeColor）
    // - 若仅在某个单图中：覆盖为 UNIQUE_GRAPH_COLORS[graphIndex]
    let mergedNodes: any[] = [];
    nodeMap.forEach(({ base, presentIn }) => {
      const isShared = presentIn.length > 1;
      const isError = typeof base.feature_type === 'string' && base.feature_type.toLowerCase().includes('error');
      const nodeColor = isError
        ? '#95a5a6'
        : (isShared ? base.nodeColor : UNIQUE_GRAPH_COLORS[presentIn[0] % UNIQUE_GRAPH_COLORS.length]);
      const sourceIndices = presentIn.slice();
      const sourceFiles = (fileNames && fileNames.length)
        ? sourceIndices.map(i => fileNames[i]).filter(Boolean)
        : undefined;
      mergedNodes.push({
        ...base,
        nodeColor,
        sourceIndices,
        sourceIndex: sourceIndices.length === 1 ? sourceIndices[0] : undefined,
        sourceFiles,
      });
    });

    // 移除备用颜色逻辑：多文件场景仅保留
    // - 共有节点：使用各自类型颜色（来自 transformCircuitData）
    // - 独有节点：使用该文件唯一颜色 UNIQUE_GRAPH_COLORS[index]

    // 合并边：以(source,target)为键；
    // - 若多图共有：保留 transform 中的颜色（红/绿取决于权重正负）和 strokeWidth（取最大）并将权重求和或取平均
    // - 若单图有：保留并将颜色覆盖为对应图的 UNIQUE_GRAPH_COLORS[gi] 的淡化版以区分（但维持正负色彩会更直观，这里沿用正负色，不改变现有绿色/红色方案）
    type LinkAccum = {
      sources: number[]; // 出现于哪些图
      weightSum: number;
      maxStroke: number;
      color: string; // 采用首次的权重颜色（正负）即可
      pctInputSum: number;
      weightsBySource: Record<number, number>;
      pctBySource: Record<number, number>;
    };

    const linkKey = (s: string, t: string) => `${s}__${t}`;
    const linkMap = new Map<string, LinkAccum>();

    graphs.forEach((g, gi) => {
      (g.links || []).forEach((e: any) => {
        const k = linkKey(e.source, e.target);
        if (!linkMap.has(k)) {
          linkMap.set(k, {
            sources: [gi],
            weightSum: e.weight ?? 0,
            maxStroke: e.strokeWidth ?? 1,
            color: e.color,
            pctInputSum: e.pctInput ?? 0,
            weightsBySource: { [gi]: e.weight ?? 0 },
            pctBySource: { [gi]: e.pctInput ?? (Math.abs(e.weight ?? 0) * 100) },
          });
        } else {
          const acc = linkMap.get(k)!;
          if (!acc.sources.includes(gi)) acc.sources.push(gi);
          acc.weightSum += (e.weight ?? 0);
          acc.maxStroke = Math.max(acc.maxStroke, e.strokeWidth ?? 1);
          // 颜色按首次的正负即可，不覆盖
          acc.pctInputSum += (e.pctInput ?? 0);
          acc.weightsBySource[gi] = (acc.weightsBySource[gi] || 0) + (e.weight ?? 0);
          acc.pctBySource[gi] = (acc.pctBySource[gi] || 0) + (e.pctInput ?? (Math.abs(e.weight ?? 0) * 100));
        }
      });
    });

    const mergedLinks: any[] = [];
    linkMap.forEach((acc, k) => {
      const [source, target] = k.split("__");
      const isShared = acc.sources.length > 1;
      const avgWeight = acc.weightSum / acc.sources.length;
      const avgPct = acc.pctInputSum / acc.sources.length;
      // 沿用 transform 的正负配色：正=绿色，负=红色
      const color = avgWeight > 0 ? "#4CAF50" : "#F44336";
      mergedLinks.push({
        source,
        target,
        pathStr: "",
        color: isShared ? color : color,
        strokeWidth: acc.maxStroke,
        weight: avgWeight,
        pctInput: avgPct,
        sources: acc.sources,
        weightsBySource: acc.weightsBySource,
        pctBySource: acc.pctBySource,
      });
    });

    // 重新为节点填充 sourceLinks/targetLinks
    const nodeById: Record<string, any> = {};
    mergedNodes.forEach(n => { nodeById[n.nodeId] = { ...n, sourceLinks: [], targetLinks: [] }; });
    mergedLinks.forEach(l => {
      if (nodeById[l.source]) nodeById[l.source].sourceLinks.push(l);
      if (nodeById[l.target]) nodeById[l.target].targetLinks.push(l);
    });

    const finalNodes = Object.values(nodeById);

    const mergedData: any = {
      nodes: finalNodes,
      links: mergedLinks,
      metadata: mergedMetadata,
    };

    return mergedData;
  }, []);

  const handleFeatureClick = useCallback((node: Node, isMetaKey: boolean) => {
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedIds.includes(node.nodeId)
        ? pinnedIds.filter(id => id !== node.nodeId)
        : [...pinnedIds, node.nodeId];
      setPinnedIds(newPinnedIds);
    } else {
      // Set clicked node
      const newClickedId = node.nodeId === clickedId ? null : node.nodeId;
      setClickedId(newClickedId);
      
      // 注释掉自动退出子图模式的逻辑，让用户在子图中可以自由点击其他节点
      // 用户可以通过"退出子图"按钮或"显示子图"按钮手动控制子图模式
      // if (newClickedId !== clickedId && showSubgraph) {
      //   setShowSubgraph(false);
      //   setSubgraphData(null);
      //   setSubgraphRootNodeId(null);
      //   console.log('🔄 切换节点，自动退出子图模式');
      // }
    }
  }, [clickedId, pinnedIds, setClickedId, setPinnedIds]);

  const handleFeatureHover = useCallback((nodeId: string | null) => {
    // Only update if the hovered ID has actually changed
    if (nodeId !== hoveredId) {
      setHoveredId(nodeId);
    }
  }, [hoveredId, setHoveredId]);

  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
  }, []);

  const handleConnectedFeaturesLoading = useCallback((_loading: boolean) => {
    // 保留回调函数以保持接口兼容性
  }, []);

  // 检查analysis_name是否在已知组合中
  const checkAnalysisNames = useCallback((metadata: any): { isValid: boolean; warnings: string[] } => {
    const knownAnalysisNames = [
      "BT4_tc_k30_e16",
      "BT4_lorsa_k30_e16",
      "BT4_tc_k64_e32",
      "BT4_lorsa_k64_e32",
      "BT4_tc_k128_e64",
      "BT4_lorsa_k128_e64",
      "BT4_tc_k256_e128",
      "BT4_lorsa_k256_e128",
      "BT4_tc",  // k128_e128 (默认组合，无后缀)
      "BT4_lorsa",  // k128_e128 (默认组合，无后缀)
    ];
    
    const warnings: string[] = [];
    const lorsaAnalysisName = metadata?.lorsa_analysis_name;
    const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
    
    if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
      if (!knownAnalysisNames.includes(lorsaAnalysisName)) {
        warnings.push(`⚠️ LoRSA analysis_name "${lorsaAnalysisName}" 不在已知组合中，将使用默认组合 k128_e128 (BT4_lorsa)`);
      }
    }
    
    if (tcAnalysisName && typeof tcAnalysisName === 'string') {
      if (!knownAnalysisNames.includes(tcAnalysisName)) {
        warnings.push(`⚠️ TC analysis_name "${tcAnalysisName}" 不在已知组合中，将使用默认组合 k128_e128 (BT4_tc)`);
      }
    }
    
    return {
      isValid: warnings.length === 0,
      warnings
    };
  }, []);

  // 单文件上传（保留，兼容）
  const handleSingleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('Please upload a JSON file');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const text = await file.text();
      const jsonData: CircuitJsonData = JSON.parse(text);
      
      // 检查analysis_name是否在已知组合中
      const metadata = jsonData?.metadata || {};
      const { isValid, warnings } = checkAnalysisNames(metadata);
      
      if (!isValid && warnings.length > 0) {
        // 显示警告，但不阻止加载
        const warningMessage = warnings.join('\n') + '\n\n将使用默认组合 k128_e128 进行 feature 分析。';
        console.warn('⚠️ Circuit文件analysis_name检查:', warnings);
        alert(warningMessage);
      }
      
      // 基础变换
      const data = transformCircuitData(jsonData);
      // 注入来源信息（单文件索引为 0）
      let annotated = {
        ...data,
        nodes: data.nodes.map(n => ({
          ...n,
          sourceIndex: 0,
          sourceIndices: [0],
          sourceFiles: [file.name],
        })),
        metadata: {
          ...data.metadata,
          sourceFileNames: [file.name],
        }
      } as any;

      setLinkGraphData(annotated);
      // Reset circuit state when loading new data
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      setOriginalCircuitJson(jsonData); // 存储原始JSON数据
      setOriginalFileName(file.name); // 存储原始文件名
      setEditingClerp(''); // 重置编辑状态
      setHasUnsavedChanges(false); // 清除未保存的更改
      setSaveHistory([]); // 清除保存历史
      setMultiOriginalJsons([{ json: jsonData, fileName: file.name }]);
      // 清空Graph Feature Diffing相关状态
      setPerturbedFen('');
      setInactiveNodes(new Set());
      setDiffingLogs([]);
      setShowDiffingLogs(false);
      // 清空子图相关状态
      setShowSubgraph(false);
      setSubgraphData(null);
      setSubgraphRootNodeId(null);
      // 重置激活显示模式
      setShowAllPositions(false);
      setAllPositionsActivationData(null);
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures, checkAnalysisNames]);

  // 多文件上传（1-4 个）
  const handleMultiFilesUpload = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files).filter(f => f.name.endsWith('.json')).slice(0, 4);
    if (list.length === 0) {
      setError('Please upload 1-4 JSON files');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const texts = await Promise.all(list.map(f => f.text()));
      const jsons: CircuitJsonData[] = texts.map(t => JSON.parse(t));
      const fileNames = list.map(f => f.name);

      // 检查所有文件的analysis_name
      const allWarnings: string[] = [];
      jsons.forEach((json, index) => {
        const metadata = json?.metadata || {};
        const { warnings } = checkAnalysisNames(metadata);
        if (warnings.length > 0) {
          warnings.forEach(w => {
            allWarnings.push(`[文件 ${fileNames[index]}]: ${w}`);
          });
        }
      });
      
      if (allWarnings.length > 0) {
        const warningMessage = allWarnings.join('\n') + '\n\n将使用默认组合 k128_e128 进行 feature 分析。';
        console.warn('⚠️ Circuit文件analysis_name检查:', allWarnings);
        alert(warningMessage);
      }

      // 合并
      const merged = jsons.length === 1 
        ? (() => {
            const data = transformCircuitData(jsons[0]);
            return {
              ...data,
              nodes: data.nodes.map(n => ({
                ...n,
                sourceIndex: 0,
                sourceIndices: [0],
                sourceFiles: [fileNames[0]],
              })),
              metadata: {
                ...data.metadata,
                sourceFileNames: [fileNames[0]],
              }
            };
          })()
        : mergeGraphs(jsons, fileNames);

      setLinkGraphData(merged);
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      setOriginalCircuitJson(jsons.length === 1 ? jsons[0] : merged); // 单图保留原始，多图保留合并结果
      setOriginalFileName(list.length === 1 ? list[0].name : `merged_${list.length}_graphs.json`);
      setEditingClerp('');
      setHasUnsavedChanges(false);
      setSaveHistory([]);
      setMultiOriginalJsons(list.map((f, i) => ({ json: jsons[i], fileName: f.name })));
      // 清空Graph Feature Diffing相关状态
      setPerturbedFen('');
      setInactiveNodes(new Set());
      setDiffingLogs([]);
      setShowDiffingLogs(false);
      // 清空子图相关状态
      setShowSubgraph(false);
      setSubgraphData(null);
      setSubgraphRootNodeId(null);
      // 重置激活显示模式
      setShowAllPositions(false);
      setAllPositionsActivationData(null);
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [mergeGraphs, setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures, checkAnalysisNames]);


  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  // 从circuit数据中提取FEN字符串
  const extractFenFromPrompt = useCallback(() => {
    if (!linkGraphData?.metadata?.prompt_tokens) return null;
    
    const promptText = linkGraphData.metadata.prompt_tokens.join(' ');
    console.log('🔍 搜索FEN字符串:', promptText);
    
    // 更宽松的FEN格式检测
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      // 检查是否包含FEN格式 - 包含斜杠且有足够的字符
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const [boardPart, activeColor] = parts;
          const boardRows = boardPart.split('/');
          
          if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
            console.log('✅ 找到FEN字符串:', trimmed);
            return trimmed;
          }
        }
      }
    }
    
    // 如果没找到完整的FEN，尝试更简单的匹配
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    if (simpleMatch) {
      console.log('✅ 找到简单FEN匹配:', simpleMatch[0]);
      return simpleMatch[0];
    }
    
    console.log('❌ 未找到FEN字符串');
    return null;
  }, [linkGraphData]);

  // 按文件从原始 JSON 提取 FEN
  const extractFenFromCircuitJson = useCallback((json: any): string | null => {
    const tokens = json?.metadata?.prompt_tokens;
    if (!tokens) return null;
    const promptText = Array.isArray(tokens) ? tokens.join(' ') : String(tokens);
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const boardRows = parts[0].split('/');
          if (boardRows.length === 8 && /^[wb]$/.test(parts[1])) return trimmed;
        }
      }
    }
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    return simpleMatch ? simpleMatch[0] : null;
  }, []);
 
  // 从prompt中提取输出移动
  const extractOutputMove = useCallback(() => {
    if (!linkGraphData) return null;

    // 1) 优先从 metadata 中读取 target_move 或 logit_moves[0]
    const tm = (linkGraphData as any)?.metadata?.target_move;
    if (typeof tm === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(tm)) {
      return tm.toLowerCase();
    }
    const lm0 = (linkGraphData as any)?.metadata?.logit_moves?.[0];
    if (typeof lm0 === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(lm0)) {
      return lm0.toLowerCase();
    }

    // 2) 回退到从 prompt_tokens 中解析
    if (!linkGraphData?.metadata?.prompt_tokens) return null;
    const promptText = linkGraphData.metadata.prompt_tokens.join(' ');

    const movePatterns = [
      /(?:Output|Move|下一步|移动)[:：]\s*([a-h][1-8][a-h][1-8])/i,
      /\b([a-h][1-8][a-h][1-8])\b/g
    ];

    for (const pattern of movePatterns) {
      const matches = promptText.match(pattern);
      if (matches) {
        const lastMatch = Array.isArray(matches) ? matches[matches.length - 1] : matches;
        const moveMatch = lastMatch.match(/[a-h][1-8][a-h][1-8]/);
        if (moveMatch) {
          return moveMatch[0].toLowerCase();
        }
      }
    }

    return null;
  }, [linkGraphData]);

  // 按文件提取输出移动
  const extractOutputMoveFromCircuitJson = useCallback((json: any): string | null => {
    if (!json) return null;

    // 1) 优先从 metadata 中读取 target_move 或 logit_moves[0]
    const tm = json?.metadata?.target_move;
    if (typeof tm === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(tm)) {
      return tm.toLowerCase();
    }
    const lm0 = json?.metadata?.logit_moves?.[0];
    if (typeof lm0 === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(lm0)) {
      return lm0.toLowerCase();
    }

    // 2) 回退到从 prompt_tokens 中解析
    const tokens = json?.metadata?.prompt_tokens;
    if (!tokens) return null;
    const promptText = Array.isArray(tokens) ? tokens.join(' ') : String(tokens);
    const patterns = [
      /(?:Output|Move|下一步|移动)[:：]\s*([a-h][1-8][a-h][1-8])/i,
      /\b([a-h][1-8][a-h][1-8])\b/g
    ];
    for (const pattern of patterns) {
      const matches = promptText.match(pattern);
      if (matches) {
        const lastMatch = Array.isArray(matches) ? matches[matches.length - 1] : matches;
        const moveMatch = lastMatch.match(/[a-h][1-8][a-h][1-8]/);
        if (moveMatch) return moveMatch[0].toLowerCase();
      }
    }
    return null;
  }, []);
 
  // 改进的getNodeActivationData函数
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    if (!nodeId || !originalCircuitJson) {
      console.log('❌ 缺少必要参数:', { nodeId, hasOriginalCircuitJson: !!originalCircuitJson });
      return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    }
    
    console.log(`🔍 查找节点 ${nodeId} 的激活数据...`);
    console.log('📋 原始JSON数据结构:', {
      type: typeof originalCircuitJson,
      isArray: Array.isArray(originalCircuitJson),
      hasNodes: !!originalCircuitJson.nodes,
      nodesLength: originalCircuitJson.nodes?.length || 0,
      keys: Object.keys(originalCircuitJson)
    });
    
    // 解析 node_id -> rawLayer, featureOrHead, ctx(position)
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const ctxIdx = Number(parts[2]) || 0;
      // 将原始层号除以2得到真实层号
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead, ctxIdx };
    };
    const parsed = parseFromNodeId(nodeId);

    // 1) 优先在 nodes 数组中做直接匹配（若节点对象内联了 activations/zPattern* 字段）
    let nodesToSearch: any[] = [];
    if (originalCircuitJson.nodes && Array.isArray(originalCircuitJson.nodes)) {
      nodesToSearch = originalCircuitJson.nodes;
    } else if (Array.isArray(originalCircuitJson)) {
      nodesToSearch = originalCircuitJson;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray((originalCircuitJson as any)[key])) {
          nodesToSearch = (originalCircuitJson as any)[key];
          break;
        }
      }
      if (nodesToSearch.length === 0) {
        const values = Object.values(originalCircuitJson);
        const arrayValue = values.find(v => Array.isArray(v)) as any[] | undefined;
        if (arrayValue) nodesToSearch = arrayValue;
      }
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find(node => node?.node_id === nodeId);
      if (exactMatch) {
        const inlineActs = exactMatch.activations;
        const inlineZIdx = exactMatch.zPatternIndices;
        const inlineZVal = exactMatch.zPatternValues;
        console.log('✅ 节点内联字段检查:', {
          hasInlineActivations: !!inlineActs,
          hasInlineZIdx: !!inlineZIdx,
          hasInlineZVal: !!inlineZVal,
        });
        if (inlineActs || (inlineZIdx && inlineZVal)) {
          return {
            activations: inlineActs,
            zPatternIndices: inlineZIdx,
            zPatternValues: inlineZVal,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // 2) 基于 node_id 解析，在 activation 记录集合中匹配（layer/position/head_idx/feature_idx）
    // 构建可扫描的记录集合（深度扫描 originalCircuitJson 的所有数组，挑出含关键键的条目）
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) {
              candidateRecords.push(item);
            }
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(originalCircuitJson);

    console.log('🧭 候选记录数:', candidateRecords.length);

    // 定义匹配函数：根据 feature_type 选择使用 head_idx 或 feature_idx
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === parsed.ctxIdx;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && posOk && indexOk;
    };

    // 为了确定 feature_type，先尽量从 nodes 中取该 nodeId 的类型
    let featureTypeForNode: string | undefined = undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find(n => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find(rec => tryMatchRecord(rec, featureTypeForNode));
    if (matched) {
      console.log('✅ 通过解析匹配到activation记录:', {
        nodeId,
        layerForActivation: parsed.layerForActivation,
        ctxIdx: parsed.ctxIdx,
        featureOrHead: parsed.featureOrHead,
        featureTypeForNode,
      });
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
      };
    }

    // 3) 回退：尝试模糊匹配 node_id 前缀
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter(node => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        console.log('🔍 使用模糊匹配节点:', {
          node_id: firstMatch.node_id,
          hasActivations: !!firstMatch.activations,
        });
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    console.log('❌ 未找到任何匹配的节点/记录');
    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, [originalCircuitJson, updateCounter]);

  const getNodeActivationDataFromJson = useCallback((jsonData: any, nodeId: string | null): NodeActivationData => {
    if (!nodeId || !jsonData) return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    const parts = nodeId.split('_');
    const rawLayer = Number(parts[0]) || 0;
    const featureOrHead = Number(parts[1]) || 0;
    const ctxIdx = Number(parts[2]) || 0;
    const layerForActivation = Math.floor(rawLayer / 2);

    // 1) 优先在 nodes 数组中做直接匹配
    let nodesToSearch: any[] = [];
    if (jsonData.nodes && Array.isArray(jsonData.nodes)) {
      nodesToSearch = jsonData.nodes;
    } else if (Array.isArray(jsonData)) {
      nodesToSearch = jsonData;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray(jsonData[key])) {
          nodesToSearch = jsonData[key];
          break;
        }
      }
      if (nodesToSearch.length === 0) {
        const values = Object.values(jsonData);
        const arrayValue = values.find(v => Array.isArray(v)) as any[] | undefined;
        if (arrayValue) nodesToSearch = arrayValue;
      }
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find((node: any) => node?.node_id === nodeId);
      if (exactMatch) {
        if (exactMatch.activations || (exactMatch.zPatternIndices && exactMatch.zPatternValues)) {
          return {
            activations: exactMatch.activations,
            zPatternIndices: exactMatch.zPatternIndices,
            zPatternValues: exactMatch.zPatternValues,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // 2) 深度扫描激活记录集合，匹配(layer/position/index)
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) candidateRecords.push(item);
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(jsonData);

    // 从 nodes 中尽量确定 feature_type
    let featureTypeForNode: string | undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find((n: any) => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find((rec: any) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;
      const layerOk = !Number.isNaN(recLayer) && recLayer === layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === ctxIdx;
      let indexOk = false;
      if (featureTypeForNode) {
        const t = featureTypeForNode.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === featureOrHead;
        else indexOk = (recHead === featureOrHead) || (recFeatIdx === featureOrHead);
      } else {
        indexOk = (recHead === featureOrHead) || (recFeatIdx === featureOrHead);
      }
      return layerOk && posOk && indexOk;
    });

    if (matched) {
      const clerp = (nodesToSearch.find((n: any) => n?.node_id === nodeId) || {}).clerp;
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp,
      };
    }

    // 3) 模糊匹配
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter((node: any) => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, []);
 
  // 辅助函数：获取SAE名称模板（不带层号）
  const getSaeNameTemplate = useCallback((layerIdx: number, isLorsa: boolean): string => {
    if (isLorsa) {
      const lorsaAnalysisName = linkGraphData?.metadata?.lorsa_analysis_name;
      if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
        // 如果analysis_name是 "BT4_lorsa"，返回模板 "BT4_lorsa_L{}A"
        if (lorsaAnalysisName === "BT4_lorsa") {
          return "BT4_lorsa_L{}A";
        } else {
          // 其他组合：BT4_lorsa_k256_e128 -> BT4_lorsa_L{}A_k256_e128
          const suffix = lorsaAnalysisName.replace("BT4_lorsa_", "");
          return `BT4_lorsa_L{}A_${suffix}`;
        }
      }
      return "BT4_lorsa_L{}A"; // 默认
    } else {
      const tcAnalysisName = (linkGraphData?.metadata as any)?.tc_analysis_name || linkGraphData?.metadata?.clt_analysis_name;
      if (tcAnalysisName && typeof tcAnalysisName === 'string') {
        if (tcAnalysisName === "BT4_tc") {
          return "BT4_tc_L{}M";
        } else {
          const suffix = tcAnalysisName.replace("BT4_tc_", "");
          return `BT4_tc_L{}M_${suffix}`;
        }
      }
      return "BT4_tc_L{}M"; // 默认
    }
  }, [linkGraphData?.metadata]);

  // 为 CircuitInterpretation 组件创建 getSaeName 函数
  const getSaeNameForCircuit = useCallback((layer: number, isLorsa: boolean) => {
    console.log('[CircuitVisualization] getSaeNameForCircuit called:', { layer, isLorsa });
    try {
      console.log('[CircuitVisualization] Calling getSaeNameTemplate...');
      const template = getSaeNameTemplate(layer, isLorsa);
      console.log('[CircuitVisualization] Template result:', template);
      const result = template.replace('{}', layer.toString());
      console.log('[CircuitVisualization] Final SAE name:', result);
      return result;
    } catch (error) {
      console.error('[CircuitVisualization] Error in getSaeNameForCircuit:', error);
      // 返回默认值
      const fallback = isLorsa ? `BT4_lorsa_L${layer}A` : `BT4_tc_L${layer}M`;
      console.log('[CircuitVisualization] Using fallback:', fallback);
      return fallback;
    }
  }, [getSaeNameTemplate]);

  // 辅助函数：根据metadata中的analysis_name（不带层号）确定字典名（带层号）
  const getDictionaryName = useCallback((layerIdx: number, isLorsa: boolean): string => {
    // 已知的BT4 SAE组合analysis_name列表（来自server/constants.py，不带层号）
    const knownAnalysisNames = [
      "BT4_tc_k30_e16",
      "BT4_lorsa_k30_e16",
      "BT4_tc_k64_e32",
      "BT4_lorsa_k64_e32",
      "BT4_tc_k128_e64",
      "BT4_lorsa_k128_e64",
      "BT4_tc_k256_e128",
      "BT4_lorsa_k256_e128",
      "BT4_tc",  // k128_e128 (默认组合，无后缀)
      "BT4_lorsa",  // k128_e128 (默认组合，无后缀)
    ];
    
    let dictionary: string;
    let usingDefault = false;
    
    if (isLorsa) {
      const lorsaAnalysisName = linkGraphData?.metadata?.lorsa_analysis_name;
      if (lorsaAnalysisName && typeof lorsaAnalysisName === 'string') {
        // 检查analysis_name是否在已知列表中
        const isKnown = knownAnalysisNames.includes(lorsaAnalysisName);
        
        if (isKnown) {
          // 根据analysis_name构建完整的字典名（加上层号）
          if (lorsaAnalysisName === "BT4_lorsa") {
            // 默认组合：BT4_lorsa -> BT4_lorsa_L{layer}A
            dictionary = `BT4_lorsa_L${layerIdx}A`;
          } else {
            // 其他组合：BT4_lorsa_k256_e128 -> BT4_lorsa_L{layer}A_k256_e128
            // 从 "BT4_lorsa_k256_e128" 提取后缀 "k256_e128"
            const suffix = lorsaAnalysisName.replace("BT4_lorsa_", "");
            dictionary = `BT4_lorsa_L${layerIdx}A_${suffix}`;
          }
        } else {
          console.warn(`⚠️ LoRSA analysis_name "${lorsaAnalysisName}" 不在已知组合中，使用默认组合 k128_e128`);
          dictionary = `BT4_lorsa_L${layerIdx}A`;
          usingDefault = true;
        }
      } else {
        dictionary = `BT4_lorsa_L${layerIdx}A`;
        usingDefault = true;
      }
    } else {
      const tcAnalysisName = (linkGraphData?.metadata as any)?.tc_analysis_name || linkGraphData?.metadata?.clt_analysis_name;
      if (tcAnalysisName && typeof tcAnalysisName === 'string') {
        const isKnown = knownAnalysisNames.includes(tcAnalysisName);
        
        if (isKnown) {
          // 根据analysis_name构建完整的字典名（加上层号）
          if (tcAnalysisName === "BT4_tc") {
            // 默认组合：BT4_tc -> BT4_tc_L{layer}M
            dictionary = `BT4_tc_L${layerIdx}M`;
          } else {
            // 其他组合：BT4_tc_k256_e128 -> BT4_tc_L{layer}M_k256_e128
            // 从 "BT4_tc_k256_e128" 提取后缀 "k256_e128"
            const suffix = tcAnalysisName.replace("BT4_tc_", "");
            dictionary = `BT4_tc_L${layerIdx}M_${suffix}`;
          }
        } else {
          console.warn(`⚠️ TC analysis_name "${tcAnalysisName}" 不在已知组合中，使用默认组合 k128_e128`);
          dictionary = `BT4_tc_L${layerIdx}M`;
          usingDefault = true;
        }
      } else {
        dictionary = `BT4_tc_L${layerIdx}M`;
        usingDefault = true;
      }
    }
    
    if (usingDefault) {
      console.warn(`⚠️ 使用默认组合 k128_e128 的字典名: ${dictionary}`);
    }
    
    return dictionary;
  }, [linkGraphData]);

  // 获取该 feature 在所有位置的激活数据
  const getAllPositionsActivationData = useCallback((nodeId: string | null, jsonData?: any): NodeActivationData | null => {
    const dataToSearch = jsonData || originalCircuitJson;
    if (!nodeId || !dataToSearch) {
      return null;
    }

    // 解析 node_id -> rawLayer, featureOrHead
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead };
    };
    const parsed = parseFromNodeId(nodeId);

    // 获取节点类型
    let nodesToSearch: any[] = [];
    if (dataToSearch.nodes && Array.isArray(dataToSearch.nodes)) {
      nodesToSearch = dataToSearch.nodes;
    } else if (Array.isArray(dataToSearch)) {
      nodesToSearch = dataToSearch;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray((dataToSearch as any)[key])) {
          nodesToSearch = (dataToSearch as any)[key];
          break;
        }
      }
    }

    let featureTypeForNode: string | undefined = undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find(n => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    // 构建可扫描的记录集合
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) {
              candidateRecords.push(item);
            }
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(dataToSearch);

    // 匹配函数：匹配相同的 layer 和 feature_idx/head_idx，但忽略 position
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && indexOk;
    };

    // 找到所有匹配的记录（所有位置）
    const matchedRecords = candidateRecords.filter(rec => tryMatchRecord(rec, featureTypeForNode));

    if (matchedRecords.length === 0) {
      console.log('❌ 未找到任何匹配的激活记录');
      return null;
    }

    console.log(`✅ 找到 ${matchedRecords.length} 个位置的激活记录`);

    // 合并所有位置的激活值
    // 策略：对于每个棋盘位置，取所有位置中该位置的最大激活值（绝对值）
    const mergedActivations = new Array(64).fill(0);
    const mergedZPatternIndices: number[][] = [];
    const mergedZPatternValues: number[] = [];
    const zPatternMap = new Map<string, number>(); // 用于合并 zPattern 值

    for (const rec of matchedRecords) {
      if (rec.activations && Array.isArray(rec.activations) && rec.activations.length === 64) {
        // 合并激活值：取最大值（绝对值）
        for (let i = 0; i < 64; i++) {
          const currentValue = mergedActivations[i];
          const newValue = rec.activations[i];
          if (Math.abs(newValue) > Math.abs(currentValue)) {
            mergedActivations[i] = newValue;
          }
        }
      }

      // 合并 zPattern 数据
      if (rec.zPatternIndices && rec.zPatternValues && 
          Array.isArray(rec.zPatternIndices) && Array.isArray(rec.zPatternValues)) {
        for (let i = 0; i < rec.zPatternIndices.length; i++) {
          const indices = rec.zPatternIndices[i];
          const value = rec.zPatternValues[i];
          if (Array.isArray(indices) && indices.length === 2) {
            const key = `${indices[0]}_${indices[1]}`;
            const existingValue = zPatternMap.get(key);
            if (existingValue === undefined || Math.abs(value) > Math.abs(existingValue)) {
              zPatternMap.set(key, value);
            }
          }
        }
      }
    }

    // 将 zPatternMap 转换回数组格式
    zPatternMap.forEach((value, key) => {
      const [qPos, kPos] = key.split('_').map(Number);
      mergedZPatternIndices.push([qPos, kPos]);
      mergedZPatternValues.push(value);
    });

    return {
      activations: mergedActivations,
      zPatternIndices: mergedZPatternIndices.length > 0 ? mergedZPatternIndices : undefined,
      zPatternValues: mergedZPatternValues.length > 0 ? mergedZPatternValues : undefined,
      nodeType: featureTypeForNode,
      clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
    };
  }, [originalCircuitJson]);

  // 提取相关数据
  const fen = extractFenFromPrompt();
  const outputMove = extractOutputMove();
  const nodeActivationData = getNodeActivationData(clickedId);
  
  // 当切换节点时，重置显示模式为单个位置
  useEffect(() => {
    setShowAllPositions(false);
    setAllPositionsActivationData(null);
  }, [clickedId]);

  // 当点击节点或切换模式时，更新所有位置的激活数据
  useEffect(() => {
    if (clickedId && showAllPositions) {
      const allPosData = getAllPositionsActivationData(clickedId);
      setAllPositionsActivationData(allPosData);
    } else {
      setAllPositionsActivationData(null);
    }
  }, [clickedId, showAllPositions, getAllPositionsActivationData]);

  // 确定要显示的激活数据
  const displayActivationData = showAllPositions && allPositionsActivationData 
    ? allPositionsActivationData 
    : nodeActivationData;

  // 修复Hook使用 - 移到组件顶层，避免条件调用
  useEffect(() => {
    if (clickedId && nodeActivationData) {
      // 无论clerp是undefined、空字符串还是有内容，都设置到编辑器中
      const clerpValue = nodeActivationData.clerp || '';
      console.log('🔄 更新编辑器状态:', {
        nodeId: clickedId,
        clerpValue,
        clerpType: typeof nodeActivationData.clerp,
        clerpLength: clerpValue.length,
        updateCounter
      });
      setEditingClerp(clerpValue);
    } else {
      // 没有选中节点时，清空编辑器
      console.log('🔄 清空编辑器状态');
      setEditingClerp('');
      // 重置所有位置显示模式
      setShowAllPositions(false);
      setAllPositionsActivationData(null);
    }
  }, [clickedId, nodeActivationData?.clerp, updateCounter]);

  const handleSaveClerp = useCallback(async () => {
    console.log('🚀 开始保存clerp:', {
      clickedId,
      hasOriginalCircuitJson: !!originalCircuitJson,
      editingClerp,
      editingClerpLength: editingClerp.length,
      trimmedLength: editingClerp.trim().length
    });
    
    if (!clickedId || !originalCircuitJson) {
      console.log('❌ 保存失败：缺少必要数据');
      return;
    }

    // 允许保存空内容，但至少要有一些变化
    const trimmedClerp = editingClerp.trim();
    
    setIsSaving(true);
    
    try {
      // 先创建深拷贝，避免直接修改原始数据
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // 查找并更新对应的节点
      let updated = false;
      let nodesToSearch: any[] = [];
      
      if (updatedCircuitJson.nodes && Array.isArray(updatedCircuitJson.nodes)) {
        nodesToSearch = updatedCircuitJson.nodes;
      } else if (Array.isArray(updatedCircuitJson)) {
        nodesToSearch = updatedCircuitJson;
      } else {
        const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
        for (const key of possibleArrayKeys) {
          if (Array.isArray(updatedCircuitJson[key])) {
            nodesToSearch = updatedCircuitJson[key];
            break;
          }
        }
        
        if (nodesToSearch.length === 0) {
          const values = Object.values(updatedCircuitJson);
          const arrayValue = values.find(v => Array.isArray(v));
          if (arrayValue) {
            nodesToSearch = arrayValue as any[];
          }
        }
      }

      // 直接通过node_id匹配并更新节点的clerp
      for (const node of nodesToSearch) {
        if (node && typeof node === 'object' && node.node_id === clickedId) {
          // 设置clerp字段，无论之前是否存在
          const previousClerp = node.clerp;
          node.clerp = trimmedClerp;
          updated = true;
          console.log('✅ 已更新节点clerp:', {
            node_id: clickedId,
            feature: node.feature,
            layer: node.layer,
            feature_type: node.feature_type,
            previousClerp: previousClerp || '(空)',
            newClerp: trimmedClerp || '(空)',
            newClerpLength: trimmedClerp.length
          });
          break;
        }
      }

      if (updated) {
        // 更新状态为修改后的深拷贝
        setOriginalCircuitJson(updatedCircuitJson);
        
        // 强制触发重新获取节点数据
        setUpdateCounter(prev => prev + 1);
        
        // 标记为有未保存的更改
        setHasUnsavedChanges(true);
        
        console.log('✅ 本地数据已更新，触发重新渲染');
        console.log('🔍 验证更新:', {
          nodeId: clickedId,
          updatedClerp: updatedCircuitJson.nodes?.find((n: any) => n.node_id === clickedId)?.clerp,
          updateCounter: updateCounter + 1
        });
        
        // 自动下载更新后的文件（使用原文件名）
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const fileName = originalFileName || 'circuit_data.json';
        const baseName = fileName.replace('.json', '');
        const updatedFileName = `${baseName}_updated_${timestamp}.json`;
        
        const updatedJsonString = JSON.stringify(updatedCircuitJson, null, 2);
        const blob = new Blob([updatedJsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = updatedFileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        // 添加到保存历史
        setSaveHistory(prev => [...prev, `${new Date().toLocaleTimeString()}: 节点 ${clickedId} - ${trimmedClerp.length === 0 ? '清空clerp' : `更新为: ${trimmedClerp.substring(0, 30)}...`}`]);
        
        console.log('📥 文件已自动下载:', updatedFileName);
        
        // 显示成功消息和使用指引
        alert(`✅ Clerp已成功保存并下载！${trimmedClerp.length === 0 ? '(保存为空内容)' : ''}\n\n📁 文件已保存到Downloads文件夹:\n${updatedFileName}\n\n💡 使用提示:\n1. 可以直接用新文件替换原文件\n2. 或者重新上传新文件到此页面\n3. 文件名包含时间戳避免覆盖`);
        
      } else {
        throw new Error(`未找到对应的节点数据 (node_id: ${clickedId})`);
      }
    } catch (err) {
      console.error('保存失败:', err);
      alert('保存失败: ' + (err instanceof Error ? err.message : '未知错误'));
    } finally {
      setIsSaving(false);
    }
  }, [clickedId, originalCircuitJson, editingClerp, originalFileName, setOriginalCircuitJson, updateCounter]);

  // 快速导出当前状态的函数
  const handleQuickExport = useCallback(() => {
    if (!originalCircuitJson) return;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const fileName = originalFileName || 'circuit_data.json';
    const baseName = fileName.replace('.json', '');
    const exportFileName = `${baseName}_export_${timestamp}.json`;
    
    const jsonString = JSON.stringify(originalCircuitJson, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    setHasUnsavedChanges(false);
    console.log('📤 快速导出完成:', exportFileName);
    alert(`📤 文件已导出到Downloads文件夹:\n${exportFileName}\n\n💡 要使用更新后的文件:\n1. 用新文件替换原文件\n2. 或者拖拽新文件到此页面重新加载`);
  }, [originalCircuitJson, originalFileName]);

  // 获取 Top Activation 数据的函数
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId) return;
    
    setLoadingTopActivations(true);
    try {
      // 从 nodeId 解析出 feature 信息
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // 确定节点类型和对应的字典名
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
      
      // 使用辅助函数获取字典名
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      
      console.log('🔍 获取 Top Activation 数据:', {
        nodeId,
        layerIdx,
        featureIndex,
        dictionary,
        isLorsa
      });
      
      // 调用后端 API 获取 feature 数据
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
      const decoded = await import("@msgpack/msgpack").then(module => module.decode(new Uint8Array(arrayBuffer)));
      const camelcaseKeys = await import("camelcase-keys").then(module => module.default);
      
      // 解析数据
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
      // 提取样本数据
      const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
      const allSamples: any[] = [];
      
      for (const group of sampleGroups) {
        if (group.samples && Array.isArray(group.samples)) {
          allSamples.push(...group.samples);
        }
      }
      
      // 查找包含 FEN 的样本并提取激活值
      const chessSamples: any[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          
          for (const line of lines) {
            const trimmed = line.trim();
            
            // 检查是否包含 FEN 格式
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
                  // 验证 FEN 格式
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
                    // 处理稀疏激活数据 - 正确映射到64格棋盘
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0; // 使用最大激活值而不是总和
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      // 创建64格的激活数组
                      activationsArray = new Array(64).fill(0);
                      
                      // 将稀疏激活值映射到正确的棋盘位置，并找到最大激活值
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        // 确保索引在有效范围内
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          // 使用最大激活值（与feature页面逻辑一致）
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                      
                      console.log('🔍 处理激活数据:', {
                        indicesLength: sample.featureActsIndices.length,
                        valuesLength: sample.featureActsValues.length,
                        nonZeroCount: activationsArray.filter(v => v !== 0).length,
                        maxActivation
                      });
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation, // 使用最大激活值作为排序依据
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break; // 找到一个有效 FEN 就跳出
                  }
                }
              }
            }
          }
        }
      }
      
      // 按最大激活值排序并取前8个（与feature页面逻辑一致）
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      console.log('✅ 获取到 Top Activation 数据:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length
      });
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error('❌ 获取 Top Activation 数据失败:', error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [linkGraphData, getDictionaryName]);

  // 获取 Token Predictions 数据的函数
  const fetchTokenPredictions = useCallback(async (nodeId: string, currentSteeringScale?: number) => {
    if (!nodeId || !fen) return;

    if (!isSaeLoaded) {
      console.warn("TC/LoRSA 未加载，跳过 steering_analysis 调用");
      alert("请先在上方加载 TC/LoRSA 组合（SaeComboLoader），再使用 steering 功能。");
      setTokenPredictions(null);
      return;
    }
    
    setLoadingTokenPredictions(true);
    try {
      // 从 nodeId 解析出特征信息
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const pos = Number(parts[2]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // 确定节点类型
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const featureType = currentNode?.feature_type?.toLowerCase() === 'lorsa' ? 'lorsa' : 'transcoder';
      
      // 使用传入的 steeringScale 或当前状态中的值
      const scaleToUse = currentSteeringScale !== undefined ? currentSteeringScale : steeringScale;
      
      console.log('🔍 获取 Token Predictions 数据:', {
        nodeId,
        layerIdx,
        featureIndex,
        pos,
        featureType,
        fen,
        steering_scale: scaleToUse
      });
      
      // 调用后端 API 进行 steering 分析（支持 steering_scale 参数）
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/steering_analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            fen: fen,
            feature_type: featureType,
            layer: layerIdx,
            pos: pos,
            feature: featureIndex,
            steering_scale: scaleToUse,
            metadata: linkGraphData?.metadata
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 获取到 Token Predictions 数据:', result);
      
      setTokenPredictions(result);
      
    } catch (error) {
      console.error('❌ 获取 Token Predictions 数据失败:', error);
      setTokenPredictions(null);
    } finally {
      setLoadingTokenPredictions(false);
    }
  }, [fen, linkGraphData, isSaeLoaded, steeringScale]);

  // 当点击节点时获取 Top Activation 数据（Token Predictions 改为手动触发）
  useEffect(() => {
    if (clickedId) {
      fetchTopActivations(clickedId);
    } else {
      setTopActivations([]);
      setTokenPredictions(null);
    }
  }, [clickedId, fetchTopActivations]);

  // 同步clerps到后端interpretations
  const syncClerpsToBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert('⚠️ 没有可用的节点数据');
      return;
    }
    
    setSyncingToBackend(true);
    try {
      // 从metadata中提取analysis_name
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      // 准备节点数据
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          clerp: node.clerp || '',
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('📤 开始同步clerps到后端:', {
        totalNodes: nodes.length,
        nodesWithClerp: nodes.filter((n: any) => n.clerp).length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_clerps_to_interpretations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 同步完成:', result);
      
      alert(
        `✅ Clerp同步到后端完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 成功同步: ${result.synced}\n` +
        `- 跳过(无clerp): ${result.skipped}\n` +
        `- 失败: ${result.errors}`
      );
      
    } catch (error) {
      console.error('❌ 同步失败:', error);
      alert(`❌ 同步失败: ${error}`);
    } finally {
      setSyncingToBackend(false);
    }
  }, [originalCircuitJson, linkGraphData]);

  // 从后端interpretations同步到clerps
  const syncClerpsFromBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert('⚠️ 没有可用的节点数据');
      return;
    }
    
    setSyncingFromBackend(true);
    try {
      // 从metadata中提取analysis_name
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      // 准备节点数据
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('📥 开始从后端同步interpretations到clerps:', {
        totalNodes: nodes.length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 同步完成:', result);
      
      // 更新原始JSON数据
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // 根据返回的updated_nodes更新clerp
      const updatedNodesMap = new Map(
        result.updated_nodes.map((n: any) => [n.node_id, n.clerp])
      );
      
      let updatedCount = 0;
      updatedCircuitJson.nodes.forEach((node: any) => {
        if (updatedNodesMap.has(node.node_id)) {
          const newClerp = updatedNodesMap.get(node.node_id);
          if (newClerp) {
            node.clerp = newClerp;
            updatedCount++;
          }
        }
      });
      
      // 更新状态
      setOriginalCircuitJson(updatedCircuitJson);
      setUpdateCounter(prev => prev + 1);
      setHasUnsavedChanges(true);
      
      alert(
        `✅ 从后端同步Interpretation完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 找到interpretation: ${result.found}\n` +
        `- 未找到: ${result.not_found}\n` +
        `- 实际更新: ${updatedCount}\n\n` +
        `💡 建议: 点击"导出"按钮保存更新后的文件`
      );
      
    } catch (error) {
      console.error('❌ 同步失败:', error);
      alert(`❌ 同步失败: ${error}`);
    } finally {
      setSyncingFromBackend(false);
    }
  }, [originalCircuitJson, linkGraphData, setOriginalCircuitJson, setUpdateCounter, setHasUnsavedChanges]);

  // 检查dense features的函数
  const checkDenseFeatures = useCallback(async () => {
    if (!linkGraphData || !linkGraphData.nodes) {
      console.warn('⚠️ 没有可用的节点数据');
      return;
    }
    
    setCheckingDenseFeatures(true);
    try {
      const threshold = denseThreshold === '' ? null : parseInt(denseThreshold);
      
      // 从linkGraphData中提取所有节点的信息
      const nodes = linkGraphData.nodes.map(node => {
        // 从nodeId解析layer和feature
        const parts = node.nodeId.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.nodeId,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      // 从metadata中提取模型名称并转换为analysis_name
      const metadata = (linkGraphData.metadata || {}) as any;
      const lorsaAnalysisNameRaw = metadata.lorsa_analysis_name;
      const tcAnalysisNameRaw = metadata.tc_analysis_name || metadata.clt_analysis_name;
      // 从metadata中读取sae_series，如果没有则使用默认值
      const saeSeries = (metadata as any).sae_series || 'BT4-exp128';
      
      // 根据analysis_name构建模板
      // 格式：如果analysis_name是 "BT4_lorsa_k30_e16"，模板应该是 "BT4_lorsa_L{}A_k30_e16"
      // 如果analysis_name是 "BT4_lorsa"（默认），模板应该是 "BT4_lorsa_L{}A"
      let lorsaAnalysisName = undefined;
      let tcAnalysisName = undefined;
      
      if (lorsaAnalysisNameRaw) {
        if (lorsaAnalysisNameRaw.includes('BT4_lorsa')) {
          // 提取后缀（如果有）："BT4_lorsa_k30_e16" -> "k30_e16"
          const suffix = lorsaAnalysisNameRaw.replace('BT4_lorsa', '').replace(/^_/, '');
          if (suffix) {
            lorsaAnalysisName = `BT4_lorsa_L{}A_${suffix}`;
          } else {
            lorsaAnalysisName = 'BT4_lorsa_L{}A';
          }
        } else if (lorsaAnalysisNameRaw.includes('T82')) {
          lorsaAnalysisName = 'lc0-lorsa-L{}';
        }
      }
      
      if (tcAnalysisNameRaw) {
        if (tcAnalysisNameRaw.includes('BT4_tc')) {
          // 提取后缀（如果有）："BT4_tc_k30_e16" -> "k30_e16"
          const suffix = tcAnalysisNameRaw.replace('BT4_tc', '').replace(/^_/, '');
          if (suffix) {
            tcAnalysisName = `BT4_tc_L{}M_${suffix}`;
          } else {
            tcAnalysisName = 'BT4_tc_L{}M';
          }
        } else if (tcAnalysisNameRaw.includes('T82')) {
          tcAnalysisName = 'lc0_L{}M_16x_k30_lr2e-03_auxk_sparseadam';
        }
      }
      
      console.log('🔍 开始检查dense features:', {
        totalNodes: nodes.length,
        threshold: threshold,
        saeSeries: saeSeries,
        lorsaAnalysisNameRaw: lorsaAnalysisNameRaw,
        tcAnalysisNameRaw: tcAnalysisNameRaw,
        lorsaAnalysisName: lorsaAnalysisName,
        tcAnalysisName: tcAnalysisName,
        sampleNodes: nodes.slice(0, 3)
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/check_dense_features`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            threshold: threshold,
            sae_series: saeSeries,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ Dense features检查完成:', {
        denseNodeCount: result.dense_nodes.length,
        totalNodes: result.total_nodes,
        threshold: result.threshold
      });
      
      setDenseNodes(new Set(result.dense_nodes));
      
    } catch (error) {
      console.error('❌ 检查dense features失败:', error);
      alert(`检查dense features失败: ${error}`);
    } finally {
      setCheckingDenseFeatures(false);
    }
  }, [linkGraphData, denseThreshold]);

  // 应用dense节点颜色覆盖
  const applyDenseNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || denseNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (denseNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#000000',  // 黑色
            isDense: true  // 标记为dense节点
          };
        }
        return node;
      })
    };
  }, [denseNodes]);

  // 应用inactive节点颜色覆盖（金色）
  const applyInactiveNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || inactiveNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (inactiveNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#FFD700',  // 金色
            isInactive: true  // 标记为inactive节点
          };
        }
        return node;
      })
    };
  }, [inactiveNodes]);

  // 获取应用了dense和inactive颜色的图数据
  const displayLinkGraphData = useMemo(() => {
    let data = linkGraphData;
    data = applyDenseNodeColors(data);
    data = applyInactiveNodeColors(data);
    return data;
  }, [linkGraphData, applyDenseNodeColors, applyInactiveNodeColors]);

  // 比较FEN激活差异
  const compareFenActivations = useCallback(async () => {
    if (!originalCircuitJson || !perturbedFen.trim()) {
      alert('请先上传graph文件并输入perturbed FEN');
      return;
    }

    // 获取原始FEN
    const originalFen = extractFenFromPrompt();
    if (!originalFen) {
      alert('无法从graph文件中提取原始FEN');
      return;
    }

    setIsComparingFens(true);
    setDiffingLogs([]);
    setShowDiffingLogs(true);

    const addLog = (message: string) => {
      const logEntry = {
        timestamp: Date.now(),
        message: `[${new Date().toLocaleTimeString()}] ${message}`
      };
      setDiffingLogs(prev => [...prev, logEntry]);
      console.log(logEntry.message);
    };

    try {
      addLog('开始比较FEN激活差异...');
      addLog(`原始FEN: ${originalFen}`);
      addLog(`扰动FEN: ${perturbedFen}`);

      // 从metadata中提取模型名称
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const modelName = metadata?.model_name || 'lc0/BT4-1024x15x32h';

      addLog(`使用模型: ${modelName}`);

      // 调用后端API
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/compare_fen_activations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            graph_json: originalCircuitJson,
            original_fen: originalFen,
            perturbed_fen: perturbedFen.trim(),
            model_name: modelName,
            activation_threshold: 0.0
          })
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `HTTP ${response.status}: ${errorText}`;
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch {
          // 如果无法解析JSON，使用原始错误文本
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      
      addLog(`✅ 比较完成:`);
      addLog(`   - 总节点数: ${result.total_nodes}`);
      addLog(`   - 未激活节点数: ${result.inactive_nodes_count}`);

      // 更新inactive nodes集合
      const inactiveNodeIds = new Set<string>(
        result.inactive_nodes.map((node: any) => String(node.node_id))
      );
      setInactiveNodes(inactiveNodeIds);

      // 显示统计信息
      if (result.statistics) {
        addLog(`按层统计:`);
        Object.entries(result.statistics.by_layer).forEach(([layer, count]) => {
          addLog(`   Layer ${layer}: ${count} 个节点`);
        });
        addLog(`按类型统计:`);
        Object.entries(result.statistics.by_type).forEach(([type, count]) => {
          addLog(`   ${type}: ${count} 个节点`);
        });
      }

      alert(
        `✅ FEN激活差异比较完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 未激活节点数: ${result.inactive_nodes_count}\n\n` +
        `💡 未激活的节点已在图中标记为金色`
      );

    } catch (error) {
      console.error('❌ 比较失败:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      addLog(`❌ 比较失败: ${errorMessage}`);
      alert(`比较失败: ${errorMessage}`);
    } finally {
      setIsComparingFens(false);
    }
  }, [originalCircuitJson, perturbedFen, linkGraphData, extractFenFromPrompt]);

  // 递归查找某个节点的所有上游节点（包括它们的上游）
  const findUpstreamNodes = useCallback((nodeId: string, graphData: any): Set<string> => {
    const upstreamNodes = new Set<string>();
    const visited = new Set<string>();
    
    const traverse = (currentNodeId: string) => {
      if (visited.has(currentNodeId)) return;
      visited.add(currentNodeId);
      upstreamNodes.add(currentNodeId);
      
      // 查找指向当前节点的所有边（入边）
      const incomingLinks = graphData.links.filter((link: any) => link.target === currentNodeId);
      
      // 递归查找每个源节点的上游
      for (const link of incomingLinks) {
        traverse(link.source);
      }
    };
    
    traverse(nodeId);
    return upstreamNodes;
  }, []);

  // 创建子图数据
  const createSubgraph = useCallback((rootNodeId: string, graphData: any) => {
    const upstreamNodeIds = findUpstreamNodes(rootNodeId, graphData);
    
    // 过滤节点：只保留上游节点
    const subgraphNodes = graphData.nodes.filter((node: any) => 
      upstreamNodeIds.has(node.nodeId)
    );
    
    // 过滤边：只保留两端都在子图中的边
    const subgraphLinks = graphData.links.filter((link: any) => 
      upstreamNodeIds.has(link.source) && upstreamNodeIds.has(link.target)
    );
    
    // 创建子图数据结构
    const subgraph = {
      nodes: subgraphNodes,
      links: subgraphLinks,
      metadata: {
        ...graphData.metadata,
        subgraphRoot: rootNodeId,
        originalNodeCount: graphData.nodes.length,
        subgraphNodeCount: subgraphNodes.length,
        originalLinkCount: graphData.links.length,
        subgraphLinkCount: subgraphLinks.length,
        createdAt: new Date().toISOString(),
        isSubgraph: true
      }
    };
    
    console.log('🔍 创建子图:', {
      rootNodeId,
      totalUpstreamNodes: upstreamNodeIds.size,
      subgraphNodes: subgraphNodes.length,
      subgraphLinks: subgraphLinks.length,
      originalNodes: graphData.nodes.length,
      originalLinks: graphData.links.length
    });
    
    return subgraph;
  }, [findUpstreamNodes]);

  // 显示子图
  const handleShowSubgraph = useCallback(() => {
    if (!clickedId || !displayLinkGraphData) return;
    
    const subgraph = createSubgraph(clickedId, displayLinkGraphData);
    setSubgraphData(subgraph);
    setSubgraphRootNodeId(clickedId);
    setShowSubgraph(true);
    
    console.log('🎯 显示子图模式:', {
      rootNodeId: clickedId,
      nodeCount: subgraph.nodes.length,
      linkCount: subgraph.links.length
    });
  }, [clickedId, displayLinkGraphData, createSubgraph]);

  // 退出子图模式
  const handleExitSubgraph = useCallback(() => {
    setShowSubgraph(false);
    setSubgraphData(null);
    setSubgraphRootNodeId(null);
    console.log('🔙 退出子图模式');
  }, []);

  // 保存子图为JSON文件
  const handleSaveSubgraph = useCallback(() => {
    if (!subgraphData || !subgraphRootNodeId) return;
    
    // 从原始数据中获取完整的节点信息（包括激活数据、z_pattern等）
    const enrichSubgraphWithOriginalData = (subgraph: any) => {
      if (!originalCircuitJson) return subgraph;
      
      const enrichedNodes = subgraph.nodes.map((node: any) => {
        // 从原始JSON中查找对应的完整节点数据
        const originalNodeData = getNodeActivationDataFromJson(originalCircuitJson, node.nodeId);
        
        return {
          ...node,
          // 添加原始数据中的完整信息
          activations: originalNodeData.activations,
          zPatternIndices: originalNodeData.zPatternIndices,
          zPatternValues: originalNodeData.zPatternValues,
          clerp: originalNodeData.clerp,
          // 保留所有原始字段
          ...(originalCircuitJson.nodes?.find((n: any) => n.node_id === node.nodeId) || {})
        };
      });
      
      return {
        ...subgraph,
        nodes: enrichedNodes
      };
    };
    
    const enrichedSubgraph = enrichSubgraphWithOriginalData(subgraphData);
    
    // 生成文件名
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const rootNodeForFilename = subgraphRootNodeId.replace(/[^a-zA-Z0-9]/g, '_');
    const fileName = `subgraph_${rootNodeForFilename}_${timestamp}.json`;
    
    // 创建下载
    const jsonString = JSON.stringify(enrichedSubgraph, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    console.log('💾 子图已保存:', {
      fileName,
      rootNodeId: subgraphRootNodeId,
      nodeCount: enrichedSubgraph.nodes.length,
      linkCount: enrichedSubgraph.links.length
    });
    
    alert(
      `✅ 子图已保存！\n\n` +
      `📁 文件名: ${fileName}\n` +
      `🎯 根节点: ${subgraphRootNodeId}\n` +
      `📊 统计:\n` +
      `  - 节点数: ${enrichedSubgraph.nodes.length}\n` +
      `  - 边数: ${enrichedSubgraph.links.length}\n` +
      `  - 包含完整激活数据和z_pattern信息\n\n` +
      `💡 文件已保存到Downloads文件夹`
    );
    
  }, [subgraphData, subgraphRootNodeId, originalCircuitJson, getNodeActivationDataFromJson]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-red-600 mb-2">Failed to load circuit visualization</h3>
          <p className="text-gray-600">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading circuit visualization...</p>
        </div>
      </div>
    );
  }

  if (!linkGraphData) {
    return (
      <div className="space-y-6">
        {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
        <SaeComboLoader />

        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Circuit Visualization</h2>
        </div>

        {/* Upload Interface */}
        <div 
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragOver 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Upload Circuit Data
              </h3>
              <p className="text-gray-600 mb-4">
                Drag and drop 1-4 JSON files here, or click to browse
              </p>
              <input
                type="file"
                accept=".json"
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
                multiple
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 cursor-pointer transition-colors"
              >
                Choose Files
              </label>
            </div>
            <p className="text-sm text-gray-500">
              Supports uploading multiple JSON files (1-4) to merge graphs
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 调试传递给ChessBoard的数据
  if (clickedId && nodeActivationData) {
    console.log('🎲 传递给ChessBoard的数据:', {
      nodeId: clickedId,
      hasActivations: !!nodeActivationData.activations,
      activationsLength: nodeActivationData.activations?.length || 0,
      hasZPatternIndices: !!nodeActivationData.zPatternIndices,
      hasZPatternValues: !!nodeActivationData.zPatternValues,
      nodeType: nodeActivationData.nodeType,
      hasClerp: !!nodeActivationData.clerp,
      clerpLength: nodeActivationData.clerp?.length || 0
    });
  }

  return (
    <div className="space-y-6 w-full max-w-full overflow-hidden">
      {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
      <SaeComboLoader />

      {/* Header */}
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex items-center space-x-2 min-w-0">
          <h2 className="text-l font-bold whitespace-nowrap">Prompt:</h2>
          <h2 className="text-l truncate">{displayLinkGraphData?.metadata?.prompt_tokens?.join(' ') || ''}</h2>
        </div>
        <div className="flex flex-wrap items-start justify-end gap-3 ml-auto">
          <button
            onClick={() => setLinkGraphData(null)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Upload New File
          </button>
          {/* Graph Feature Diffing 控件 - 只在单图时显示 */}
          {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-yellow-50 rounded-md border border-yellow-200">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-700">Perturb FEN:</label>
                <input
                  type="text"
                  value={perturbedFen}
                  onChange={(e) => setPerturbedFen(e.target.value)}
                  placeholder="输入扰动后的FEN..."
                  className="w-64 px-2 py-1 text-sm border border-gray-300 rounded"
                  disabled={isComparingFens}
                  title="输入扰动后的FEN字符串，用于比较激活差异"
                />
                <button
                  onClick={compareFenActivations}
                  disabled={isComparingFens || !perturbedFen.trim()}
                  className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="比较原始FEN和扰动FEN的激活差异"
                >
                  {isComparingFens ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                      比较中...
                    </>
                  ) : (
                    '比较激活差异'
                  )}
                </button>
                {inactiveNodes.size > 0 && (
                  <span className="text-sm text-yellow-700 font-medium">
                    {inactiveNodes.size} 个未激活节点
                  </span>
                )}
                <button
                  onClick={() => setShowDiffingLogs(!showDiffingLogs)}
                  className="px-2 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  title="显示/隐藏比较日志"
                >
                  {showDiffingLogs ? '隐藏日志' : '显示日志'}
                </button>
              </div>
            </div>
          )}
          {/* Clerp同步控件 */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-blue-50 rounded-md border border-blue-200">
            <button
              onClick={syncClerpsToBackend}
              disabled={syncingToBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="将JSON中所有节点的clerp同步到后端MongoDB的interpretation"
            >
              {syncingToBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  上传中...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  上传Clerp
                </>
              )}
            </button>
            <button
              onClick={syncClerpsFromBackend}
              disabled={syncingFromBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="从后端MongoDB读取interpretation并同步到JSON节点的clerp"
            >
              {syncingFromBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  下载中...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                  </svg>
                  下载Clerp
                </>
              )}
            </button>
          </div>
          
          {/* Dense Feature检查控件 */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-md">
            <label className="text-sm text-gray-700">Dense阈值:</label>
            <input
              type="number"
              value={denseThreshold}
              onChange={(e) => setDenseThreshold(e.target.value)}
              placeholder="无限大"
              className="w-24 px-2 py-1 text-sm border border-gray-300 rounded"
              title="激活次数阈值，空表示无限大（所有节点保留）"
            />
            <button
              onClick={checkDenseFeatures}
              disabled={checkingDenseFeatures}
              className="px-3 py-1 text-sm bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="检查哪些节点是dense feature"
            >
              {checkingDenseFeatures ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  检查中...
                </>
              ) : (
                '判断Dense'
              )}
            </button>
            {denseNodes.size > 0 && (
              <span className="text-sm text-purple-700 font-medium">
                {denseNodes.size} 个Dense节点
              </span>
            )}
          </div>
          
          {/* 颜色-文件名图例（多文件时显示） */}
          {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="hidden md:flex items-center space-x-3 mr-4">
              {displayLinkGraphData.metadata.sourceFileNames.map((name, idx) => (
                <div key={idx} className="flex items-center space-x-1 text-xs">
                  <span
                    className="inline-block rounded-full"
                    style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                    title={name}
                  />
                  <span className="text-gray-600 truncate max-w-[140px]" title={name}>{name}</span>
                </div>
              ))}
            </div>
          )}
          {hasUnsavedChanges && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-orange-100 text-orange-800 rounded-md text-sm">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span>有未导出的更改</span>
              <button
                onClick={handleQuickExport}
                className="ml-2 px-2 py-1 bg-orange-200 hover:bg-orange-300 text-orange-900 rounded text-xs transition-colors"
                title="立即导出所有更改"
              >
                导出
              </button>
            </div>
          )}
          {saveHistory.length > 0 && (
            <div className="relative group">
              <button className="px-3 py-1 text-sm bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors">
                保存历史 ({saveHistory.length})
              </button>
              <div className="absolute right-0 top-full mt-1 w-80 bg-white border rounded-lg shadow-lg z-10 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-3">
                  <h4 className="font-medium text-gray-900 mb-2">最近的更改:</h4>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {saveHistory.slice(-5).reverse().map((entry, index) => (
                      <div key={index} className="text-xs text-gray-600 p-2 bg-gray-50 rounded">
                        {entry}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chess Board Display - 单文件 */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="flex justify-center mb-6">
          <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-center flex-1">
                Circuit棋盘状态
                {clickedId && displayActivationData && (
                  <span className="text-sm font-normal text-blue-600 ml-2">
                    (节点: {clickedId}{displayActivationData.nodeType ? ` - ${displayActivationData.nodeType.toUpperCase()}` : ''})
                  </span>
                )}
              </h3>
              {clickedId && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setShowAllPositions(!showAllPositions)}
                    className={`px-3 py-1 text-sm rounded transition-colors ${
                      showAllPositions
                        ? 'bg-blue-500 text-white hover:bg-blue-600'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    title={showAllPositions ? '显示单个位置的激活' : '显示所有位置的激活（合并）'}
                  >
                    {showAllPositions ? '单位置模式' : '所有位置模式'}
                  </button>
                </div>
              )}
            </div>
            {outputMove && (
              <div className="text-center mb-2 text-sm text-green-600 font-medium">
                输出移动: {outputMove} 🎯
              </div>
            )}
            {clickedId && displayActivationData && displayActivationData.activations && (
              <div className="text-center mb-2 text-sm text-purple-600">
                {showAllPositions ? (
                  <>
                    所有位置合并激活: {displayActivationData.activations.filter((v: number) => v !== 0).length} 个非零激活
                    {displayActivationData.zPatternIndices && displayActivationData.zPatternValues && 
                      `, ${displayActivationData.zPatternValues.length} 个Z模式连接`
                    }
                    <span className="text-xs text-gray-500 ml-2">(取每个位置的最大激活值)</span>
                  </>
                ) : (
                  <>
                    激活数据: {displayActivationData.activations.filter((v: number) => v !== 0).length} 个非零激活
                    {displayActivationData.zPatternIndices && displayActivationData.zPatternValues && 
                      `, ${displayActivationData.zPatternValues.length} 个Z模式连接`
                    }
                  </>
                )}
              </div>
            )}
            <ChessBoard
              fen={fen}
              size="medium"
              showCoordinates={true}
              move={outputMove || undefined}
              activations={displayActivationData?.activations}
              zPatternIndices={displayActivationData?.zPatternIndices}
              zPatternValues={displayActivationData?.zPatternValues}
              flip_activation={Boolean(fen && fen.split(' ')[1] === 'b')}
              autoFlipWhenBlack={true}
              sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
              analysisName={displayActivationData?.nodeType || 'Circuit Node'}
              moveColor={(clickedId ? (displayLinkGraphData.nodes.find(n => n.nodeId === clickedId)?.nodeColor) : undefined) as any}
            />
          </div>
        </div>
      )}

      {/* Graph Feature Diffing 日志显示 - 只在单图时显示 */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && showDiffingLogs && (
        <div className="w-full border rounded-lg overflow-hidden mb-6">
          <div className="bg-yellow-800 text-white px-4 py-2 flex items-center justify-between">
            <h3 className="font-semibold">FEN激活差异比较日志</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setDiffingLogs([])}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                清空日志
              </button>
              <button
                onClick={() => setShowDiffingLogs(false)}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                隐藏
              </button>
            </div>
          </div>
          <div 
            id="diffing-logs-container"
            className="bg-gray-900 text-green-400 p-4 font-mono text-sm max-h-64 overflow-y-auto"
          >
            <div className="space-y-1">
              {diffingLogs.length === 0 ? (
                <div className="text-gray-500">暂无日志...</div>
              ) : (
                diffingLogs.map((log, index) => (
                  <div key={index} className="whitespace-pre-wrap">
                    {log.message}
                  </div>
                ))
              )}
              {isComparingFens && (
                <div className="text-yellow-400 animate-pulse">比较中...</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chess Board Display - 多文件：为每个源文件渲染一个棋盘，并按来源显示激活 */}
      {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
        <div className="space-y-4 mb-6">
          {/* 所有位置模式切换按钮（多文件时） */}
          {clickedId && (
            <div className="flex justify-center">
              <button
                onClick={() => setShowAllPositions(!showAllPositions)}
                className={`px-4 py-2 text-sm rounded transition-colors ${
                  showAllPositions
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                title={showAllPositions ? '显示单个位置的激活' : '显示所有位置的激活（合并）'}
              >
                {showAllPositions ? '单位置模式' : '所有位置模式'}
              </button>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {multiOriginalJsons.map((entry, idx) => {
              const fileFen = extractFenFromCircuitJson(entry.json);
              if (!fileFen) return null;
              const fileMove = extractOutputMoveFromCircuitJson(entry.json);
              // 判断当前选中节点是否属于该文件
              const currentNode = clickedId ? displayLinkGraphData.nodes.find(n => n.nodeId === clickedId) : null;
              const belongs = currentNode && (currentNode.sourceIndices?.includes(idx) || currentNode.sourceIndex === idx);
              
              // 获取该文件的激活数据
              let perFileActivation: NodeActivationData = { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
              if (clickedId && belongs) {
                if (showAllPositions) {
                  // 获取所有位置的激活数据（使用该文件的 JSON 数据）
                  const allPosData = getAllPositionsActivationData(clickedId, entry.json);
                  perFileActivation = allPosData || perFileActivation;
                } else {
                  // 获取单个位置的激活数据
                  perFileActivation = getNodeActivationDataFromJson(entry.json, clickedId);
                }
              }
              
              return (
                <div key={idx} className="bg-white rounded-lg border shadow-sm p-4 pb-8">
                  <h3 className="text-md font-semibold mb-3 flex items-center justify-center">
                    <span
                      className="inline-block rounded-full mr-2"
                      style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                      title={entry.fileName}
                    />
                    <span className="truncate" title={entry.fileName}>{entry.fileName}</span>
                    {clickedId && belongs && (
                      <span className="text-xs font-normal text-blue-600 ml-2">(含该节点)</span>
                    )}
                  </h3>
                  {fileMove && (
                    <div className="text-center mb-2 text-sm text-green-600 font-medium">
                      输出移动: {fileMove} 🎯
                    </div>
                  )}
                  {clickedId && belongs && perFileActivation.activations && (
                    <div className="text-center mb-2 text-sm text-purple-600">
                      {showAllPositions ? (
                        <>
                          所有位置合并激活: {perFileActivation.activations.filter((v: number) => v !== 0).length} 个非零激活
                          {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                            `, ${perFileActivation.zPatternValues.length} 个Z模式连接`}
                          <span className="text-xs text-gray-500 ml-2">(取每个位置的最大激活值)</span>
                        </>
                      ) : (
                        <>
                          激活数据: {perFileActivation.activations.filter((v: number) => v !== 0).length} 个非零激活
                          {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                            `, ${perFileActivation.zPatternValues.length} 个Z模式连接`}
                        </>
                      )}
                    </div>
                  )}
                  <ChessBoard
                    fen={fileFen}
                    size="medium"
                    showCoordinates={true}
                    move={fileMove || undefined}
                    activations={belongs ? perFileActivation.activations : undefined}
                    zPatternIndices={belongs ? perFileActivation.zPatternIndices : undefined}
                    zPatternValues={belongs ? perFileActivation.zPatternValues : undefined}
                    flip_activation={Boolean(fileFen && fileFen.split(' ')[1] === 'b')}
                    autoFlipWhenBlack={true}
                    sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
                    analysisName={(perFileActivation?.nodeType || 'Circuit Node') + ` @${idx+1}`}
                    moveColor={UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length]}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Circuit Visualization Layout */}
      <div className="space-y-6 w-full max-w-full overflow-hidden">
        {/* 子图模式控制栏 */}
        {clickedId && (
          <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-blue-900">选中节点:</span>
                <code className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-mono">
                  {clickedId}
                </code>
              </div>
              
              {!showSubgraph ? (
                <button
                  onClick={handleShowSubgraph}
                  className="px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors flex items-center"
                  title="显示以该节点为根的子图（包含所有上游节点）"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                  显示子图
                </button>
              ) : (
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2 text-sm text-green-700">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">子图模式</span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                      {subgraphData?.nodes.length || 0} 个节点
                    </span>
                  </div>
                  
                  <button
                    onClick={handleSaveSubgraph}
                    className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded hover:bg-green-600 transition-colors flex items-center"
                    title="保存子图为JSON文件（包含完整的激活数据和z_pattern）"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    保存子图
                  </button>
                  
                  <button
                    onClick={handleExitSubgraph}
                    className="px-3 py-1 bg-gray-500 text-white text-sm font-medium rounded hover:bg-gray-600 transition-colors flex items-center"
                    title="退出子图模式，显示完整图形"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    退出子图
                  </button>
                </div>
              )}
            </div>
            
            {showSubgraph && subgraphData && (
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex items-center space-x-2">
                  <span>根节点:</span>
                  <code className="px-1 bg-gray-100 rounded">{subgraphRootNodeId}</code>
                </div>
                <div className="flex items-center space-x-4">
                  <span>
                    节点: {subgraphData.nodes.length}/{subgraphData.metadata?.originalNodeCount || 0}
                    <span className="text-green-600 ml-1">
                      ({((subgraphData.nodes.length / (subgraphData.metadata?.originalNodeCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                  <span>
                    边: {subgraphData.links.length}/{subgraphData.metadata?.originalLinkCount || 0}
                    <span className="text-blue-600 ml-1">
                      ({((subgraphData.links.length / (subgraphData.metadata?.originalLinkCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Top Row: Link Graph and Node Connections side by side */}
        <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
          {/* Link Graph Component - Left Side */}
          <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <div className="w-full h-full overflow-hidden relative">
              {(showSubgraph ? subgraphData : displayLinkGraphData) && (
                <LinkGraphContainer 
                  data={showSubgraph ? subgraphData : displayLinkGraphData} 
                  onNodeClick={handleFeatureClick}
                  onNodeHover={handleFeatureHover}
                  onFeatureSelect={handleFeatureSelect}
                  onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                  onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  pinnedIds={pinnedIds}
                />
              )}
            </div>
          </div>

          {/* Node Connections Component - Right Side */}
          <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            {(showSubgraph ? subgraphData : displayLinkGraphData) && (
              <NodeConnections
                data={showSubgraph ? subgraphData : displayLinkGraphData}
                clickedId={clickedId}
                hoveredId={hoveredId}
                pinnedIds={pinnedIds}
                hiddenIds={hiddenIds}
                onFeatureClick={handleFeatureClick}
                onFeatureSelect={handleFeatureSelect}
                onFeatureHover={handleFeatureHover}
              />
            )}
          </div>
        </div>

        {/* Top Activation Section */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Top Activation 棋盘</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">节点: {clickedId}</span>
                {loadingTopActivations && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">加载中...</span>
                  </div>
                )}
              </div>
            </div>
            
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">正在获取 Top Activation 数据...</p>
                </div>
              </div>
            ) : topActivations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {topActivations.map((sample, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                    <div className="text-center mb-2">
                      <div className="text-sm font-medium text-gray-700">
                        Top #{index + 1}
                      </div>
                      <div className="text-xs text-gray-500">
                        最大激活值: {sample.activationStrength.toFixed(3)}
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
                      flip_activation={Boolean(sample.fen && sample.fen.split(' ')[1] === 'b')}
                      autoFlipWhenBlack={true}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>未找到包含棋盘的激活样本</p>
              </div>
            )}
          </div>
        )}

        {/* Token Predictions Section (简化版) */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Token Predictions</h3>
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-600" htmlFor="steering-scale-input">steering_scale:</label>
                  <input
                    id="steering-scale-input"
                    type="number"
                    step={0.1}
                    className="w-24 px-2 py-1 border rounded text-sm"
                    value={steeringScaleInput}
                    onChange={(e) => {
                      const inputValue = e.target.value;
                      setSteeringScaleInput(inputValue);
                      const v = parseFloat(inputValue);
                      if (Number.isFinite(v)) {
                        setSteeringScale(v);
                      }
                    }}
                    onBlur={() => {
                      if (steeringScaleInput === '' || steeringScaleInput === '-') {
                        setSteeringScale(0);
                        setSteeringScaleInput('0');
                      }
                    }}
                    title="调节steering放大系数，支持负数输入"
                  />
                </div>
                <button
                  onClick={() => clickedId && fetchTokenPredictions(clickedId)}
                  disabled={loadingTokenPredictions || !clickedId || !fen}
                  className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="运行特征干预分析"
                >
                  {loadingTokenPredictions ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      分析中...
                    </>
                  ) : (
                    '开始分析'
                  )}
                </button>
                {loadingTokenPredictions && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">分析中...</span>
                  </div>
                )}
              </div>
            </div>

            {loadingTokenPredictions ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">正在运行特征干预分析...</p>
                </div>
              </div>
            ) : tokenPredictions ? (
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-3 border">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600">steering_scale:</span>
                      <span className="ml-1 font-medium">{Number(steeringScale).toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">合法移动数:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.total_legal_moves}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">平均概率差:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_prob_diff?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">平均Logit差:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_logit_diff?.toFixed(4)}</span>
                    </div>
                  </div>
                </div>

                {/* 概率差异最大前5（增加最多） */}
                {tokenPredictions.promoting_moves && tokenPredictions.promoting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">概率差异最大（增加最多）Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.promoting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>排名: #{index + 1}</div>
                              <div>概率差: <span className="font-medium">{(move.prob_diff * 100).toFixed(2)}%</span></div>
                              <div>原始概率: {(move.original_prob * 100).toFixed(2)}%</div>
                              <div>修改后概率: {(move.modified_prob * 100).toFixed(2)}%</div>
                              <div>Logit差: {move.diff?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 概率差异最小前5（减少最多，负数最小） */}
                {tokenPredictions.inhibiting_moves && tokenPredictions.inhibiting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">概率差异最小（减少最多）Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.inhibiting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>排名: #{index + 1}</div>
                              <div>概率差: <span className="font-medium">{(move.prob_diff * 100).toFixed(2)}%</span></div>
                              <div>原始概率: {(move.original_prob * 100).toFixed(2)}%</div>
                              <div>修改后概率: {(move.modified_prob * 100).toFixed(2)}%</div>
                              <div>Logit差: {move.diff?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>点击"开始分析"按钮以运行Token Predictions分析</p>
                <p className="text-sm mt-2">请先在上方加载 TC/LoRSA 组合（SaeComboLoader）</p>
              </div>
            )}
          </div>
        )}

        {/* Clerp Editor - New Section */}
        {clickedId && nodeActivationData && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Node Clerp Editor</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">节点: {clickedId}</span>
                {nodeActivationData.nodeType && (
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
                    {nodeActivationData.nodeType.toUpperCase()}
                  </span>
                )}
              </div>
            </div>
            
            {/* 始终显示编辑器，无论clerp是否存在或为空 */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium text-gray-700">
                  Clerp内容 (可编辑)
                  {nodeActivationData.clerp === undefined && (
                    <span className="text-xs text-gray-500 ml-2">(节点暂无clerp字段，可新建)</span>
                  )}
                  {nodeActivationData.clerp === '' && (
                    <span className="text-xs text-gray-500 ml-2">(当前为空，可编辑)</span>
                  )}
                </label>
                <div className="text-xs text-gray-500">
                  字符数: {editingClerp.length}
                </div>
              </div>
              <textarea
                value={editingClerp}
                onChange={(e) => setEditingClerp(e.target.value)}
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder={
                  nodeActivationData.clerp === undefined 
                    ? "该节点暂无clerp字段，您可以在此输入新的clerp内容..." 
                    : "输入或编辑节点的clerp内容..."
                }
              />
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setEditingClerp(nodeActivationData.clerp || '')}
                  className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  disabled={isSaving}
                >
                  重置
                </button>
                {(() => {
                  const isDisabled = isSaving || editingClerp.trim() === (nodeActivationData.clerp || '');
                  console.log('🔍 按钮状态调试:', {
                    isSaving,
                    editingClerpTrimmed: editingClerp.trim(),
                    nodeActivationDataClerp: nodeActivationData.clerp,
                    nodeActivationDataClerpOrEmpty: nodeActivationData.clerp || '',
                    isEqual: editingClerp.trim() === (nodeActivationData.clerp || ''),
                    isDisabled
                  });
                  
                  return (
                    <button
                      onClick={handleSaveClerp}
                      disabled={isDisabled}
                      className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                      title="保存更改并自动下载更新后的文件到Downloads文件夹"
                    >
                      {isSaving && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      )}
                      {isSaving ? '保存中...' : '保存并下载'}
                    </button>
                  );
                })()}
              </div>
              {editingClerp.trim() !== (nodeActivationData.clerp || '') && (
                <div className="text-xs text-orange-600 bg-orange-50 p-2 rounded">
                  ⚠️ 内容已修改，请点击"保存到文件"以保存更改
                </div>
              )}
              
              {/* 显示当前状态信息 */}
              <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                <div className="flex justify-between">
                  <span>
                    原始状态: {
                      nodeActivationData.clerp === undefined 
                        ? '无clerp字段' 
                        : nodeActivationData.clerp === '' 
                          ? '空字符串' 
                          : `有内容 (${nodeActivationData.clerp.length} 字符)`
                    }
                  </span>
                  <span>
                    当前编辑: {editingClerp === '' ? '空' : `${editingClerp.length} 字符`}
                  </span>
                </div>
              </div>
              
              {/* 使用说明 */}
              <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                <div className="font-medium mb-1">💡 文件更新工作流程:</div>
                <ol className="list-decimal list-inside space-y-1 text-blue-700">
                  <li>编辑clerp内容后点击"保存并下载"</li>
                  <li>更新后的文件会自动下载到Downloads文件夹</li>
                  <li>用新文件替换原文件，或重新拖拽到此页面</li>
                  <li>文件名包含时间戳，避免意外覆盖</li>
                </ol>
                <div className="mt-2 text-xs">
                  <strong>提示:</strong> 由于浏览器安全限制，无法直接修改原文件，但下载的文件包含所有更改。
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Row: Feature Card below Link Graph Container */}
        {clickedId && displayLinkGraphData && (() => {
          // 获取当前选中节点的信息
          const currentNode = displayLinkGraphData.nodes.find(node => node.nodeId === clickedId);
          
          if (!currentNode) {
            return null;
          }
          
          // 从node_id解析真正的feature ID (格式: layer_featureId_ctxIdx)
          // 注意：layer需要除以2得到实际的模型层数，因为M和A分别占一层
          const parseNodeId = (nodeId: string) => {
            const parts = nodeId.split('_');
            if (parts.length >= 2) {
              const rawLayer = parseInt(parts[0]) || 0;
              return {
                layerIdx: Math.floor(rawLayer / 2), // 除以2得到实际模型层数
                featureIndex: parseInt(parts[1]) || 0
              };
            }
            return { layerIdx: 0, featureIndex: 0 };
          };
          
          const { layerIdx, featureIndex } = parseNodeId(currentNode.nodeId);
          const isLorsa = currentNode.feature_type?.toLowerCase() === 'lorsa';
          
          // 调试节点连接信息
          console.log('🔍 节点连接调试:', {
            nodeId: currentNode.nodeId,
            hasSourceLinks: !!currentNode.sourceLinks,
            sourceLinksCount: currentNode.sourceLinks?.length || 0,
            hasTargetLinks: !!currentNode.targetLinks,
            targetLinksCount: currentNode.targetLinks?.length || 0,
            totalLinksInData: displayLinkGraphData.links.length
          });
          
          // 使用辅助函数获取字典名
          const dictionary = getDictionaryName(layerIdx, isLorsa);
          
          const nodeTypeDisplay = isLorsa ? 'LORSA' : 'SAE';
          
          return (
            <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Selected Feature Details</h3>
                <div className="flex items-center space-x-4">
                  {connectedFeatures.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">Connected features:</span>
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                        {connectedFeatures.length}
                      </span>
                    </div>
                  )}
                  {/* Circuit 标注按钮 */}
                  {currentNode && featureIndex !== undefined && (
                    <button
                      onClick={() => {
                        setSelectedNodeForCircuit({
                          nodeId: currentNode.nodeId,
                          layer: layerIdx,
                          feature: featureIndex,
                          feature_type: currentNode.feature_type || '',
                        });
                        setShowCircuitInterpretation(true);
                      }}
                      className="inline-flex items-center px-3 py-2 bg-purple-500 text-white text-sm font-medium rounded-md hover:bg-purple-600 transition-colors"
                      title="查看/管理该feature所属的circuit标注"
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Circuit 标注
                    </button>
                  )}
                  {/* 跳转到Feature页面的链接 */}
                  {currentNode && featureIndex !== undefined && (
                    <Link
                      to={`/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${featureIndex}`}
                      className="inline-flex items-center px-3 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors"
                      title={`跳转到L${layerIdx} ${nodeTypeDisplay} Feature #${featureIndex}`}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                      查看L{layerIdx} {nodeTypeDisplay} #{featureIndex}
                    </Link>
                  )}
                </div>
              </div>
              {selectedFeature ? (
                <FeatureCard feature={selectedFeature} />
              ) : (
                <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                  <div className="text-center">
                    <p className="text-gray-600">No feature is available for this node</p>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
      </div>

      {/* Circuit Interpretation Modal */}
      {showCircuitInterpretation && selectedNodeForCircuit && (
        <Suspense
          fallback={
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white p-4 rounded shadow-lg">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p className="mt-2 text-gray-600">Loading Circuit Interpretation...</p>
              </div>
            </div>
          }
        >
          <CircuitInterpretation
            node={selectedNodeForCircuit}
            saeComboId={
              (typeof window !== 'undefined' 
                ? window.localStorage.getItem("bt4_sae_combo_id") 
                : null) || 'k_30_e_16'
            }
            saeSeries="BT4-exp128"
            getSaeName={getSaeNameForCircuit}
            visible={showCircuitInterpretation}
            onClose={() => {
              console.log('[CircuitVisualization] CircuitInterpretation onClose called');
              setShowCircuitInterpretation(false);
              setSelectedNodeForCircuit(null);
            }}
          />
        </Suspense>
      )}
    </div>
  );
};

export default CircuitVisualization;
