import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { useCircuitState } from "@/contexts/AppStateContext";
import { LinkGraphContainer } from "./link-graph-container";
import { NodeConnections } from "./node-connections";
import { transformCircuitData, CircuitJsonData } from "./link-graph/utils";
import { Node } from "./link-graph/types";
import { Feature } from "@/types/feature";
import { FeatureCard } from "@/components/feature/feature-card";
import { ChessBoard } from "@/components/chess/chess-board";
import React from "react"; // Added missing import for React

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

  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [isLoadingConnectedFeatures, setIsLoadingConnectedFeatures] = useState(false);
  const [originalCircuitJson, setOriginalCircuitJson] = useState<any>(null); // 存储原始JSON数据（单图或合并后的）
  const [editingClerp, setEditingClerp] = useState<string>(''); // 当前编辑的clerp
  const [isSaving, setIsSaving] = useState(false); // 保存状态
  const [originalFileName, setOriginalFileName] = useState<string>(''); // 原始文件名（单文件时）
  const [updateCounter, setUpdateCounter] = useState(0); // 用于强制更新的计数器
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // 是否有未保存的更改
  const [saveHistory, setSaveHistory] = useState<string[]>([]); // 保存历史记录

  // 多图支持：存放多份原始 JSON 及其文件名
  const [multiOriginalJsons, setMultiOriginalJsons] = useState<{ json: CircuitJsonData; fileName: string }[]>([]);

  // 为“各自独有”的节点/边分配的颜色表（最多4个图）
  const UNIQUE_GRAPH_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B"]; // 蓝、橙、绿、红
  const DUPLICATE_FEATURE_ALT_COLOR = "#FF9800"; // 同一feature不同位置的备用颜色（琥珀橙）

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
      setClickedId(node.nodeId === clickedId ? null : node.nodeId);
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
    setIsLoadingConnectedFeatures(false);
  }, []);

  const handleConnectedFeaturesLoading = useCallback((loading: boolean) => {
    setIsLoadingConnectedFeatures(loading);
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
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures]);

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
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [mergeGraphs, setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures]);

  const handleFileUpload = useCallback(async (file: File) => {
    // 兼容旧调用，保留单文件路径
    return handleSingleFileUpload(file);
  }, [handleSingleFileUpload]);

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
          const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
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
    console.log('🔍 搜索输出移动:', promptText);

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
          console.log('✅ 找到移动:', moveMatch[0]);
          return moveMatch[0].toLowerCase();
        }
      }
    }

    console.log('❌ 未找到输出移动');
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
            const keys = Object.keys(item);
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
 
  // 提取相关数据
  const fen = extractFenFromPrompt();
  const outputMove = extractOutputMove();
  const nodeActivationData = getNodeActivationData(clickedId);

  // 修复Hook使用 - 移到组件顶层，避免条件调用
  React.useEffect(() => {
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
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <h2 className="text-l font-bold">Prompt:</h2>
          <h2 className="text-l">{linkGraphData.metadata.prompt_tokens.join(' ')}</h2>
        </div>
        <div className="flex items-center space-x-2">
          {/* 颜色-文件名图例（多文件时显示） */}
          {linkGraphData.metadata.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="hidden md:flex items-center space-x-3 mr-4">
              {linkGraphData.metadata.sourceFileNames.map((name, idx) => (
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
          <button
            onClick={() => setLinkGraphData(null)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Upload New File
          </button>
        </div>
      </div>

      {/* Chess Board Display - 单文件 */}
      {(!linkGraphData.metadata.sourceFileNames || linkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="flex justify-center mb-6">
          <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
            <h3 className="text-lg font-semibold mb-4 text-center">
              Circuit棋盘状态
              {clickedId && nodeActivationData && (
                <span className="text-sm font-normal text-blue-600 ml-2">
                  (节点: {clickedId}{nodeActivationData.nodeType ? ` - ${nodeActivationData.nodeType.toUpperCase()}` : ''})
                </span>
              )}
            </h3>
            {outputMove && (
              <div className="text-center mb-2 text-sm text-green-600 font-medium">
                输出移动: {outputMove} 🎯
              </div>
            )}
            {clickedId && nodeActivationData && nodeActivationData.activations && (
              <div className="text-center mb-2 text-sm text-purple-600">
                激活数据: {nodeActivationData.activations.filter((v: number) => v !== 0).length} 个非零激活
                {nodeActivationData.zPatternIndices && nodeActivationData.zPatternValues && 
                  `, ${nodeActivationData.zPatternValues.length} 个Z模式连接`
                }
              </div>
            )}
            <ChessBoard
              fen={fen}
              size="medium"
              showCoordinates={true}
              move={outputMove || undefined}
              activations={nodeActivationData?.activations}
              zPatternIndices={nodeActivationData?.zPatternIndices}
              zPatternValues={nodeActivationData?.zPatternValues}
              flip_activation={false}
              sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
              analysisName={nodeActivationData?.nodeType || 'Circuit Node'}
              moveColor={(clickedId ? (linkGraphData.nodes.find(n => n.nodeId === clickedId)?.nodeColor) : undefined) as any}
            />
          </div>
        </div>
      )}

      {/* Chess Board Display - 多文件：为每个源文件渲染一个棋盘，并按来源显示激活 */}
      {linkGraphData.metadata.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {multiOriginalJsons.map((entry, idx) => {
            const fileFen = extractFenFromCircuitJson(entry.json);
            if (!fileFen) return null;
            const fileMove = extractOutputMoveFromCircuitJson(entry.json);
            // 判断当前选中节点是否属于该文件
            const currentNode = clickedId ? linkGraphData.nodes.find(n => n.nodeId === clickedId) : null;
            const belongs = currentNode && (currentNode.sourceIndices?.includes(idx) || currentNode.sourceIndex === idx);
            const perFileActivation = (clickedId && belongs)
              ? getNodeActivationDataFromJson(entry.json, clickedId)
              : { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
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
                    激活数据: {perFileActivation.activations.filter((v: number) => v !== 0).length} 个非零激活
                    {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                      `, ${perFileActivation.zPatternValues.length} 个Z模式连接`}
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
                  flip_activation={false}
                  sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
                  analysisName={(perFileActivation?.nodeType || 'Circuit Node') + ` @${idx+1}`}
                  moveColor={UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length]}
                />
              </div>
            );
          })}
        </div>
      )}

      {/* Circuit Visualization Layout */}
      <div className="space-y-6 w-full max-w-full overflow-hidden">
        {/* Top Row: Link Graph and Node Connections side by side */}
        <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
          {/* Link Graph Component - Left Side */}
          <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <div className="w-full h-full overflow-hidden relative">
              <LinkGraphContainer 
                data={linkGraphData} 
                onNodeClick={handleFeatureClick}
                onNodeHover={handleFeatureHover}
                onFeatureSelect={handleFeatureSelect}
                onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                clickedId={clickedId}
                hoveredId={hoveredId}
                pinnedIds={pinnedIds}
              />
            </div>
          </div>

          {/* Node Connections Component - Right Side */}
          <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <NodeConnections
              data={linkGraphData}
              clickedId={clickedId}
              hoveredId={hoveredId}
              pinnedIds={pinnedIds}
              hiddenIds={hiddenIds}
              onFeatureClick={handleFeatureClick}
              onFeatureSelect={handleFeatureSelect}
              onFeatureHover={handleFeatureHover}
            />
          </div>
        </div>

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
        {clickedId && (() => {
          // 获取当前选中节点的信息
          const currentNode = linkGraphData.nodes.find(node => node.nodeId === clickedId);
          
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
            totalLinksInData: linkGraphData.links.length
          });
          
          // 根据节点类型构建正确的dictionary名
          const dictionary = isLorsa 
            ? `lc0-lorsa-L${layerIdx}`
            : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
          
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
    </div>
  );
};

export default CircuitVisualization;
