import React, { useState, useCallback, useEffect } from 'react';
import { AppNavbar } from '@/components/app/navbar';
import { LinkGraphContainer } from '@/components/circuits/link-graph-container';
import { LinkGraphData, Node, Link } from '@/components/circuits/link-graph/types';
import { Feature } from '@/types/feature';
import { ChessBoard } from '@/components/chess/chess-board';
import { CustomFenInput } from '@/components/feature/custom-fen-input';
import { FeatureConnections } from '@/components/circuits/feature-connections';

interface CsvRow {
  source_layer: number;
  source_pos: number;
  source_feature: number;
  source_type: string;
  target_layer: number;
  target_pos: number;
  target_feature: number;
  target_type: string;
  reduction_ratio: number;
  original_activation?: number;
  modified_activation?: number;
  activation_change?: number;
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

export const InteractionCircuitPage = () => {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [linkGraphData, setLinkGraphData] = useState<LinkGraphData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clickedId, setClickedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [pinnedIds, setPinnedIds] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [topActivations, setTopActivations] = useState<TopActivationSample[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);
  const [currentFen, setCurrentFen] = useState<string>('');
  const [fenActivationData, setFenActivationData] = useState<{
    activations?: number[];
    zPatternIndices?: number[][];
    zPatternValues?: number[];
  } | null>(null);
  const [loadingFenActivation, setLoadingFenActivation] = useState(false);

  // 解析 CSV 文件（支持带引号的字段）
  const parseCsvFile = useCallback(async (file: File): Promise<LinkGraphData> => {
    const text = await file.text();
    const lines = text.split('\n').filter(line => line.trim());
    
    if (lines.length < 2) {
      throw new Error('CSV 文件格式错误：至少需要包含表头和数据行');
    }

    // 简单的 CSV 解析函数（处理引号）
    const parseCsvLine = (line: string): string[] => {
      const result: string[] = [];
      let current = '';
      let inQuotes = false;
      
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          if (inQuotes && line[i + 1] === '"') {
            // 转义的引号
            current += '"';
            i++;
          } else {
            // 切换引号状态
            inQuotes = !inQuotes;
          }
        } else if (char === ',' && !inQuotes) {
          // 字段分隔符
          result.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      result.push(current.trim()); // 最后一个字段
      return result;
    };

    // 解析表头
    const headers = parseCsvLine(lines[0]).map(h => h.trim());
    const requiredHeaders = ['source_layer', 'source_pos', 'source_feature', 'source_type', 
                            'target_layer', 'target_pos', 'target_feature', 'target_type', 'reduction_ratio'];
    
    const missingHeaders = requiredHeaders.filter(h => !headers.includes(h));
    if (missingHeaders.length > 0) {
      throw new Error(`CSV 文件缺少必需的列: ${missingHeaders.join(', ')}`);
    }

    console.log('📋 CSV 表头:', headers);

    // 解析数据行
    const rows: CsvRow[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = parseCsvLine(lines[i]);
      if (values.length < requiredHeaders.length) {
        console.warn(`⚠️ 跳过行 ${i + 1}：字段数量不足 (${values.length} < ${requiredHeaders.length})`);
        continue;
      }

      try {
        // 确保正确解析每个字段，避免从错误的位置读取
        const sourceLayerIdx = headers.indexOf('source_layer');
        const sourcePosIdx = headers.indexOf('source_pos');
        const sourceFeatureIdx = headers.indexOf('source_feature');
        const sourceTypeIdx = headers.indexOf('source_type');
        const targetLayerIdx = headers.indexOf('target_layer');
        const targetPosIdx = headers.indexOf('target_pos');
        const targetFeatureIdx = headers.indexOf('target_feature');
        const targetTypeIdx = headers.indexOf('target_type');
        const reductionRatioIdx = headers.indexOf('reduction_ratio');

        if (sourceLayerIdx === -1 || sourcePosIdx === -1 || sourceFeatureIdx === -1 || 
            sourceTypeIdx === -1 || targetLayerIdx === -1 || targetPosIdx === -1 || 
            targetFeatureIdx === -1 || targetTypeIdx === -1 || reductionRatioIdx === -1) {
          console.warn(`⚠️ 跳过行 ${i + 1}：缺少必需的列索引`);
          continue;
        }

        const row: CsvRow = {
          source_layer: parseInt(values[sourceLayerIdx] || '0', 10) || 0,
          source_pos: parseInt(values[sourcePosIdx] || '0', 10) || 0,
          source_feature: parseInt(values[sourceFeatureIdx] || '0', 10) || 0,
          source_type: (values[sourceTypeIdx] || 'transcoder').trim().toLowerCase(),
          target_layer: parseInt(values[targetLayerIdx] || '0', 10) || 0,
          target_pos: parseInt(values[targetPosIdx] || '0', 10) || 0,
          target_feature: parseInt(values[targetFeatureIdx] || '0', 10) || 0,
          target_type: (values[targetTypeIdx] || 'transcoder').trim().toLowerCase(),
          reduction_ratio: parseFloat(values[reductionRatioIdx] || '0') || 0,
        };

        if (headers.includes('original_activation')) {
          row.original_activation = parseFloat(values[headers.indexOf('original_activation')] || '0') || 0;
        }
        if (headers.includes('modified_activation')) {
          row.modified_activation = parseFloat(values[headers.indexOf('modified_activation')] || '0') || 0;
        }
        if (headers.includes('activation_change')) {
          row.activation_change = parseFloat(values[headers.indexOf('activation_change')] || '0') || 0;
        }

        rows.push(row);
      } catch (err) {
        console.warn(`⚠️ 解析行 ${i + 1} 时出错:`, err);
        continue;
      }
    }

    console.log(`✅ 成功解析 ${rows.length} 行数据`);

    // 构建节点和边
    const nodeMap = new Map<string, Node>();
    const links: Link[] = [];

    // 节点颜色映射
    const getNodeColor = (featureType: string): string => {
      switch (featureType.toLowerCase()) {
        case 'lorsa':
          return '#7a4cff';
        case 'transcoder':
          return '#4ecdc4';
        default:
          return '#95a5a6';
      }
    };

    // 构建节点
    for (const row of rows) {
      // Source 节点
      const sourceNodeId = `${row.source_layer}_${row.source_pos}_${row.source_feature}`;
      if (!nodeMap.has(sourceNodeId)) {
        const isLorsa = row.source_type.toLowerCase() === 'lorsa';
        // 对于显示：Lorsa layer 0 = A0, Transcoder layer 0 = M0
        // 但在构建字典名称时，我们直接使用 layer 值
        const displayLayer = row.source_layer;
        
        const sourceFeatureType = row.source_type.trim().toLowerCase();
        // 计算 Y 轴位置：lorsa 显示在 Ai 行，transcoder 显示在 Mi 行
        // 公式：layerIdx = layer * 2 + (isLorsa ? 0 : 1)
        // 这样：A0=0, M0=1, A1=2, M1=3, A2=4, M2=5, ...
        const yAxisLayer = displayLayer * 2 + (isLorsa ? 0 : 1);
        
        nodeMap.set(sourceNodeId, {
          id: sourceNodeId,
          nodeId: sourceNodeId,
          featureId: row.source_feature.toString(),
          feature_type: sourceFeatureType, // 确保 feature_type 是小写的
          ctx_idx: row.source_pos,
          layerIdx: yAxisLayer, // 使用计算后的 Y 轴位置
          pos: [0, 0],
          xOffset: 0,
          yOffset: 0,
          nodeColor: getNodeColor(sourceFeatureType),
          featureIndex: row.source_feature,
          // 格式：Mi#feature_id@position 或 Ai#feature_id@position
          localClerp: `${isLorsa ? 'A' : 'M'}${displayLayer}#${row.source_feature}@${row.source_pos}`,
        });
      }

      // Target 节点
      const targetNodeId = `${row.target_layer}_${row.target_pos}_${row.target_feature}`;
      if (!nodeMap.has(targetNodeId)) {
        const isLorsa = row.target_type.toLowerCase() === 'lorsa';
        const displayLayer = row.target_layer;
        
        const targetFeatureType = row.target_type.trim().toLowerCase();
        // 计算 Y 轴位置（与 source 节点相同的方式）
        // 公式：layerIdx = layer * 2 + (isLorsa ? 0 : 1)
        // 这样：A0=0, M0=1, A1=2, M1=3, A2=4, M2=5, ...
        const targetYAxisLayer = displayLayer * 2 + (isLorsa ? 0 : 1);
        nodeMap.set(targetNodeId, {
          id: targetNodeId,
          nodeId: targetNodeId,
          featureId: row.target_feature.toString(),
          feature_type: targetFeatureType, // 确保 feature_type 是小写的
          ctx_idx: row.target_pos,
          layerIdx: targetYAxisLayer, // 使用计算后的 Y 轴位置
          pos: [0, 0],
          xOffset: 0,
          yOffset: 0,
          nodeColor: getNodeColor(targetFeatureType),
          featureIndex: row.target_feature,
          // 格式：Mi#feature_id@position 或 Ai#feature_id@position
          localClerp: `${isLorsa ? 'A' : 'M'}${displayLayer}#${row.target_feature}@${row.target_pos}`,
        });
      }

      // 构建边（边权统一设为1）
      const strokeWidth = 2; // 固定线宽
      const color = '#4CAF50'; // 统一颜色

      links.push({
        source: sourceNodeId,
        target: targetNodeId,
        pathStr: '',
        color,
        strokeWidth,
        weight: 1, // 边权统一设为1
        pctInput: 100, // 百分比设为100%
      });
    }

    // 为节点设置 sourceLinks 和 targetLinks
    // sourceLinks: 从该节点出发的边（link.source === node.nodeId）
    // targetLinks: 指向该节点的边（link.target === node.nodeId）
    const nodes = Array.from(nodeMap.values());
    for (const node of nodes) {
      node.sourceLinks = links.filter(l => l.source === node.nodeId);
      node.targetLinks = links.filter(l => l.target === node.nodeId);
    }

    return {
      nodes,
      links,
      metadata: {
        prompt_tokens: [],
        lorsa_analysis_name: 'BT4_lorsa_L{}A_k30_e16',
        tc_analysis_name: 'BT4_tc_L{}M_k30_e16',
      },
    };
  }, []);

  // 处理文件上传
  const handleFileUpload = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setLinkGraphData(null);
    setClickedId(null);
    setSelectedFeature(null);
    setTopActivations([]);

    try {
      const data = await parseCsvFile(file);
      setLinkGraphData(data);
      setCsvFile(file);
      console.log(`✅ 成功解析 CSV 文件: ${data.nodes.length} 个节点, ${data.links.length} 条边`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '解析 CSV 文件时发生错误';
      setError(errorMessage);
      console.error('❌ 解析 CSV 文件失败:', err);
    } finally {
      setIsLoading(false);
    }
  }, [parseCsvFile]);

  // 处理文件拖拽
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files);
    const csvFile = files.find(f => f.name.endsWith('.csv'));
    
    if (csvFile) {
      handleFileUpload(csvFile);
    } else {
      setError('请上传 CSV 文件');
    }
  }, [handleFileUpload]);

  // 处理文件选择
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.csv')) {
      handleFileUpload(file);
    } else {
      setError('请选择 CSV 文件');
    }
  }, [handleFileUpload]);

  // 获取字典名称
  // Lorsa layer 0 = A0, Transcoder layer 0 = M0
  // 所以 CSV 中的 layer 值直接用于构建字典名称
  const getDictionaryName = useCallback((layer: number, isLorsa: boolean): string => {
    // 使用默认的 BT4 格式，layer 值直接使用（0就是0，对应A0或M0）
    if (isLorsa) {
      return `BT4_lorsa_L${layer}A_k30_e16`;
    } else {
      return `BT4_tc_L${layer}M_k30_e16`;
    }
  }, []);

  // 辅助函数：标准化 Z Pattern 数据
  const normalizeZPattern = useCallback(
    (
      zPatternIndicesRaw: unknown,
      zPatternValuesRaw: unknown
    ): { zPatternIndices?: number[][]; zPatternValues?: number[] } => {
      if (!Array.isArray(zPatternIndicesRaw) || !Array.isArray(zPatternValuesRaw)) {
        return {};
      }

      const zPatternValues = zPatternValuesRaw as number[];
      const zPatternIndices =
        zPatternIndicesRaw.length > 0 && Array.isArray(zPatternIndicesRaw[0])
          ? (zPatternIndicesRaw as number[][])
          : ([zPatternIndicesRaw as number[]] as number[][]);

      return { zPatternIndices, zPatternValues };
    },
    []
  );

  // 解析 Top Activation 数据（与 circuit-visualization.tsx 保持一致）
  const parseTopActivationData = useCallback((camelData: any): TopActivationSample[] => {
    const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
    const allSamples: any[] = [];

    for (const group of sampleGroups) {
      if (group.samples && Array.isArray(group.samples)) {
        allSamples.push(...group.samples);
      }
    }

    // 查找包含 FEN 的样本并提取激活值
    const chessSamples: TopActivationSample[] = [];

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
                  let maxActivation = 0;

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
                        // 使用最大激活值
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
                    activationStrength: maxActivation,
                    activations: activationsArray,
                    ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
                    contextId: sample.contextIdx || sample.context_idx,
                    sampleIndex: sample.sampleIndex || 0,
                  });

                  break; // 找到一个有效 FEN 就跳出
                }
              }
            }
          }
        }
      }
    }

    // 按最大激活值排序并取前8个
    const topSamples = chessSamples
      .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
      .slice(0, 8);

    console.log('✅ 获取到 Top Activation 数据:', {
      totalChessSamples: chessSamples.length,
      topSamplesCount: topSamples.length
    });

    return topSamples;
  }, [normalizeZPattern]);

  // 获取 Top Activations
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId || !linkGraphData) return;

    setLoadingTopActivations(true);
    try {
      const node = linkGraphData.nodes.find(n => n.nodeId === nodeId);
      if (!node) {
        console.error('❌ 未找到节点:', nodeId);
        setTopActivations([]);
        return;
      }

      // 从 nodeId 中提取原始 layer 和 feature（nodeId 格式: {layer}_{pos}_{feature}）
      // 注意：node.layerIdx 现在是 Y 轴位置，不是原始 layer 值
      const parts = nodeId.split('_');
      const layer = parseInt(parts[0]) || 0; // 从 nodeId 中提取原始 layer
      const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(node.featureId) || (parts.length >= 3 ? parseInt(parts[2]) : 0));

      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';

      const dictionary = getDictionaryName(layer, isLorsa);

      console.log('🔍 获取 Top Activation 数据:', {
        nodeId,
        layer,
        featureIndex,
        dictionary,
        isLorsa,
      });

      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: 'GET',
          headers: {
            Accept: 'application/x-msgpack',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const decoded = await import('@msgpack/msgpack').then(module => 
        module.decode(new Uint8Array(arrayBuffer))
      );
      const camelcaseKeys = await import('camelcase-keys').then(module => 
        module.default
      );

      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ['sample_groups.samples.context'],
      }) as any;

      const topSamples = parseTopActivationData(camelData);
      setTopActivations(topSamples);

      console.log('✅ 获取到 Top Activation 数据:', {
        topSamplesCount: topSamples.length,
      });
    } catch (err) {
      console.error('❌ 获取 Top Activation 数据失败:', err);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [linkGraphData, getDictionaryName, parseTopActivationData]);

  // 处理节点点击
  const handleNodeClick = useCallback(async (node: Node, isMetaKey: boolean) => {
    if (isMetaKey) {
      // Meta/Ctrl 键：切换固定状态
      setPinnedIds(prev => {
        if (prev.includes(node.nodeId)) {
          return prev.filter(id => id !== node.nodeId);
        } else {
          return [...prev, node.nodeId];
        }
      });
    } else {
      // 普通点击：切换选中状态
      if (clickedId === node.nodeId) {
        setClickedId(null);
        setSelectedFeature(null);
        setTopActivations([]);
      } else {
        setClickedId(node.nodeId);
        
        // 获取 feature 信息（从 nodeId 中提取原始 layer，因为 node.layerIdx 现在是 Y 轴位置）
        const nodeType = node.feature_type?.toLowerCase();
        if ((nodeType === 'transcoder' || nodeType === 'lorsa' || nodeType === 'cross layer transcoder') && linkGraphData) {
          // 从 nodeId 中提取原始 layer（nodeId 格式: {layer}_{pos}_{feature}）
          const nodeIdParts = node.nodeId.split('_');
          const layer = parseInt(nodeIdParts[0]) || 0;
          const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(node.featureId) || 0);
          const isLorsa = nodeType === 'lorsa';
          
          try {
            const dictionary = getDictionaryName(layer, isLorsa);
            console.log('🔍 获取 feature 信息:', { layer, featureIndex, dictionary, isLorsa, nodeType });
            
            const { fetchFeature } = await import('@/utils/api');
            const feature = await fetchFeature(dictionary, layer, featureIndex);
            if (feature) {
              setSelectedFeature(feature);
              console.log('✅ 成功获取 feature 信息');
            } else {
              console.warn('⚠️ 未找到 feature 信息');
              setSelectedFeature(null);
            }
          } catch (err) {
            console.error('❌ 获取 feature 信息失败:', err);
            setSelectedFeature(null);
          }
        } else {
          console.log('⚠️ 不支持的节点类型:', nodeType);
          setSelectedFeature(null);
        }
        
        // 获取 Top Activations
        fetchTopActivations(node.nodeId);
        
        // 如果有 Top Activations，使用第一个作为默认 FEN
        // 这将在 fetchTopActivations 完成后通过 useEffect 处理
      }
    }
  }, [clickedId, fetchTopActivations, linkGraphData, getDictionaryName]);

  // 当 Top Activations 更新时，设置默认 FEN
  useEffect(() => {
    if (topActivations.length > 0 && !currentFen) {
      setCurrentFen(topActivations[0].fen);
      // 如果有激活数据，也设置
      if (topActivations[0].activations) {
        setFenActivationData({
          activations: topActivations[0].activations,
          zPatternIndices: topActivations[0].zPatternIndices,
          zPatternValues: topActivations[0].zPatternValues,
        });
      }
    }
  }, [topActivations, currentFen]);

  // 当点击节点改变时，清空 FEN 相关状态
  useEffect(() => {
    if (!clickedId) {
      setCurrentFen('');
      setFenActivationData(null);
    }
  }, [clickedId]);

  // 获取当前选中节点的 FEN 激活数据
  const fetchFenActivation = useCallback(async (fen: string) => {
    if (!clickedId || !linkGraphData || !fen.trim()) return;

    setLoadingFenActivation(true);
    try {
      const node = linkGraphData.nodes.find(n => n.nodeId === clickedId);
      if (!node) {
        console.error('❌ 未找到节点:', clickedId);
        return;
      }

      // 从 nodeId 中提取原始 layer（nodeId 格式: {layer}_{pos}_{feature}）
      const nodeIdParts = clickedId.split('_');
      const layer = parseInt(nodeIdParts[0]) || 0;
      const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(node.featureId) || 0);
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      const dictionary = getDictionaryName(layer, isLorsa);

      console.log('🔍 获取 FEN 激活数据:', { clickedId, fen, dictionary, layer, featureIndex });

      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({ fen: fen.trim() }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();

      // 解析激活数据
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

      setFenActivationData({
        activations,
        zPatternIndices,
        zPatternValues,
      });

      console.log('✅ 获取到 FEN 激活数据:', {
        hasActivations: !!activations,
        activationsLength: activations?.length,
        hasZPattern: !!zPatternIndices && !!zPatternValues,
      });
    } catch (err) {
      console.error('❌ 获取 FEN 激活数据失败:', err);
      setFenActivationData(null);
    } finally {
      setLoadingFenActivation(false);
    }
  }, [clickedId, linkGraphData, getDictionaryName]);

  // 处理连接的 feature 选择
  const handleConnectedFeaturesSelect = useCallback((_features: Feature[]) => {
    // 保留回调函数以保持接口兼容性
    // 连接的 feature 信息由 NodeConnections 组件内部管理
  }, []);

  const handleConnectedFeaturesLoading = useCallback((_loading: boolean) => {
    // 保留回调函数以保持接口兼容性
  }, []);

  // 处理节点悬停 - 参考 circuit-visualization.tsx 的实现
  // Only update if the hovered ID has actually changed
  // This prevents unnecessary re-renders and keeps hoveredId stable
  const handleNodeHover = useCallback((nodeId: string | null) => {
    if (nodeId !== hoveredId) {
      setHoveredId(nodeId);
    }
  }, [hoveredId]);

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Interaction Circuit Visualization</h1>
          <p className="text-gray-600 mt-2">
            上传包含 feature interaction 数据的 CSV 文件，点击节点查看详细的 feature 信息。
          </p>
        </div>

        {/* 文件上传区域 */}
        <div className="mb-6">
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-gray-400 transition-colors"
            onClick={() => document.getElementById('csv-file-input')?.click()}
          >
            <input
              id="csv-file-input"
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
            />
            <p className="text-gray-600">
              拖拽 CSV 文件到这里，或点击选择文件
            </p>
            {csvFile && (
              <p className="text-sm text-gray-500 mt-2">
                已加载: {csvFile.name}
              </p>
            )}
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        {/* 加载状态 */}
        {isLoading && (
          <div className="mb-6 p-4 bg-blue-100 border border-blue-400 text-blue-700 rounded">
            正在解析 CSV 文件...
          </div>
        )}

        {/* 图可视化 */}
        {linkGraphData && !isLoading && (
          <div className="space-y-6">
            {/* 主图区域 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <div className="bg-white rounded-lg shadow p-4" style={{ height: '800px' }}>
                  <LinkGraphContainer
                    data={linkGraphData}
                    onNodeClick={handleNodeClick}
                    onNodeHover={handleNodeHover}
                    clickedId={clickedId}
                    hoveredId={hoveredId}
                    pinnedIds={pinnedIds}
                    onFeatureSelect={setSelectedFeature}
                    onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                    onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                  />
                </div>
              </div>

              {/* 右侧面板：连接的 Feature */}
              <div className="space-y-4">
                <div className="bg-white rounded-lg shadow p-4" style={{ height: '800px', overflowY: 'auto' }}>
                  <FeatureConnections
                    data={linkGraphData}
                    clickedId={clickedId}
                    hoveredId={hoveredId}
                    onFeatureClick={(node) => handleNodeClick(node, false)}
                    onFeatureHover={handleNodeHover}
                    getDictionaryNameForNode={getDictionaryName}
                  />
                </div>
              </div>
            </div>

            {/* Feature 信息区域 - 显示在下方 */}
            {selectedFeature && clickedId && linkGraphData && (
              <div className="space-y-6">
                {/* 激活棋盘显示 */}
                <div className="bg-white rounded-lg shadow p-4">
                  <h3 className="text-lg font-semibold mb-4">Feature 激活棋盘</h3>
                  {(() => {
                    const node = linkGraphData.nodes.find(n => n.nodeId === clickedId);
                    if (!node) return null;

                    // 从 nodeId 中提取原始 layer（nodeId 格式: {layer}_{pos}_{feature}）
                    const nodeIdParts = clickedId.split('_');
                    const layer = parseInt(nodeIdParts[0]) || 0;
                    const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(node.featureId) || 0);
                    const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
                    const dictionary = getDictionaryName(layer, isLorsa);

                    return (
                      <div className="space-y-4">
                        {/* CustomFenInput 组件 */}
                        <CustomFenInput
                          dictionary={dictionary}
                          featureIndex={featureIndex}
                          disabled={false}
                          className="mb-4"
                        />

                        {/* 加载状态 */}
                        {loadingFenActivation && (
                          <div className="text-center py-4">
                            <div className="flex items-center justify-center space-x-2">
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                              <span className="text-sm text-gray-600">正在加载激活数据...</span>
                            </div>
                          </div>
                        )}

                        {/* 如果已经加载了激活数据，显示棋盘 */}
                        {currentFen && fenActivationData && !loadingFenActivation && (
                          <div className="flex justify-center">
                            <div className="space-y-2">
                              <div className="text-center text-sm text-gray-600 mb-2">
                                FEN: {currentFen}
                              </div>
                              <ChessBoard
                                fen={currentFen}
                                size="medium"
                                showCoordinates={true}
                                activations={fenActivationData.activations}
                                zPatternIndices={fenActivationData.zPatternIndices}
                                zPatternValues={fenActivationData.zPatternValues}
                                flip_activation={currentFen.includes(' b ')}
                                autoFlipWhenBlack={true}
                                analysisName={`Feature #${featureIndex} ${isLorsa ? 'LoRSA' : 'TC'}`}
                              />
                            </div>
                          </div>
                        )}

                        {/* Top Activations 选择 */}
                        {loadingTopActivations ? (
                          <div className="mt-4 text-center py-4">
                            <div className="flex items-center justify-center space-x-2">
                              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                              <span className="text-sm text-gray-600">正在加载 Top Activations...</span>
                            </div>
                          </div>
                        ) : topActivations.length > 0 ? (
                          <div className="mt-4">
                            <h4 className="text-md font-semibold mb-2">Top Activations (点击选择 FEN)</h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              {topActivations.map((sample, idx) => (
                                <div
                                  key={idx}
                                  className={`border rounded p-2 cursor-pointer transition-colors ${
                                    currentFen === sample.fen
                                      ? 'border-blue-500 bg-blue-50'
                                      : 'border-gray-300 hover:border-gray-400'
                                  }`}
                                  onClick={() => {
                                    setCurrentFen(sample.fen);
                                    if (sample.activations) {
                                      setFenActivationData({
                                        activations: sample.activations,
                                        zPatternIndices: sample.zPatternIndices,
                                        zPatternValues: sample.zPatternValues,
                                      });
                                    } else {
                                      // 如果没有激活数据，从后端获取
                                      fetchFenActivation(sample.fen);
                                    }
                                  }}
                                >
                                  <div className="text-center mb-2">
                                    <div className="text-xs font-medium text-gray-700">Top #{idx + 1}</div>
                                    <div className="text-xs text-gray-500">
                                      激活: {sample.activationStrength.toFixed(3)}
                                    </div>
                                  </div>
                                  <ChessBoard
                                    fen={sample.fen}
                                    size="small"
                                    showCoordinates={false}
                                    activations={sample.activations}
                                    flip_activation={sample.fen.includes(' b ')}
                                    autoFlipWhenBlack={true}
                                  />
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    );
                  })()}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 统计信息 */}
        {linkGraphData && !isLoading && (
          <div className="mt-6 p-4 bg-gray-100 rounded">
            <p className="text-sm text-gray-600">
              节点数: {linkGraphData.nodes.length} | 边数: {linkGraphData.links.length}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
