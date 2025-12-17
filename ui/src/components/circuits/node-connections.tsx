import React, { useCallback, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Node, Link as LinkType, LinkGraphData } from './link-graph/types';
import { extractLayerAndFeature } from './link-graph/utils';
import { fetchFeature, getDictionaryName } from "@/utils/api";
import { Feature } from "@/types/feature";

interface NodeConnectionsProps {
  data: LinkGraphData;
  clickedId: string | null;
  hoveredId: string | null;
  pinnedIds: string[];
  hiddenIds: string[];
  onFeatureClick: (node: Node, isMetaKey: boolean) => void;
  onFeatureSelect: (feature: Feature | null) => void;
  onFeatureHover: (nodeId: string | null) => void;
}

interface ConnectionSection {
  title: string;
  nodes: Node[];
}

interface ConnectionType {
  id: 'input' | 'output';
  title: string;
  sections: ConnectionSection[];
}

export const NodeConnections: React.FC<NodeConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  pinnedIds,
  hiddenIds,
  onFeatureClick,
  onFeatureSelect,
  onFeatureHover,
}) => {  

  // Memoize the clicked node to avoid re-finding it on every render
  const clickedNode = useMemo(() => 
    data.nodes.find(node => node.nodeId === clickedId), 
    [data.nodes, clickedId]
  );

  // Memoize the connection types computation - this is expensive and should only recalculate when relevant data changes
  const connectionTypes = useMemo((): ConnectionType[] => {
    if (!clickedNode || !clickedNode.sourceLinks || !clickedNode.targetLinks) {
      return [];
    }
    
    // Input features: nodes that have links TO the clicked node
    const inputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId &&
      node.sourceLinks &&
      node.sourceLinks.some(link => link.target === clickedNode.nodeId)
    );
    
    // Output features: nodes that the clicked node has links TO
    const outputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId &&
      clickedNode.sourceLinks &&
      clickedNode.sourceLinks.some(link => link.target === node.nodeId)
    );

    // 辅助：构建“节点-链接-来源文件”条目
    type PerFileEntry = { node: Node; fileIndex: number; weight: number; pctInput: number };
    const toPerFileEntries = (nodes: Node[], dir: 'input' | 'output'): PerFileEntry[] => {
      const entries: PerFileEntry[] = [];
      for (const n of nodes) {
        const link = dir === 'input'
          ? n.sourceLinks?.find(l => l.target === clickedNode.nodeId)
          : clickedNode.sourceLinks?.find(l => l.target === n.nodeId);
        if (!link) continue;
        // 优先使用每文件值；若不存在则回退到总体值
        if (link.sources && link.weightsBySource && link.pctBySource) {
          for (const fi of link.sources) {
            const w = link.weightsBySource[fi];
            const p = link.pctBySource[fi];
            if (w === undefined || p === undefined) continue;
            entries.push({ node: n, fileIndex: fi, weight: w, pctInput: p });
          }
        } else if (link.weight !== undefined) {
          entries.push({ node: n, fileIndex: (n.sourceIndex ?? 0), weight: link.weight, pctInput: link.pctInput || 0 });
        }
      }
      return entries;
    };

    const inputEntries = toPerFileEntries(inputNodes, 'input');
    const outputEntries = toPerFileEntries(outputNodes, 'output');

    const buildSections = (entries: PerFileEntry[]) => ['Positive', 'Negative'].map(title => {
      // 先筛选正负
      const filtered = entries.filter(e => (title === 'Positive' ? e.weight > 0 : e.weight < 0));

      // 合并相同位置的 error 节点（同 nodeId 只保留一个，取绝对值更大的权重）
      const mergedErrorByNodeId = new Map<string, PerFileEntry>();
      for (const e of filtered) {
        const isError = typeof e.node.feature_type === 'string' && e.node.feature_type.toLowerCase().includes('error');
        if (!isError) continue;
        const key = e.node.nodeId; // nodeId 唯一标识同层同位置同feature
        const prev = mergedErrorByNodeId.get(key);
        if (!prev || Math.abs(e.weight) > Math.abs(prev.weight)) {
          // 仅保留一个代表条目（移除文件索引以强调合并）
          mergedErrorByNodeId.set(key, { node: e.node, fileIndex: -1, weight: e.weight, pctInput: e.pctInput });
        }
      }

      // 非 error 的条目保持原条目
      const nonError = filtered.filter(e => {
        const isError = typeof e.node.feature_type === 'string' && e.node.feature_type.toLowerCase().includes('error');
        return !isError;
      });

      // 重新组成列表：非 error + 合并后的 error
      const normalized: PerFileEntry[] = [
        ...nonError,
        ...Array.from(mergedErrorByNodeId.values()),
      ];

      // 自定义排序：同层同位置，shared 优先；其次各文件按文件索引升序；再按绝对权重降序；error 放最后
      normalized.sort((a, b) => {
        const aLayer = a.node.layerIdx;
        const bLayer = b.node.layerIdx;
        const aCtx = a.node.ctx_idx;
        const bCtx = b.node.ctx_idx;
        const samePos = aLayer === bLayer && aCtx === bCtx;
        if (samePos) {
          const aIsError = typeof a.node.feature_type === 'string' && a.node.feature_type.toLowerCase().includes('error');
          const bIsError = typeof b.node.feature_type === 'string' && b.node.feature_type.toLowerCase().includes('error');
          if (aIsError !== bIsError) return aIsError ? 1 : -1; // error 最后

          const aShared = (a.node.sourceIndices && a.node.sourceIndices.length > 1) ? 1 : 0;
          const bShared = (b.node.sourceIndices && b.node.sourceIndices.length > 1) ? 1 : 0;
          if (aShared !== bShared) return bShared - aShared; // 共有优先

          // 都是 per-file 的情况：按文件顺序
          if (aShared === 0 && bShared === 0 && !aIsError && !bIsError) {
            const aFile = a.fileIndex;
            const bFile = b.fileIndex;
            if (aFile !== bFile) return aFile - bFile; // 文件索引升序
          }

          // 次序相同则按绝对权重降序
          return Math.abs(b.weight) - Math.abs(a.weight);
        }
        // 不同位置，按绝对权重降序
        return Math.abs(b.weight) - Math.abs(a.weight);
      });

      // 映射到渲染节点（保留 per-file 的文件索引；合并后的 error fileIndex 为 -1，不影响灰点）
      const sectionNodes = normalized.map(e => ({
        ...e.node,
        // @ts-ignore
        __fileIndex: e.fileIndex >= 0 ? e.fileIndex : undefined,
        // @ts-ignore
        __weight: e.weight,
        // @ts-ignore
        __pct: e.pctInput,
      }) as Node);

      return { title, nodes: sectionNodes };
    });

    return [
      { id: 'input', title: 'Input Features', sections: buildSections(inputEntries) },
      { id: 'output', title: 'Output Features', sections: buildSections(outputEntries) },
    ];
  }, [data.nodes, clickedNode?.nodeId, clickedNode?.sourceLinks, clickedNode?.targetLinks]);

  // Memoize the formatFeatureId function to avoid recreating it on every render
  const formatFeatureId = useMemo(() => (node: Node): string => {
    if (node.feature_type === 'cross layer transcoder') {
      const layerIdx = Math.floor(node.layerIdx / 2) - 1;
      const featureId = node.id.split('_')[1];
      return `M${layerIdx}#${featureId}@${node.ctx_idx}`;
    } else if (node.feature_type === 'lorsa') {
      const layerIdx = Math.floor(node.layerIdx / 2);
      const featureId = node.id.split('_')[1];
      return `A${layerIdx}#${featureId}@${node.ctx_idx}`;
    } else if (node.feature_type === 'embedding') {
      return `Emb@${node.ctx_idx}`;
    } else if (node.feature_type === 'mlp reconstruction error') {
      return `M${Math.floor(node.layerIdx / 2) - 1}Error@${node.ctx_idx}`;
    } else if (node.feature_type === 'lorsa error') {
      return `A${Math.floor(node.layerIdx / 2)}Error@${node.ctx_idx}`;
    }
    return ' ';
  }, []);

  const handleNodeClick = useCallback(async (nodeId: string, metaKey: boolean) => {
    const node = data.nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Always call parent handler first to update global state
    onFeatureClick?.(node, metaKey);
    
    // If not meta key (not pinning), handle feature selection
    if (!metaKey) {
      if (clickedId === nodeId) {
        // Deselecting the same node
        onFeatureSelect?.(null);
      } else {
        // Only fetch feature data for supported node types
        if (node.feature_type === 'cross layer transcoder' || node.feature_type === 'lorsa') {
          const layerAndFeature = extractLayerAndFeature(nodeId);
          if (layerAndFeature) {
            const { layer, featureId, isLorsa } = layerAndFeature;
            const dictionaryName = getDictionaryName(data.metadata, layer, isLorsa);

            if (dictionaryName) {
              try {
                const feature = await fetchFeature(dictionaryName, layer, featureId);
                if (feature) {
                  onFeatureSelect?.(feature);
                } else {
                  onFeatureSelect?.(null);
                }
              } catch (error) {
                console.error('Failed to fetch feature:', error);
                onFeatureSelect?.(null);
              }
            } else {
              onFeatureSelect?.(null);
            }
          } else {
            onFeatureSelect?.(null);
          }
        } else {
          // For unsupported node types, clear the selection
          onFeatureSelect?.(null);
        }
      }
    }
  }, [onFeatureClick, clickedId, data.nodes, data.metadata, onFeatureSelect]);

  // Memoize the feature row renderer to avoid recreating the function on every render
  const renderFeatureRow = useMemo(() => (node: Node, type: 'input' | 'output') => {
    if (!clickedNode) return null;
    
    // 从节点的临时字段读取该行对应文件与权重
    const anyNode = node as any;
    const rowFileIndex: number | undefined = anyNode.__fileIndex;
    const rowWeight: number | undefined = anyNode.__weight;
    const rowPct: number | undefined = anyNode.__pct;

    // 找到链接（用于回退）
    const link = type === 'input' 
      ? node.sourceLinks?.find(link => link.target === clickedNode.nodeId)
      : clickedNode.sourceLinks?.find(link => link.target === node.nodeId);

    if (!rowWeight && (!link || link.weight === undefined)) return null;

    const weight = rowWeight !== undefined ? rowWeight : (link!.weight as number);
    const pctInput = rowPct !== undefined ? rowPct : (link!.pctInput || 0);
    const isPinned = pinnedIds.includes(node.nodeId);
    const isHidden = hiddenIds.includes(node.featureId);
    const isHovered = node.nodeId === hoveredId;
    const isClicked = node.nodeId === clickedId;

    // 文件颜色与名称（优先使用该行绑定的文件索引）
    const chosenIndex = (rowFileIndex !== undefined) ? rowFileIndex : (node.sourceIndex ?? (node.sourceIndices?.[0] ?? 0));
    const sourceFileNames = data.metadata.sourceFileNames || [];
    const colorByIndex = (i: number) => {
      const palette = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B"]; // 与可视化一致
      return palette[i % palette.length];
    };
    const sourceTitle = sourceFileNames[chosenIndex] || `#${(chosenIndex ?? 0) + 1}`;
    const isErrorNode = typeof node.feature_type === 'string' && node.feature_type.toLowerCase().includes('error');
    const dotColor = isErrorNode ? '#95a5a6' : colorByIndex(chosenIndex || 0);

    return (
      <div
        key={`${node.nodeId}-${chosenIndex}`}
        className={`feature-row p-1 border rounded cursor-pointer transition-colors ${
          isPinned ? 'bg-yellow-100 border-yellow-300' : 'bg-gray-50 border-gray-200'
        } ${isHidden ? 'opacity-50' : ''} ${isHovered ? 'ring-2 ring-blue-300' : ''} ${
          isClicked ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => {
          onFeatureClick(node, false);
          handleNodeClick(node.nodeId, false);
        }}
        onMouseEnter={() => onFeatureHover(node.nodeId)}
        onMouseLeave={() => onFeatureHover(null)}
        title={`Source: ${sourceTitle}`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span
              className="inline-block rounded-full"
              style={{ width: 8, height: 8, backgroundColor: dotColor }}
              title={sourceTitle}
            />
            <span className="text-sm font-mono text-gray-600">
              {formatFeatureId(node)}
            </span>
            <span className="text-sm font-medium">
              {node.localClerp || node.remoteClerp || ''}
            </span>
          </div>
          <div className="text-right">
            <div className="text-sm font-mono">
              {weight > 0 ? '+' : ''}{weight.toFixed(3)}
            </div>
            <div className="text-xs text-gray-500">
              {pctInput.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    );
  }, [clickedNode, pinnedIds, hiddenIds, hoveredId, clickedId, onFeatureClick, formatFeatureId, data.metadata.sourceFileNames]);

  // Memoize the header styling to avoid recalculating classes on every render
  const headerClassName = useMemo(() => 
    `header-top-row section-title mb-3 cursor-pointer p-2 rounded-lg border ${
      clickedNode && pinnedIds.includes(clickedNode.nodeId)
        ? 'bg-yellow-50 border-yellow-200 text-yellow-800' 
        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
    }`, 
    [pinnedIds, clickedNode?.nodeId]
  );

  // 解析节点信息用于跳转到 global weight 页面（必须在早期返回之前调用）
  const globalWeightParams = useMemo(() => {
    if (!clickedNode) return null;
    
    // 从 nodeId 解析 layer 和 feature
    const parts = clickedNode.nodeId.split('_');
    if (parts.length < 2) return null;
    
    const rawLayer = parseInt(parts[0]) || 0;
    const layerIdx = Math.floor(rawLayer / 2); // 除以2得到实际模型层数
    const featureIdx = parseInt(parts[1]) || 0;
    const isLorsa = clickedNode.feature_type?.toLowerCase() === 'lorsa';
    const featureType = isLorsa ? 'lorsa' : 'tc';
    
    // 从 localStorage 读取 sae_combo_id
    const saeComboId = typeof window !== 'undefined' 
      ? window.localStorage.getItem('bt4_sae_combo_id') 
      : null;
    
    const params = new URLSearchParams({
      feature_type: featureType,
      layer_idx: layerIdx.toString(),
      feature_idx: featureIdx.toString(),
    });
    
    if (saeComboId) {
      params.append('sae_combo_id', saeComboId);
    }
    
    return params.toString();
  }, [clickedNode]);

  // Early return after all hooks have been called
  if (!clickedNode) {
    return (
      <div className="node-connections flex flex-col h-full overflow-y-auto">
        <div className="header-top-row section-title mb-3">
          Click a feature on the left for details
        </div>
      </div>
    );
  }

  return (
    <div className="node-connections flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div 
        className={headerClassName}
      >
        <span className="inline-block mr-2 font-mono tabular-nums w-20 text-sm">
          {formatFeatureId(clickedNode)}
        </span>
        <span className="feature-title font-medium text-sm">
          {clickedNode.localClerp || clickedNode.remoteClerp || ''}
        </span>
      </div>

      {/* Connections */}
      <div className="connections flex-1 flex flex-col gap-3 min-h-0">
        {/* Global Weight 跳转按钮 - 显示在 Input Features 和 Output Features 上方 */}
        {globalWeightParams && (
          <div className="w-full flex-shrink-0">
            <Link
              to={`/global-weight?${globalWeightParams}`}
              className="inline-flex items-center justify-center w-full px-3 py-2 bg-purple-500 text-white text-sm font-medium rounded-md hover:bg-purple-600 transition-colors"
              title="查看该特征的全局权重分析"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              查看全局权重
            </Link>
          </div>
        )}
        
        {/* Input Features 和 Output Features */}
        <div className="flex-1 flex gap-5 min-h-0">
          {connectionTypes.map(type => (
            <div
              key={type.id}
              className={`features flex-1 flex flex-col min-h-0 ${type.id === 'output' ? 'output' : 'input'}`}
            >
              <div className="section-title text-lg font-semibold mb-2 text-gray-800 flex-shrink-0">
                {type.title}
              </div>
            
            <div className="effects space-y-2 flex-1 overflow-y-auto pr-2">
              {type.sections.map(section => (
                <div key={section.title} className="section">
                  <h4 className={`text-sm font-medium mb-1 px-2 py-1 rounded ${
                    section.title === 'Positive' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {section.title}
                  </h4>
                  <div className="space-y-1">
                    {section.nodes.map(node => renderFeatureRow(node, type.id))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
        </div>
      </div>
    </div>
  );
}; 