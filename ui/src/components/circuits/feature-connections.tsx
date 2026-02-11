import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { Node, LinkGraphData } from './link-graph/types';
import { fetchFeature } from '@/utils/api';
import { Feature } from '@/types/feature';

// This file is for Interaction Circuit Page, which is used to display the feature connections of the interaction circuit.

interface FeatureConnectionsProps {
  data: LinkGraphData;
  clickedId: string | null;
  hoveredId: string | null;
  onFeatureClick: (node: Node) => void;
  onFeatureHover: (nodeId: string | null) => void;
  getDictionaryNameForNode: (layer: number, isLorsa: boolean) => string;
}

interface FeatureWithInterpretation {
  node: Node;
  interpretation: string | null;
  feature: Feature | null;
  loading: boolean;
  editing: boolean;
  editText: string;
}

export const FeatureConnections: React.FC<FeatureConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  onFeatureClick,
  onFeatureHover,
  getDictionaryNameForNode,
}) => {
  const [featureInterpretations, setFeatureInterpretations] = useState<Map<string, FeatureWithInterpretation>>(new Map());
  const [savingNodeId, setSavingNodeId] = useState<string | null>(null);
  const [syncingAllInterpretations, setSyncingAllInterpretations] = useState(false);
  const loadedNodesRef = useRef<Set<string>>(new Set());

  // 找到被点击的节点
  const clickedNode = useMemo(() => 
    data.nodes.find(node => node.nodeId === clickedId),
    [data.nodes, clickedId]
  );

  // 计算 Input Features 和 Output Features
  const { inputNodes, outputNodes } = useMemo(() => {
    if (!clickedNode) {
      return { inputNodes: [], outputNodes: [] };
    }

    // Input Features: 所有指向当前节点的节点（即targetLinks的source）
    // 先找到所有指向当前节点的链接的source节点ID
    const inputNodeIds = clickedNode.targetLinks?.map(link => link.source) || [];
    const inputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId &&
      inputNodeIds.includes(node.nodeId)
    );

    // Output Features: 当前节点指向的所有节点（即sourceLinks的target）
    // 先找到当前节点指向的所有链接的target节点ID
    const outputNodeIds = clickedNode.sourceLinks?.map(link => link.target) || [];
    const outputNodes = data.nodes.filter(node =>
      node.nodeId !== clickedNode.nodeId &&
      outputNodeIds.includes(node.nodeId)
    );

    return { inputNodes, outputNodes };
  }, [data.nodes, clickedNode]);

  // 格式化节点ID显示
  const formatNodeId = useCallback((node: Node): string => {
    const parts = node.nodeId.split('_');
    if (parts.length < 3) return node.nodeId;
    
    const layer = parseInt(parts[0]) || 0;
    const feature = parseInt(parts[2]) || 0;
    const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
    
    return `${isLorsa ? 'A' : 'M'}${layer}#${feature}@${parts[1]}`;
  }, []);

  // 获取节点的解释
  const fetchNodeInterpretation = useCallback(async (node: Node) => {
    const nodeId = node.nodeId;
    
    // 如果已经加载过，跳过
    if (loadedNodesRef.current.has(nodeId)) {
      return;
    }

    // 标记为正在加载
    loadedNodesRef.current.add(nodeId);

    // 设置加载状态
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      newMap.set(nodeId, {
        node,
        interpretation: null,
        feature: null,
        loading: true,
        editing: false,
        editText: '',
      });
      return newMap;
    });

    try {
      // 从nodeId中提取layer和feature
      const parts = nodeId.split('_');
      const layer = parseInt(parts[0]) || 0;
      const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(parts[2]) || 0);
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      
      const dictionary = getDictionaryNameForNode(layer, isLorsa);
      
      // fetchFeature需要analysisName（dictionary name），layer，和featureId
      // 注意：fetchFeature内部会处理{}替换，但我们传入的dictionary已经是完整名称
      const feature = await fetchFeature(dictionary, layer, featureIndex);
      
      const interpretation = feature?.interpretation?.text || null;
      
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        newMap.set(nodeId, {
          node,
          interpretation,
          feature,
          loading: false,
          editing: false,
          editText: interpretation || '',
        });
        return newMap;
      });
    } catch (error) {
      console.error('Failed to fetch feature:', error);
      loadedNodesRef.current.delete(nodeId); // 加载失败，允许重试
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        newMap.set(nodeId, {
          node,
          interpretation: null,
          feature: null,
          loading: false,
          editing: false,
          editText: '',
        });
        return newMap;
      });
    }
  }, [getDictionaryNameForNode]);

  // 当clickedId、inputNodes或outputNodes改变时，获取解释
  useEffect(() => {
    if (!clickedId) {
      loadedNodesRef.current.clear();
      setFeatureInterpretations(new Map());
      return;
    }
    
    // 获取被点击节点自身的解释
    if (clickedNode) {
      fetchNodeInterpretation(clickedNode);
    }
    
    // 获取输入和输出节点的解释
    const allNodes = [...inputNodes, ...outputNodes];
    allNodes.forEach(node => {
      fetchNodeInterpretation(node);
    });
  }, [inputNodes, outputNodes, clickedId, clickedNode, fetchNodeInterpretation]);

  // 保存解释
  const saveInterpretation = useCallback(async (nodeId: string) => {
    const item = featureInterpretations.get(nodeId);
    if (!item || !item.feature) return;

    setSavingNodeId(nodeId);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${item.feature.dictionaryName}/features/${item.feature.featureIndex}/interpret?type=custom&custom_interpretation=${encodeURIComponent(item.editText)}`,
        {
          method: 'POST',
        }
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      // 更新本地状态
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(nodeId);
        if (existing) {
          newMap.set(nodeId, {
            ...existing,
            interpretation: item.editText,
            editing: false,
          });
        }
        return newMap;
      });
    } catch (error) {
      console.error('Failed to save interpretation:', error);
      alert(`保存失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setSavingNodeId(null);
    }
  }, [featureInterpretations]);

  // 开始编辑
  const startEditing = useCallback((nodeId: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editing: true,
          editText: existing.interpretation || '',
        });
      }
      return newMap;
    });
  }, []);

  // 取消编辑
  const cancelEditing = useCallback((nodeId: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editing: false,
          editText: existing.interpretation || '',
        });
      }
      return newMap;
    });
  }, []);

  // 更新编辑文本
  const updateEditText = useCallback((nodeId: string, text: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editText: text,
        });
      }
      return newMap;
    });
  }, []);

  // 批量同步所有节点的解释
  const syncAllInterpretations = useCallback(async () => {
    if (!data || !data.nodes.length) {
      alert('⚠️ 没有可用的节点数据');
      return;
    }

    setSyncingAllInterpretations(true);
    try {
      // 清除所有已加载节点的缓存
      loadedNodesRef.current.clear();
      setFeatureInterpretations(new Map());

      // 批量获取所有节点的解释（使用 Promise.all 并行请求，但限制并发数）
      const allNodes = data.nodes;
      const batchSize = 10; // 每批处理10个节点
      let foundCount = 0;
      let notFoundCount = 0;

      console.log('🔄 开始批量同步所有节点的解释:', {
        totalNodes: allNodes.length,
      });

      // 分批处理，避免过多并发请求
      for (let i = 0; i < allNodes.length; i += batchSize) {
        const batch = allNodes.slice(i, i + batchSize);
        await Promise.all(
          batch.map(async (node) => {
            try {
              const parts = node.nodeId.split('_');
              const layer = parseInt(parts[0]) || 0;
              const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(parts[2]) || 0);
              const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
              
              const dictionary = getDictionaryNameForNode(layer, isLorsa);
              const feature = await fetchFeature(dictionary, layer, featureIndex);
              
              if (feature?.interpretation?.text) {
                foundCount++;
                setFeatureInterpretations(prev => {
                  const newMap = new Map(prev);
                  newMap.set(node.nodeId, {
                    node,
                    interpretation: feature.interpretation!.text!,
                    feature,
                    loading: false,
                    editing: false,
                    editText: feature.interpretation!.text!,
                  });
                  return newMap;
                });
              } else {
                notFoundCount++;
                setFeatureInterpretations(prev => {
                  const newMap = new Map(prev);
                  newMap.set(node.nodeId, {
                    node,
                    interpretation: null,
                    feature,
                    loading: false,
                    editing: false,
                    editText: '',
                  });
                  return newMap;
                });
              }
            } catch (error) {
              console.error(`❌ 获取节点 ${node.nodeId} 的解释失败:`, error);
              notFoundCount++;
            }
          })
        );

        // 显示进度
        console.log(`✅ 已处理 ${Math.min(i + batchSize, allNodes.length)}/${allNodes.length} 个节点`);
      }

      console.log('✅ 批量同步完成:', {
        total: allNodes.length,
        found: foundCount,
        notFound: notFoundCount
      });

      alert(`✅ 同步完成！找到 ${foundCount} 个解释，${notFoundCount} 个未找到。\n\n提示：解释已从后端MongoDB同步，请点击节点查看。`);
    } catch (error) {
      console.error('❌ 批量同步解释失败:', error);
      alert(`❌ 同步失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setSyncingAllInterpretations(false);
    }
  }, [data, getDictionaryNameForNode]);

  // 渲染feature行
  const renderFeatureRow = useCallback((node: Node) => {
    const item = featureInterpretations.get(node.nodeId);
    const isHovered = node.nodeId === hoveredId;
    const isEditing = item?.editing || false;
    const interpretation = item?.interpretation || null;
    const loading = item?.loading || false;
    const editText = item?.editText || '';

    return (
      <div
        key={node.nodeId}
        className={`p-2 border rounded mb-2 cursor-pointer transition-colors ${
          isHovered ? 'bg-blue-50 border-blue-300' : 'bg-gray-50 border-gray-200'
        }`}
        onClick={() => onFeatureClick(node)}
        onMouseEnter={() => onFeatureHover(node.nodeId)}
        onMouseLeave={() => onFeatureHover(null)}
      >
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="text-sm font-mono text-gray-700 mb-1">
              {formatNodeId(node)}
            </div>
            {loading ? (
              <div className="text-xs text-gray-500">加载中...</div>
            ) : isEditing ? (
              <div className="mt-2">
                <textarea
                  value={editText}
                  onChange={(e) => updateEditText(node.nodeId, e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded text-sm"
                  rows={2}
                  onClick={(e) => e.stopPropagation()}
                />
                <div className="flex gap-2 mt-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      saveInterpretation(node.nodeId);
                    }}
                    disabled={savingNodeId === node.nodeId}
                    className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 disabled:opacity-50"
                  >
                    {savingNodeId === node.nodeId ? '保存中...' : '保存'}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      cancelEditing(node.nodeId);
                    }}
                    className="px-2 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
                  >
                    取消
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-xs text-gray-600 mt-1">
                {interpretation || '无解释'}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    startEditing(node.nodeId);
                  }}
                  className="ml-2 text-blue-600 hover:text-blue-800 underline"
                >
                  编辑
                </button>
              </div>
            )}
          </div>
          <div className="text-right ml-2">
            <div className="text-sm font-mono text-gray-500">1.000</div>
            <div className="text-xs text-gray-400">100.0%</div>
          </div>
        </div>
      </div>
    );
  }, [featureInterpretations, hoveredId, formatNodeId, onFeatureClick, onFeatureHover, saveInterpretation, cancelEditing, startEditing, updateEditText, savingNodeId]);

  // 获取被点击节点的解释信息（必须在早期返回之前）
  const clickedNodeInterpretation = useMemo(() => {
    if (!clickedNode) return null;
    return featureInterpretations.get(clickedNode.nodeId);
  }, [clickedNode, featureInterpretations]);

  // 渲染被点击节点的解释编辑区域（必须在早期返回之前）
  const renderClickedNodeInterpretation = useCallback(() => {
    if (!clickedNode || !clickedNodeInterpretation) {
      return (
        <div className="text-xs text-gray-500 mt-2">加载中...</div>
      );
    }

    const { interpretation, editing, editText, loading } = clickedNodeInterpretation;
    const isEditing = editing || false;
    const currentText = editText || '';
    const isSaving = savingNodeId === clickedNode.nodeId;

    if (loading) {
      return (
        <div className="text-xs text-gray-500 mt-2">加载中...</div>
      );
    }

    if (isEditing) {
      return (
        <div className="mt-2">
          <textarea
            value={currentText}
            onChange={(e) => updateEditText(clickedNode.nodeId, e.target.value)}
            className="w-full p-2 border border-gray-300 rounded text-sm"
            rows={3}
            placeholder="输入feature解释..."
          />
          <div className="flex gap-2 mt-1">
            <button
              onClick={() => saveInterpretation(clickedNode.nodeId)}
              disabled={isSaving}
              className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 disabled:opacity-50"
            >
              {isSaving ? '保存中...' : '保存'}
            </button>
            <button
              onClick={() => cancelEditing(clickedNode.nodeId)}
              className="px-3 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
            >
              取消
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-2">
        <div className="text-xs text-gray-600">
          {interpretation || '无解释'}
        </div>
        <button
          onClick={() => startEditing(clickedNode.nodeId)}
          className="mt-1 text-xs text-blue-600 hover:text-blue-800 underline"
        >
          编辑
        </button>
      </div>
    );
  }, [clickedNode, clickedNodeInterpretation, savingNodeId, saveInterpretation, cancelEditing, startEditing, updateEditText]);

  // 早期返回必须在所有hooks之后
  if (!clickedNode) {
    return (
      <div className="flex flex-col h-full overflow-y-auto">
        <div className="text-gray-500 text-center py-8">
          点击左侧节点查看连接的features
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="mb-4 p-2 bg-gray-100 rounded">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-mono">{formatNodeId(clickedNode)}</div>
          <button
            onClick={syncAllInterpretations}
            disabled={syncingAllInterpretations}
            className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-1"
            title="从后端MongoDB批量同步所有feature的解释"
          >
            {syncingAllInterpretations ? (
              <>
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                同步中...
              </>
            ) : (
              <>
                <span>🔄</span>
                同步所有解释
              </>
            )}
          </button>
        </div>
        <div className="text-sm text-gray-600 mb-2">{clickedNode.localClerp || ''}</div>
        <div className="border-t pt-2 mt-2">
          <div className="text-xs font-semibold text-gray-700 mb-1">解释:</div>
          {renderClickedNodeInterpretation()}
        </div>
      </div>

      {/* Input Features 和 Output Features */}
      <div className="flex-1 flex gap-4">
        {/* Input Features */}
        <div className="flex-1">
          <div className="text-lg font-semibold mb-2">Input Features</div>
          <div className="space-y-2">
            {inputNodes.length === 0 ? (
              <div className="text-sm text-gray-500 text-center py-4">无输入features</div>
            ) : (
              inputNodes.map(node => renderFeatureRow(node))
            )}
          </div>
        </div>

        {/* Output Features */}
        <div className="flex-1">
          <div className="text-lg font-semibold mb-2">Output Features</div>
          <div className="space-y-2">
            {outputNodes.length === 0 ? (
              <div className="text-sm text-gray-500 text-center py-4">无输出features</div>
            ) : (
              outputNodes.map(node => renderFeatureRow(node))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

