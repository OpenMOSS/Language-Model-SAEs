import React, { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Loader2 } from 'lucide-react';
import { LinkGraphContainer } from './link-graph-container';
import { NodeConnections } from './node-connections';
import { FeatureCard } from '@/components/feature/feature-card';
import { Feature } from '@/types/feature';
import { transformCircuitData } from './link-graph/utils';

interface CircuitTracingProps {
  gameFen: string; // move之前的FEN
  previousFen?: string | null; // 上一个FEN状态
  currentFen?: string; // 当前FEN状态
  gameHistory: string[];
  lastMove?: string | null; // 最后一个移动
  onCircuitTraceStart?: () => void;
  onCircuitTraceEnd?: () => void;
  isTracing?: boolean;
}

export const CircuitTracing: React.FC<CircuitTracingProps> = ({
  gameFen,
  previousFen,
  currentFen,
  gameHistory,
  lastMove,
  onCircuitTraceStart,
  onCircuitTraceEnd,
  isTracing = false,
}) => {
  const [circuitTraceResult, setCircuitTraceResult] = useState<any>(null);
  const [circuitVisualizationData, setCircuitVisualizationData] = useState<any>(null);
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  const [hiddenNodeIds, setHiddenNodeIds] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [, setIsLoadingConnectedFeatures] = useState(false);

  // 新增：handleCircuitTrace函数
  const handleCircuitTraceResult = useCallback((result: any) => {
    if (result && result.nodes) {
      try {
        const transformedData = transformCircuitData(result);
        setCircuitVisualizationData(transformedData);
        setCircuitTraceResult(result);
      } catch (error) {
        console.error('Circuit数据转换失败:', error);
        alert('Circuit数据转换失败: ' + (error instanceof Error ? error.message : '未知错误'));
      }
    }
  }, []);

  // 新增：处理节点点击 - 修复参数传递
  const handleNodeClick = useCallback((node: any, isMetaKey: boolean) => {
    const nodeId = node.nodeId || node.id;
    console.log('🔍 节点被点击:', { nodeId, isMetaKey, currentClickedId: clickedNodeId, node });
    
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedNodeIds.includes(nodeId)
        ? pinnedNodeIds.filter(id => id !== nodeId)
        : [...pinnedNodeIds, nodeId];
      setPinnedNodeIds(newPinnedIds);
      console.log('📌 切换固定状态:', newPinnedIds);
    } else {
      // Set clicked node
      const newClickedId = nodeId === clickedNodeId ? null : nodeId;
      setClickedNodeId(newClickedId);
      console.log('🎯 设置选中节点:', newClickedId);
      
      // 清除之前的特征选择
      if (newClickedId === null) {
        setSelectedFeature(null);
        setConnectedFeatures([]);
      }
    }
  }, [clickedNodeId, pinnedNodeIds]);

  // 新增：处理节点悬停 - 修复参数传递
  const handleNodeHover = useCallback((nodeId: string | null) => {
    if (nodeId !== hoveredNodeId) {
      setHoveredNodeId(nodeId);
    }
  }, [hoveredNodeId]);

  // 新增：处理特征选择
  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  // 新增：处理连接特征选择
  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
    setIsLoadingConnectedFeatures(false);
  }, []);

  // 新增：处理连接特征加载
  const handleConnectedFeaturesLoading = useCallback((loading: boolean) => {
    setIsLoadingConnectedFeatures(loading);
  }, []);

  // 修改handleCircuitTrace函数
  const handleCircuitTrace = useCallback(async () => {
    if (!lastMove) {
      alert('请先走一步棋，再进行Circuit Trace');
      return;
    }
    
    // 使用传入的lastMove，它应该是UCI格式
    const moveUci = lastMove;
    
    console.log('🔍 Circuit Trace 参数:', {
      fen: gameFen, // move之前的FEN
      move_uci: moveUci,
      current_fen: currentFen,
      game_history: gameHistory
    });
    
    onCircuitTraceStart?.();
    
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          fen: gameFen, // 使用move之前的FEN
          move_uci: moveUci 
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        handleCircuitTraceResult(data);
      } else {
        const errorText = await response.text();
        console.error('Circuit trace API调用失败:', response.status, response.statusText, errorText);
        alert('Circuit trace失败: ' + errorText);
      }
    } catch (error) {
      console.error('Circuit trace出错:', error);
      alert('Circuit trace出错: ' + (error instanceof Error ? error.message : '未知错误'));
    } finally {
      onCircuitTraceEnd?.();
    }
  }, [gameFen, currentFen, lastMove, gameHistory, onCircuitTraceStart, onCircuitTraceEnd, handleCircuitTraceResult]);

  // 新增：保存原始graph JSON（与后端create_graph_files一致的数据结构）
  const handleSaveGraphJson = useCallback(() => {
    try {
      const raw = circuitTraceResult || circuitVisualizationData;
      if (!raw) {
        alert('没有可保存的图数据');
        return;
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const slug = raw?.metadata?.slug || 'circuit_trace';
      // 从当前FEN解析全回合数（第6段），若解析失败则回退为基于历史长度估算
      const fenParts = gameFen.split(' ');
      const fullmove = fenParts.length >= 6 && !Number.isNaN(parseInt(fenParts[5]))
        ? parseInt(fenParts[5])
        : Math.max(1, Math.ceil(gameHistory.length / 2));
      const fileName = `${slug}_m${fullmove}_${timestamp}.json`;
      const jsonString = JSON.stringify(raw, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('保存JSON失败:', error);
      alert('保存JSON失败');
    }
  }, [circuitTraceResult, circuitVisualizationData, gameFen, gameHistory]);

  return (
    <div className="space-y-6">
      {/* Circuit Trace 控制面板 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Circuit Trace 分析</span>
            <div className="flex gap-2">
              <Button
                onClick={handleCircuitTrace}
                disabled={isTracing || gameHistory.length === 0}
                variant={isTracing ? 'destructive' : 'default'}
                size="sm"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Circuit Trace中...
                  </>
                ) : (
                  'Circuit Trace'
                )}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">分析FEN (移动前):</span>
                <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 break-all border border-blue-200">
                  {gameFen}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">当前FEN (移动后):</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 break-all border border-green-200">
                  {currentFen || gameFen}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">分析的移动:</span>
                <div className="font-mono text-xs bg-yellow-50 p-2 rounded mt-1 border border-yellow-200">
                  {lastMove || '暂无移动'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">移动历史:</span>
                <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1">
                  {gameHistory.length > 0 ? gameHistory.join(' ') : '暂无移动'}
                </div>
              </div>
            </div>
            
            {gameHistory.length === 0 && (
              <div className="text-center py-4 text-gray-500 bg-yellow-50 rounded-lg border border-yellow-200">
                <p>请先走一步棋，再进行Circuit Trace分析</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Circuit可视化区域 */}
      {circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Circuit Trace 可视化</span>
              <div className="flex gap-2">
                <Button
                  onClick={handleSaveGraphJson}
                  variant="outline"
                  size="sm"
                >
                  保存JSON
                </Button>
                <Button
                  onClick={() => {
                    setCircuitVisualizationData(null);
                    setCircuitTraceResult(null);
                    setClickedNodeId(null);
                    setHoveredNodeId(null);
                    setPinnedNodeIds([]);
                    setHiddenNodeIds([]);
                    setSelectedFeature(null);
                    setConnectedFeatures([]);
                  }}
                  variant="outline"
                  size="sm"
                >
                  清除可视化
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Circuit Visualization Layout */}
            <div className="space-y-6 w-full max-w-full overflow-hidden">
              {/* Top Row: Link Graph and Node Connections side by side */}
              <div className="flex gap-6 h-[900px] w-full max-w-full overflow-hidden">
                {/* Link Graph Component - Left Side */}
                <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <div className="w-full h-full overflow-hidden relative">
                    <LinkGraphContainer 
                      data={circuitVisualizationData} 
                      onNodeClick={handleNodeClick}
                      onNodeHover={handleNodeHover}
                      onFeatureSelect={handleFeatureSelect}
                      onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                      onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                      clickedId={clickedNodeId}
                      hoveredId={hoveredNodeId}
                      pinnedIds={pinnedNodeIds}
                    />
                  </div>
                </div>

                {/* Node Connections Component - Right Side */}
                <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <NodeConnections
                    data={circuitVisualizationData}
                    clickedId={clickedNodeId}
                    hoveredId={hoveredNodeId}
                    pinnedIds={pinnedNodeIds}
                    hiddenIds={hiddenNodeIds}
                    onFeatureClick={handleNodeClick}
                    onFeatureSelect={handleFeatureSelect}
                    onFeatureHover={handleNodeHover}
                  />
                </div>
              </div>

              {/* Bottom Row: Feature Card */}
              {clickedNodeId && (() => {
                const currentNode = circuitVisualizationData.nodes.find((node: any) => node.nodeId === clickedNodeId);
                
                if (!currentNode) {
                  console.log('❌ 未找到节点:', clickedNodeId);
                  return null;
                }
                
                console.log('✅ 找到节点:', currentNode);
                
                // 从node_id解析真正的feature ID (格式: layer_featureId_ctxIdx)
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
                    
                    {/* 节点基本信息 */}
                    <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">节点ID:</span>
                          <span className="ml-2 font-mono text-blue-600">{currentNode.nodeId}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">特征类型:</span>
                          <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                            {currentNode.feature_type || 'Unknown'}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">层数:</span>
                          <span className="ml-2">{layerIdx}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">特征索引:</span>
                          <span className="ml-2">{featureIndex}</span>
                        </div>
                        {currentNode.sourceLinks && (
                          <div>
                            <span className="font-medium text-gray-700">出边数:</span>
                            <span className="ml-2">{currentNode.sourceLinks.length}</span>
                          </div>
                        )}
                        {currentNode.targetLinks && (
                          <div>
                            <span className="font-medium text-gray-700">入边数:</span>
                            <span className="ml-2">{currentNode.targetLinks.length}</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {selectedFeature ? (
                      <FeatureCard feature={selectedFeature} />
                    ) : (
                      <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                        <div className="text-center">
                          <p className="text-gray-600 mb-2">No feature is available for this node</p>
                          <p className="text-sm text-gray-500">
                            点击上方的"查看L{layerIdx} {nodeTypeDisplay} #{featureIndex}"链接查看详细信息
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 保留原有的简单circuitTraceResult显示，但移除跳转按钮 */}
      {circuitTraceResult && !circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle>Circuit Trace Result</CardTitle>
          </CardHeader>
          <CardContent>
            {circuitTraceResult.nodes ? (
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">分析摘要</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-blue-700">节点数量:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.nodes.length}</span>
                    </div>
                    <div>
                      <span className="text-blue-700">连接数量:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.links?.length || 0}</span>
                    </div>
                    {circuitTraceResult.metadata?.target_move && (
                      <div>
                        <span className="text-blue-700">目标移动:</span>
                        <span className="ml-2 font-mono text-green-600">{circuitTraceResult.metadata.target_move}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">关键节点 (前10个)</h4>
                  {circuitTraceResult.nodes.slice(0, 10).map((node: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{node.node_id}</span>
                        <Badge variant="outline" className="text-xs">
                          {node.feature_type}
                        </Badge>
                      </div>
                      {node.node_id && (
                        <Link
                          to={`/features?nodeId=${encodeURIComponent(node.node_id)}`}
                          className="text-blue-600 underline text-sm hover:text-blue-800"
                        >
                          查看Feature
                        </Link>
                      )}
                    </div>
                  ))}
                  {circuitTraceResult.nodes.length > 10 && (
                    <div className="text-center text-sm text-gray-500">
                      还有 {circuitTraceResult.nodes.length - 10} 个节点
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                无节点数据
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};