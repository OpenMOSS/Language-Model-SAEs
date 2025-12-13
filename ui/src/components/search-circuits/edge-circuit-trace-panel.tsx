import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, Settings, ChevronDown, ChevronUp, Maximize2, Minimize2, X, Zap, AlertCircle } from 'lucide-react';
import { LinkGraphContainer } from '@/components/circuits/link-graph-container';
import { NodeConnections } from '@/components/circuits/node-connections';
import { ChessBoard } from '@/components/chess/chess-board';
import { transformCircuitData } from '@/components/circuits/link-graph/utils';
import { useModelLoadingStatus } from '@/components/shared/model-loading-status';

// 搜索边数据类型
interface SearchEdge {
  parent: string;
  child: string;
  move: string;
  score: number;
  q: number;
  u: number;
  m: number;
  visits: number;
  stage: string;
  timestamp: number;
}

// Circuit Trace 参数类型
interface CircuitTraceParams {
  max_feature_nodes: number;
  node_threshold: number;
  edge_threshold: number;
  max_act_times: number | null;
}

// 边的 Circuit Trace 结果类型
export interface EdgeCircuitTraceResult {
  edgeKey: string;
  parentFen: string;
  childFen: string;
  move: string;
  traceResult: any;
  visualizationData: any;
  timestamp: number;
  params: CircuitTraceParams;
  orderMode: 'positive' | 'negative';
  side: 'q' | 'k' | 'both';
}

interface EdgeCircuitTracePanelProps {
  edge: SearchEdge;
  edgeKey: string;
  existingResult?: EdgeCircuitTraceResult | null;
  onTraceComplete?: (result: EdgeCircuitTraceResult) => void;
  onClose?: () => void;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

export const EdgeCircuitTracePanel: React.FC<EdgeCircuitTracePanelProps> = ({
  edge,
  edgeKey,
  existingResult,
  onTraceComplete,
  onClose,
  isExpanded = false,
  onToggleExpand,
}) => {
  // Trace 状态
  const [isTracing, setIsTracing] = useState(false);
  const [traceResult, setTraceResult] = useState<any>(existingResult?.traceResult || null);
  const [visualizationData, setVisualizationData] = useState<any>(existingResult?.visualizationData || null);
  
  // 参数状态
  const [showParamsDialog, setShowParamsDialog] = useState(false);
  const [circuitParams, setCircuitParams] = useState<CircuitTraceParams>(
    existingResult?.params || {
      max_feature_nodes: 4096,
      node_threshold: 0.73,
      edge_threshold: 0.57,
      max_act_times: null,
    }
  );
  
  // Side 和 Order Mode 状态
  const [traceSide, setTraceSide] = useState<'q' | 'k' | 'both'>(existingResult?.side || 'k');
  const [orderMode, setOrderMode] = useState<'positive' | 'negative'>(existingResult?.orderMode || 'positive');
  
  // Graph 可视化状态
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  
  // 全屏模式
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // 折叠状态
  const [isCollapsed, setIsCollapsed] = useState(!isExpanded);
  
  // 模型加载状态（与 play-game 页面共享）
  const { isLoading: isModelLoading, isLoaded: isModelLoaded } = useModelLoadingStatus();

  // 处理参数变更
  const handleParamsChange = useCallback((key: keyof CircuitTraceParams, value: string) => {
    setCircuitParams(prev => ({
      ...prev,
      [key]: key === 'max_feature_nodes' ? parseInt(value) || 1024 :
              key === 'max_act_times' ? (() => {
                if (value === '') return null;
                const num = parseInt(value);
                if (isNaN(num)) return null;
                const clamped = Math.max(10000000, Math.min(100000000, num));
                return Math.round(clamped / 10000000) * 10000000;
              })() :
              parseFloat(value) || prev[key]
    }));
  }, []);

  // 处理 Circuit Trace
  const handleCircuitTrace = useCallback(async () => {
    // 先检查后端是否有正在进行的circuit tracing进程
    try {
      const statusResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        if (status.is_tracing) {
          alert('后端正在执行另一个circuit tracing进程，请等待完成后再试');
          return;
        }
      }
    } catch (error) {
      console.error('检查circuit tracing状态失败:', error);
      // 如果检查失败，仍然继续执行（避免因为网络问题阻止用户操作）
    }
    
    setIsTracing(true);
    
    try {
      const requestBody = {
        fen: edge.parent,
        move_uci: edge.move,
        side: traceSide,
        order_mode: orderMode,
        max_feature_nodes: circuitParams.max_feature_nodes,
        node_threshold: circuitParams.node_threshold,
        edge_threshold: circuitParams.edge_threshold,
        max_act_times: circuitParams.max_act_times,
        save_activation_info: true,
      };
      
      console.log('🔍 Edge Circuit Trace 请求:', {
        edgeKey,
        move: edge.move,
        parentFen: edge.parent,
        ...requestBody,
      });
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // 设置 BT4 模型 metadata
        if (data.metadata) {
          data.metadata.lorsa_analysis_name = 'BT4_lorsa_L{}A';
          data.metadata.tc_analysis_name = 'BT4_tc_L{}M';
        }
        
        // 转换数据用于可视化
        const transformedData = transformCircuitData(data);
        
        setTraceResult(data);
        setVisualizationData(transformedData);
        
        // 通知父组件
        const result: EdgeCircuitTraceResult = {
          edgeKey,
          parentFen: edge.parent,
          childFen: edge.child,
          move: edge.move,
          traceResult: data,
          visualizationData: transformedData,
          timestamp: Date.now(),
          params: circuitParams,
          orderMode,
          side: traceSide,
        };
        
        onTraceComplete?.(result);
        
        console.log('✅ Edge Circuit Trace 完成:', {
          edgeKey,
          nodesCount: data.nodes?.length || 0,
          linksCount: data.links?.length || 0,
        });
      } else {
        const errorText = await response.text();
        console.error('❌ Edge Circuit Trace 失败:', response.status, errorText);
        alert(`Circuit Trace 失败: ${errorText}`);
      }
    } catch (error) {
      console.error('❌ Edge Circuit Trace 出错:', error);
      alert(`Circuit Trace 出错: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setIsTracing(false);
    }
  }, [edge, edgeKey, traceSide, orderMode, circuitParams, onTraceComplete]);

  // 处理节点点击
  const handleNodeClick = useCallback((node: any, isMetaKey: boolean) => {
    const nodeId = node.nodeId || node.id;
    
    if (isMetaKey) {
      const newPinnedIds = pinnedNodeIds.includes(nodeId)
        ? pinnedNodeIds.filter(id => id !== nodeId)
        : [...pinnedNodeIds, nodeId];
      setPinnedNodeIds(newPinnedIds);
    } else {
      const newClickedId = nodeId === clickedNodeId ? null : nodeId;
      setClickedNodeId(newClickedId);
    }
  }, [clickedNodeId, pinnedNodeIds]);

  // 处理节点悬停
  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredNodeId(nodeId);
  }, []);

  // 切换折叠状态
  const toggleCollapse = useCallback(() => {
    setIsCollapsed(prev => !prev);
  }, []);

  // 渲染折叠的预览
  const renderCollapsedPreview = () => (
    <div
      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors"
      onClick={toggleCollapse}
    >
      <div className="flex items-center space-x-3">
        <Zap className={`w-4 h-4 ${traceResult ? 'text-green-500' : 'text-gray-400'}`} />
        <span className="font-mono text-sm font-medium">{edge.move}</span>
        <span className="text-xs text-gray-500">
          {edge.parent.split(' ')[0].slice(0, 15)}... → {edge.child.split(' ')[0].slice(0, 15)}...
        </span>
        {traceResult && (
          <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">
            已完成 ({traceResult.nodes?.length || 0} 节点)
          </span>
        )}
      </div>
      <div className="flex items-center space-x-2">
        {traceResult && (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              setIsFullscreen(true);
            }}
          >
            <Maximize2 className="w-4 h-4" />
          </Button>
        )}
        <ChevronDown className="w-4 h-4 text-gray-400" />
      </div>
    </div>
  );

  // 渲染展开的面板
  const renderExpandedPanel = () => (
    <Card className="border-2 border-blue-200">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-base">
          <div className="flex items-center space-x-2">
            <Zap className={`w-4 h-4 ${traceResult ? 'text-green-500' : 'text-blue-500'}`} />
            <span>边 Circuit Trace: {edge.move}</span>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" onClick={() => setShowParamsDialog(true)}>
              <Settings className="w-4 h-4" />
            </Button>
            {traceResult && (
              <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(true)}>
                <Maximize2 className="w-4 h-4" />
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={toggleCollapse}>
              <ChevronUp className="w-4 h-4" />
            </Button>
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 边信息 */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">父节点 FEN:</span>
            <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 break-all">
              {edge.parent}
            </div>
          </div>
          <div>
            <span className="font-medium text-gray-700">子节点 FEN:</span>
            <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 break-all">
              {edge.child}
            </div>
          </div>
        </div>
        
        {/* 边统计 */}
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div className="bg-gray-50 p-2 rounded">
            <span className="text-gray-500">Visits:</span>
            <span className="ml-1 font-mono">{edge.visits}</span>
          </div>
          <div className="bg-gray-50 p-2 rounded">
            <span className="text-gray-500">Q:</span>
            <span className="ml-1 font-mono">{edge.q.toFixed(3)}</span>
          </div>
          <div className="bg-gray-50 p-2 rounded">
            <span className="text-gray-500">U:</span>
            <span className="ml-1 font-mono">{edge.u.toFixed(3)}</span>
          </div>
          <div className="bg-gray-50 p-2 rounded">
            <span className="text-gray-500">Score:</span>
            <span className="ml-1 font-mono">{edge.score.toFixed(3)}</span>
          </div>
        </div>
        
        {/* Trace 控制 */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Label className="text-xs">分析侧</Label>
            <Select value={traceSide} onValueChange={(v: 'q' | 'k' | 'both') => setTraceSide(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="q">Q侧</SelectItem>
                <SelectItem value="k">K侧</SelectItem>
                <SelectItem value="both">Q+K</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex-1">
            <Label className="text-xs">Order Mode</Label>
            <Select value={orderMode} onValueChange={(v: 'positive' | 'negative') => setOrderMode(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="positive">Positive</SelectItem>
                <SelectItem value="negative">Negative</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-end">
            <Button
              onClick={handleCircuitTrace}
              disabled={isTracing || isModelLoading || !isModelLoaded}
              size="sm"
              className="h-8"
              title={!isModelLoaded ? 'TC/LoRSA 模型未加载，请先加载模型' : ''}
            >
              {isTracing ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Tracing...
                </>
              ) : isModelLoading ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  模型加载中...
                </>
              ) : !isModelLoaded ? (
                <>
                  <AlertCircle className="w-3 h-3 mr-1" />
                  模型未加载
                </>
              ) : (
                <>
                  <Zap className="w-3 h-3 mr-1" />
                  Trace
                </>
              )}
            </Button>
          </div>
        </div>
        
        {/* 模型加载提示 */}
        {!isModelLoaded && !isModelLoading && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-2 text-yellow-700 text-xs flex items-center">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>TC/LoRSA 模型尚未加载。请点击页面顶部的"模型加载状态"按钮加载模型后再进行 Circuit Trace。</span>
          </div>
        )}
        
        {/* 棋盘预览 */}
        <div className="flex justify-center space-x-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">移动前</div>
            <ChessBoard
              fen={edge.parent}
              size="small"
              showCoordinates={false}
              move={edge.move}
            />
          </div>
          <div className="flex items-center">
            <span className="text-2xl text-gray-400">→</span>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">移动后</div>
            <ChessBoard
              fen={edge.child}
              size="small"
              showCoordinates={false}
            />
          </div>
        </div>
        
        {/* Trace 结果预览 */}
        {traceResult && (
          <div className="bg-green-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-green-700">Trace 结果</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsFullscreen(true)}
              >
                <Maximize2 className="w-3 h-3 mr-1" />
                查看完整图
              </Button>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500">节点数:</span>
                <span className="ml-1 font-mono">{traceResult.nodes?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-500">连接数:</span>
                <span className="ml-1 font-mono">{traceResult.links?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-500">目标移动:</span>
                <span className="ml-1 font-mono">{traceResult.metadata?.target_move || edge.move}</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );

  // 渲染全屏对话框
  const renderFullscreenDialog = () => (
    <Dialog open={isFullscreen} onOpenChange={setIsFullscreen}>
      <DialogContent className="max-w-[95vw] max-h-[95vh] w-full h-full overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-green-500" />
              <span>Circuit Trace: {edge.move}</span>
              <span className="text-sm font-normal text-gray-500">
                ({edge.parent.split(' ')[0].slice(0, 20)}...)
              </span>
            </div>
            <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(false)}>
              <Minimize2 className="w-4 h-4" />
            </Button>
          </DialogTitle>
        </DialogHeader>
        
        {visualizationData && (
          <div className="flex gap-4 h-[calc(95vh-120px)] overflow-hidden">
            {/* Link Graph */}
            <div className="flex-1 border rounded-lg p-4 bg-white overflow-hidden">
              <LinkGraphContainer
                data={visualizationData}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
                clickedId={clickedNodeId}
                hoveredId={hoveredNodeId}
                pinnedIds={pinnedNodeIds}
              />
            </div>
            
            {/* Node Connections */}
            <div className="w-80 border rounded-lg p-4 bg-white overflow-hidden">
              <NodeConnections
                data={visualizationData}
                clickedId={clickedNodeId}
                hoveredId={hoveredNodeId}
                pinnedIds={pinnedNodeIds}
                hiddenIds={[]}
                onFeatureClick={handleNodeClick}
                onFeatureHover={handleNodeHover}
              />
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );

  // 渲染参数设置对话框
  const renderParamsDialog = () => (
    <Dialog open={showParamsDialog} onOpenChange={setShowParamsDialog}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Circuit Trace 参数设置
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="max_feature_nodes">最大特征节点数</Label>
            <Input
              id="max_feature_nodes"
              type="number"
              min="1"
              max="10000"
              value={circuitParams.max_feature_nodes}
              onChange={(e) => handleParamsChange('max_feature_nodes', e.target.value)}
              className="font-mono"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="node_threshold">节点阈值</Label>
            <Input
              id="node_threshold"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={circuitParams.node_threshold}
              onChange={(e) => handleParamsChange('node_threshold', e.target.value)}
              className="font-mono"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="edge_threshold">边阈值</Label>
            <Input
              id="edge_threshold"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={circuitParams.edge_threshold}
              onChange={(e) => handleParamsChange('edge_threshold', e.target.value)}
              className="font-mono"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="max_act_times">最大激活次数</Label>
            <Input
              id="max_act_times"
              type="number"
              min="10000000"
              max="100000000"
              step="10000000"
              value={circuitParams.max_act_times || ''}
              onChange={(e) => handleParamsChange('max_act_times', e.target.value)}
              className="font-mono"
              placeholder="留空表示无限制"
            />
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowParamsDialog(false)}>
            取消
          </Button>
          <Button
            variant="outline"
            onClick={() => setCircuitParams({
              max_feature_nodes: 4096,
              node_threshold: 0.73,
              edge_threshold: 0.57,
              max_act_times: null,
            })}
          >
            重置
          </Button>
          <Button onClick={() => setShowParamsDialog(false)}>
            确定
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  return (
    <div className="edge-circuit-trace-panel">
      {isCollapsed ? renderCollapsedPreview() : renderExpandedPanel()}
      {renderFullscreenDialog()}
      {renderParamsDialog()}
    </div>
  );
};

export default EdgeCircuitTracePanel;

