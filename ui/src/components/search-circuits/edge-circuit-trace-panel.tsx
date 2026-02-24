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

// Search edge data type
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

// Circuit trace parameter type
interface CircuitTraceParams {
  max_feature_nodes: number;
  node_threshold: number;
  edge_threshold: number;
  max_act_times: number | null;
}

// Circuit trace result type for a single edge
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
  // Trace state
  const [isTracing, setIsTracing] = useState(false);
  const [traceResult, setTraceResult] = useState<any>(existingResult?.traceResult || null);
  const [visualizationData, setVisualizationData] = useState<any>(existingResult?.visualizationData || null);
  
  // Parameter state
  const [showParamsDialog, setShowParamsDialog] = useState(false);
  const [circuitParams, setCircuitParams] = useState<CircuitTraceParams>(
    existingResult?.params || {
      max_feature_nodes: 4096,
      node_threshold: 0.73,
      edge_threshold: 0.57,
      max_act_times: null,
    }
  );
  
  // Side and order mode state
  const [traceSide, setTraceSide] = useState<'q' | 'k' | 'both'>(existingResult?.side || 'k');
  const [orderMode, setOrderMode] = useState<'positive' | 'negative'>(existingResult?.orderMode || 'positive');
  
  // Graph visualization state
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  
  // Fullscreen mode
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Collapsed / expanded state
  const [isCollapsed, setIsCollapsed] = useState(!isExpanded);
  
  // Model loading state (shared with play-game page)
  const { isLoading: isModelLoading, isLoaded: isModelLoaded } = useModelLoadingStatus();

  // Handle parameter changes
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

  // Handle circuit trace execution
  const handleCircuitTrace = useCallback(async () => {
    // First check if the backend is already running another circuit tracing process
    try {
      const statusResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        if (status.is_tracing) {
          alert('The backend is currently running another circuit tracing process. Please wait for it to finish and try again.');
          return;
        }
      }
    } catch (error) {
      console.error('Failed to check circuit tracing status:', error);
      // If the status check fails, still proceed (avoid blocking the user due to network issues)
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
      
      console.log('🔍 Edge circuit trace request:', {
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
        
        // Set BT4 model metadata
        if (data.metadata) {
          data.metadata.lorsa_analysis_name = 'BT4_lorsa_L{}A';
          data.metadata.tc_analysis_name = 'BT4_tc_L{}M';
        }
        
        // Transform data for visualization
        const transformedData = transformCircuitData(data);
        
        setTraceResult(data);
        setVisualizationData(transformedData);
        
        // Notify parent component
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
        
        console.log('✅ Edge circuit trace completed:', {
          edgeKey,
          nodesCount: data.nodes?.length || 0,
          linksCount: data.links?.length || 0,
        });
      } else {
        const errorText = await response.text();
        console.error('❌ Edge circuit trace failed:', response.status, errorText);
        alert(`Circuit trace failed: ${errorText}`);
      }
    } catch (error) {
      console.error('❌ Edge circuit trace error:', error);
      alert(`Circuit trace error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsTracing(false);
    }
  }, [edge, edgeKey, traceSide, orderMode, circuitParams, onTraceComplete]);

  // Handle node click
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

  // Handle node hover
  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredNodeId(nodeId);
  }, []);

  // Toggle collapsed state
  const toggleCollapse = useCallback(() => {
    setIsCollapsed(prev => !prev);
  }, []);

  // Render collapsed preview bar
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
            Completed ({traceResult.nodes?.length || 0} nodes)
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

  // Render expanded panel
  const renderExpandedPanel = () => (
    <Card className="border-2 border-blue-200">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-base">
          <div className="flex items-center space-x-2">
            <Zap className={`w-4 h-4 ${traceResult ? 'text-green-500' : 'text-blue-500'}`} />
            <span>Edge Circuit Trace: {edge.move}</span>
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
        {/* Edge information */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">Parent FEN:</span>
            <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 break-all">
              {edge.parent}
            </div>
          </div>
          <div>
            <span className="font-medium text-gray-700">Child FEN:</span>
            <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 break-all">
              {edge.child}
            </div>
          </div>
        </div>
        
        {/* Edge statistics */}
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
        
        {/* Trace controls */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Label className="text-xs">Analysis side</Label>
            <Select value={traceSide} onValueChange={(v: 'q' | 'k' | 'both') => setTraceSide(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="q">Q side</SelectItem>
                <SelectItem value="k">K side</SelectItem>
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
              title={!isModelLoaded ? 'TC/Lorsa model is not loaded. Please load the model first.' : ''}
            >
              {isTracing ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Tracing...
                </>
              ) : isModelLoading ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Model loading...
                </>
              ) : !isModelLoaded ? (
                <>
                  <AlertCircle className="w-3 h-3 mr-1" />
                  Model not loaded
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
        
        {/* Model loading hint */}
        {!isModelLoaded && !isModelLoading && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-2 text-yellow-700 text-xs flex items-center">
            <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
            <span>
              The TC/LoRSA model is not loaded yet. Please click the "Model Loading Status" button at the top of the page
              to load the model before running a circuit trace.
            </span>
          </div>
        )}
        
        {/* Chessboard preview */}
        <div className="flex justify-center space-x-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Before move</div>
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
            <div className="text-xs text-gray-500 mb-1">After move</div>
            <ChessBoard
              fen={edge.child}
              size="small"
              showCoordinates={false}
            />
          </div>
        </div>
        
        {/* Trace result summary */}
        {traceResult && (
          <div className="bg-green-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-green-700">Trace result</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsFullscreen(true)}
              >
                <Maximize2 className="w-3 h-3 mr-1" />
                View full graph
              </Button>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Nodes:</span>
                <span className="ml-1 font-mono">{traceResult.nodes?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-500">Edges:</span>
                <span className="ml-1 font-mono">{traceResult.links?.length || 0}</span>
              </div>
              <div>
                <span className="text-gray-500">Target move:</span>
                <span className="ml-1 font-mono">{traceResult.metadata?.target_move || edge.move}</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );

  // Render fullscreen dialog
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

  // Render parameter settings dialog
  const renderParamsDialog = () => (
    <Dialog open={showParamsDialog} onOpenChange={setShowParamsDialog}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Circuit Trace Parameters
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="max_feature_nodes">Max feature nodes</Label>
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
            <Label htmlFor="node_threshold">Node threshold</Label>
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
            <Label htmlFor="edge_threshold">Edge threshold</Label>
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
            <Label htmlFor="max_act_times">Max activation count</Label>
            <Input
              id="max_act_times"
              type="number"
              min="10000000"
              max="100000000"
              step="10000000"
              value={circuitParams.max_act_times || ''}
              onChange={(e) => handleParamsChange('max_act_times', e.target.value)}
              className="font-mono"
              placeholder="Leave empty for no limit"
            />
          </div>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={() => setShowParamsDialog(false)}>
            Cancel
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
            Reset
          </Button>
          <Button onClick={() => setShowParamsDialog(false)}>
            Confirm
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

