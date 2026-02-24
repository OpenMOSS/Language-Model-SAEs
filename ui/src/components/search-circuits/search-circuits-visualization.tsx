import { useState, useCallback, useMemo, useRef } from "react";
import { ChessBoard } from "@/components/chess/chess-board";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Zap, ChevronDown, ChevronRight, Maximize2, Download, Trash2, Play, Square, Settings, Loader2, AlertCircle } from "lucide-react";
import { EdgeCircuitTracePanel, EdgeCircuitTraceResult } from "./edge-circuit-trace-panel";
import { ModelLoadingStatus, useModelLoadingStatus } from "@/components/shared/model-loading-status";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
interface SearchNode {
  fen: string;
  moves: string[];
  policies: number[];
  is_tt_hit: boolean | null;
  timestamp: number | null;
}

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

interface SearchMetadata {
  root_fen: string;
  search_params: {
    max_playouts: number;
    target_minibatch_size: number;
    cpuct: number;
    max_depth: number;
    moves_left_slope: number;
    moves_left_max_effect: number;
    fpu_value: number;
    fpu_absolute: boolean;
    draw_score: number;
  };
  search_results: {
    best_move: string;
    total_playouts: number;
    max_depth: number;
  };
  trace_stats: {
    num_edge_records: number;
    num_expansion_records: number;
  };
  export_timestamp: string;
}

interface SearchTraceData {
  nodes: SearchNode[];
  edges: SearchEdge[];
  metadata: SearchMetadata;
}

// for visualizing the tree node
interface TreeNode {
  fen: string;
  shortFen: string;
  moves: string[];
  policies: number[];
  is_tt_hit: boolean | null;
  children: TreeNode[];
  parentEdge?: SearchEdge;
  depth: number;
  totalVisits: number;
  bestMove?: string;
}

export const SearchCircuitsVisualization = () => {
  const [searchData, setSearchData] = useState<SearchTraceData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedNode, setSelectedNode] = useState<SearchNode | null>(null);
  const [selectedNodeFen, setSelectedNodeFen] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [fileName, setFileName] = useState<string>("");
  
  // Edge Circuit Trace related states
  const [edgeTraceResults, setEdgeTraceResults] = useState<Map<string, EdgeCircuitTraceResult>>(new Map());
  const [selectedEdgeForTrace, setSelectedEdgeForTrace] = useState<SearchEdge | null>(null);
  const [showEdgeTraceDialog, setShowEdgeTraceDialog] = useState(false);
  const [expandedEdgeTraces, setExpandedEdgeTraces] = useState<Set<string>>(new Set());
  
  // Batch Trace related states
  const [showBatchTraceDialog, setShowBatchTraceDialog] = useState(false);
  const [isBatchTracing, setIsBatchTracing] = useState(false);
  const [batchTraceProgress, setBatchTraceProgress] = useState({ current: 0, total: 0, currentEdge: '' });
  const [batchTraceParams, setBatchTraceParams] = useState({
    max_feature_nodes: 4096,
    node_threshold: 0.73,
    edge_threshold: 0.57,
    max_act_times: null as number | null,
    side: 'both' as 'q' | 'k' | 'both',
    orderMode: 'positive' as 'positive' | 'negative',
    skipExisting: true,  // skip edges with existing results
  });
  const batchTraceAbortRef = useRef(false);
  const batchTraceControllerRef = useRef<AbortController | null>(null);
  
  // model loading status
  const { isLoading: isModelLoading, isLoaded: isModelLoaded } = useModelLoadingStatus();

  // generate the unique key of the edge
  const getEdgeKey = useCallback((edge: SearchEdge) => {
    return `${edge.parent}__${edge.child}__${edge.move}`;
  }, []);

  // build the tree structure
  const treeData = useMemo(() => {
    if (!searchData) return null;

    const nodeMap = new Map<string, SearchNode>();
    searchData.nodes.forEach(node => {
      nodeMap.set(node.fen, node);
    });

    // aggregate edge information: for the same (parent, child, move) combination, take the last edge information
    const edgeMap = new Map<string, SearchEdge>();
    searchData.edges.forEach(edge => {
      const key = `${edge.parent}__${edge.child}__${edge.move}`;
      edgeMap.set(key, edge);
    });

    // build the child node mapping
    const childrenMap = new Map<string, { child: string; edge: SearchEdge }[]>();
    edgeMap.forEach((edge) => {
      if (!childrenMap.has(edge.parent)) {
        childrenMap.set(edge.parent, []);
      }
      childrenMap.get(edge.parent)!.push({ child: edge.child, edge });
    });

    // recursively build the tree
    const buildTree = (fen: string, depth: number, visited: Set<string>): TreeNode | null => {
      if (visited.has(fen)) return null;
      visited.add(fen);

      const node = nodeMap.get(fen);
      if (!node) return null;

      const children: TreeNode[] = [];
      const childEdges = childrenMap.get(fen) || [];

      // calculate the total visits of the node
      let totalVisits = 0;
      childEdges.forEach(({ edge }) => {
        totalVisits = Math.max(totalVisits, edge.visits);
      });

      // find the best move (the one with the most visits)
      let bestMove: string | undefined;
      let maxVisits = 0;
      childEdges.forEach(({ edge }) => {
        if (edge.visits > maxVisits) {
          maxVisits = edge.visits;
          bestMove = edge.move;
        }
      });

      // sort the children by the visits
      const sortedChildEdges = [...childEdges].sort((a, b) => b.edge.visits - a.edge.visits);

      sortedChildEdges.forEach(({ child, edge }) => {
        const childTree = buildTree(child, depth + 1, visited);
        if (childTree) {
          childTree.parentEdge = edge;
          children.push(childTree);
        }
      });

      // create a short FEN for displaying
      const shortFen = fen.split(' ')[0].slice(0, 20) + '...';

      return {
        fen,
        shortFen,
        moves: node.moves,
        policies: node.policies,
        is_tt_hit: node.is_tt_hit,
        children,
        depth,
        totalVisits,
        bestMove,
      };
    };

    const rootFen = searchData.metadata.root_fen;
    return buildTree(rootFen, 0, new Set());
  }, [searchData]);

  // handle the file upload
  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('Please upload a JSON file');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const text = await file.text();
      const jsonData: SearchTraceData = JSON.parse(text);

      // Validate basic JSON structure
      if (!jsonData.nodes || !jsonData.edges || !jsonData.metadata) {
        throw new Error('Invalid search trace JSON: missing "nodes", "edges", or "metadata" fields.');
      }

      setSearchData(jsonData);
      setFileName(file.name);
      setSelectedNode(null);
      setSelectedNodeFen(null);
      setEdgeTraceResults(new Map());
      setExpandedEdgeTraces(new Set());
      
      // By default, expand the root node
      if (jsonData.metadata.root_fen) {
        setExpandedNodes(new Set([jsonData.metadata.root_fen]));
      }
    } catch (err) {
      console.error('Failed to load search trace data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load search trace data.');
    } finally {
      setIsLoading(false);
    }
  }, []);

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
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, [handleFileUpload]);

  // Toggle node expanded/collapsed
  const toggleNode = useCallback((fen: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(fen)) {
        next.delete(fen);
      } else {
        next.add(fen);
      }
      return next;
    });
  }, []);

  // Select a node in the search tree
  const selectNode = useCallback((fen: string) => {
    if (!searchData) return;
    const node = searchData.nodes.find(n => n.fen === fen);
    setSelectedNode(node || null);
    setSelectedNodeFen(fen);
  }, [searchData]);

  // Get incoming edges for a node
  const getNodeInEdges = useCallback((fen: string) => {
    if (!searchData) return [];
    return searchData.edges.filter(e => e.child === fen);
  }, [searchData]);

  // Get outgoing edges for a node (aggregated by (parent, child, move))
  const getNodeOutEdges = useCallback((fen: string) => {
    if (!searchData) return [];
    // Aggregate edges with the same (parent, child, move) combination
    const edgeMap = new Map<string, SearchEdge>();
    searchData.edges.filter(e => e.parent === fen).forEach(edge => {
      const key = `${edge.child}__${edge.move}`;
      edgeMap.set(key, edge);
    });
    return Array.from(edgeMap.values()).sort((a, b) => b.visits - a.visits);
  }, [searchData]);

  // Handle completion of a single edge Circuit Trace
  const handleEdgeTraceComplete = useCallback((result: EdgeCircuitTraceResult) => {
    setEdgeTraceResults(prev => {
      const next = new Map(prev);
      next.set(result.edgeKey, result);
      return next;
    });
  }, []);

  // Toggle edge trace expanded/collapsed state
  const toggleEdgeTraceExpand = useCallback((edgeKey: string) => {
    setExpandedEdgeTraces(prev => {
      const next = new Set(prev);
      if (next.has(edgeKey)) {
        next.delete(edgeKey);
      } else {
        next.add(edgeKey);
      }
      return next;
    });
  }, []);

  // Open edge trace dialog
  const openEdgeTraceDialog = useCallback((edge: SearchEdge) => {
    setSelectedEdgeForTrace(edge);
    setShowEdgeTraceDialog(true);
  }, []);

  // Export all edge trace results
  const exportAllEdgeTraces = useCallback(() => {
    if (edgeTraceResults.size === 0) {
      alert('No edge trace results to export');
      return;
    }

    const exportData = {
      searchMetadata: searchData?.metadata,
      edgeTraces: Array.from(edgeTraceResults.values()).map(result => ({
        edgeKey: result.edgeKey,
        parentFen: result.parentFen,
        childFen: result.childFen,
        move: result.move,
        params: result.params,
        orderMode: result.orderMode,
        side: result.side,
        timestamp: result.timestamp,
        traceResult: result.traceResult,
      })),
      exportTimestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `search_edge_traces_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [edgeTraceResults, searchData]);

  // Clear all edge trace results
  const clearAllEdgeTraces = useCallback(() => {
    if (confirm('Are you sure you want to clear all edge trace results?')) {
      setEdgeTraceResults(new Map());
      setExpandedEdgeTraces(new Set());
    }
  }, []);

  // Get all unique edges in the search trace
  const getAllUniqueEdges = useCallback((): SearchEdge[] => {
    if (!searchData) return [];
    
    const edgeMap = new Map<string, SearchEdge>();
    searchData.edges.forEach(edge => {
      const key = `${edge.parent}__${edge.child}__${edge.move}`;
      edgeMap.set(key, edge);
    });
    
    return Array.from(edgeMap.values());
  }, [searchData]);

  // Run Circuit Trace in batch for all edges
  const startBatchTrace = useCallback(async () => {
    if (!searchData || !isModelLoaded) return;
    
    // First, check whether the backend is already running another circuit tracing job
    try {
      const statusResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        if (status.is_tracing) {
          alert('The backend is already running another circuit tracing job. Please wait for it to finish and try again.');
          return;
        }
      }
    } catch (error) {
      console.error('Failed to check circuit tracing status:', error);
      // If the status check fails, still proceed (avoid blocking user due to network errors)
    }
    
    const allEdges = getAllUniqueEdges();
    const edgesToTrace = batchTraceParams.skipExisting 
      ? allEdges.filter(edge => !edgeTraceResults.has(getEdgeKey(edge)))
      : allEdges;
    
    if (edgesToTrace.length === 0) {
      alert('No edges to trace (all edges already have results).');
      return;
    }
    
    setIsBatchTracing(true);
    setBatchTraceProgress({ current: 0, total: edgesToTrace.length, currentEdge: '' });
    batchTraceAbortRef.current = false;
    setShowBatchTraceDialog(false);
    
    for (let i = 0; i < edgesToTrace.length; i++) {
      // Check if batch tracing has been aborted
      if (batchTraceAbortRef.current) {
        console.log('🛑 Batch trace aborted by user');
        break;
      }
      
      // For each iteration, also check status to ensure no concurrent tracing job was started
      try {
        const statusResponse = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace/status`);
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          if (status.is_tracing && i > 0) {
            // If this is not the first request and a new trace started, abort this batch
            console.log('🛑 Detected a new circuit tracing job, aborting batch trace');
            batchTraceAbortRef.current = true;
            break;
          }
        }
      } catch (error) {
        console.error('Failed to check circuit tracing status:', error);
        // If the status check fails, continue
      }
      
      const edge = edgesToTrace[i];
      const edgeKey = getEdgeKey(edge);
      
      setBatchTraceProgress({
        current: i + 1,
        total: edgesToTrace.length,
        currentEdge: `${edge.move} (${edge.parent.split(' ')[0].slice(0, 15)}...)`
      });
      
      try {
        console.log(`🔍 Batch Trace [${i + 1}/${edgesToTrace.length}]: ${edge.move}`);
        
        // Create an AbortController so the user can abort the current request
        const controller = new AbortController();
        batchTraceControllerRef.current = controller;

        const requestBody = {
          fen: edge.parent,
          move_uci: edge.move,
          side: batchTraceParams.side,
          order_mode: batchTraceParams.orderMode,
          max_feature_nodes: batchTraceParams.max_feature_nodes,
          node_threshold: batchTraceParams.node_threshold,
          edge_threshold: batchTraceParams.edge_threshold,
          max_act_times: batchTraceParams.max_act_times,
          save_activation_info: true,
        };
        
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        
        if (response.ok) {
          const data = await response.json();
          
          const result: EdgeCircuitTraceResult = {
            edgeKey,
            parentFen: edge.parent,
            childFen: edge.child,
            move: edge.move,
            traceResult: data,
            visualizationData: null,  // In batch mode we do not precompute visualization data
            timestamp: Date.now(),
            params: {
              max_feature_nodes: batchTraceParams.max_feature_nodes,
              node_threshold: batchTraceParams.node_threshold,
              edge_threshold: batchTraceParams.edge_threshold,
              max_act_times: batchTraceParams.max_act_times,
            },
            orderMode: batchTraceParams.orderMode,
            side: batchTraceParams.side,
          };
          
          setEdgeTraceResults(prev => {
            const next = new Map(prev);
            next.set(edgeKey, result);
            return next;
          });
          
          console.log(`✅ Batch Trace completed [${i + 1}/${edgesToTrace.length}]: ${edge.move}`);
        } else {
          const errorText = await response.text();
          console.error(`❌ Batch Trace failed [${i + 1}/${edgesToTrace.length}]: ${edge.move}`, errorText);
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          console.warn(`⏹️ Current trace aborted: ${edge.move}`);
          break;
        }
        console.error(`❌ Batch Trace error [${i + 1}/${edgesToTrace.length}]: ${edge.move}`, error);
      }
      
      // Short delay to avoid sending requests too quickly
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    setIsBatchTracing(false);
    setBatchTraceProgress({ current: 0, total: 0, currentEdge: '' });
    batchTraceControllerRef.current = null;
  }, [searchData, isModelLoaded, batchTraceParams, edgeTraceResults, getAllUniqueEdges, getEdgeKey]);

  // Abort batch trace
  const abortBatchTrace = useCallback(() => {
    batchTraceAbortRef.current = true;
    if (batchTraceControllerRef.current) {
      batchTraceControllerRef.current.abort();
    }
    setIsBatchTracing(false);
  }, []);

  // Render a single tree node in the search tree
  const renderTreeNode = useCallback((node: TreeNode, isRoot: boolean = false): React.ReactNode => {
    const isExpanded = expandedNodes.has(node.fen);
    const isSelected = selectedNodeFen === node.fen;
    const hasChildren = node.children.length > 0;

    // Determine background intensity based on visit count
    const maxVisits = treeData?.totalVisits || 1;
    const visitRatio = node.totalVisits / Math.max(maxVisits, 1);
    const bgOpacity = Math.max(0.1, Math.min(0.8, visitRatio));

    // Check whether this node's parent edge already has a trace result
    const parentEdgeKey = node.parentEdge ? getEdgeKey(node.parentEdge) : null;
    const hasParentEdgeTrace = parentEdgeKey ? edgeTraceResults.has(parentEdgeKey) : false;

    return (
      <div key={node.fen} className="ml-4">
        <div
          className={`
            flex items-center space-x-2 py-2 px-3 rounded-lg cursor-pointer
            transition-colors duration-200
            ${isSelected ? 'bg-blue-100 border-2 border-blue-500' : 'hover:bg-gray-100'}
            ${isRoot ? 'bg-green-50 border border-green-300' : ''}
          `}
          style={{
            backgroundColor: isSelected ? undefined : `rgba(59, 130, 246, ${bgOpacity * 0.2})`,
          }}
          onClick={() => selectNode(node.fen)}
        >
          {/* Expand / collapse toggle */}
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleNode(node.fen);
              }}
              className="w-6 h-6 flex items-center justify-center text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded"
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          )}
          {!hasChildren && <div className="w-6" />}

          {/* Node info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              {isRoot && (
                <span className="px-2 py-0.5 bg-green-500 text-white text-xs rounded-full">Root</span>
              )}
              {node.parentEdge && (
                <span className={`px-2 py-0.5 text-white text-xs rounded font-mono ${hasParentEdgeTrace ? 'bg-purple-500' : 'bg-blue-500'}`}>
                  {node.parentEdge.move}
                  {hasParentEdgeTrace && <Zap className="inline w-3 h-3 ml-1" />}
                </span>
              )}
              <span className="text-sm text-gray-600 truncate font-mono">
                {node.shortFen}
              </span>
            </div>
            <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
              <span>Depth: {node.depth}</span>
              <span>Moves: {node.moves.length}</span>
              {node.parentEdge && (
                <>
                  <span>Visits: {node.parentEdge.visits}</span>
                  <span>Q: {node.parentEdge.q.toFixed(3)}</span>
                  <span>Score: {node.parentEdge.score.toFixed(3)}</span>
                </>
              )}
              {node.is_tt_hit !== null && (
                <span className={node.is_tt_hit ? 'text-green-600' : 'text-gray-400'}>
                  {node.is_tt_hit ? 'TT hit' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Circuit Trace button */}
          {node.parentEdge && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                openEdgeTraceDialog(node.parentEdge!);
              }}
              className={`p-1.5 rounded transition-colors ${
                hasParentEdgeTrace 
                  ? 'bg-purple-100 text-purple-600 hover:bg-purple-200' 
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              }`}
              title="Circuit Trace this edge"
            >
              <Zap className="w-4 h-4" />
            </button>
          )}

          {/* Best move tag */}
          {node.bestMove && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded border border-yellow-300">
              Best: {node.bestMove}
            </span>
          )}
        </div>

        {/* Child nodes */}
        {isExpanded && hasChildren && (
          <div className="border-l-2 border-gray-200 ml-3">
            {node.children.map(child => renderTreeNode(child, false))}
          </div>
        )}
      </div>
    );
  }, [expandedNodes, selectedNodeFen, selectNode, toggleNode, treeData, edgeTraceResults, getEdgeKey, openEdgeTraceDialog]);

  // Render edge trace result list
  const renderEdgeTraceList = () => {
    if (edgeTraceResults.size === 0) return null;

    return (
      <div className="bg-white rounded-lg border p-4 mt-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center">
            <Zap className="w-5 h-5 mr-2 text-purple-500" />
            Edge Circuit Trace results ({edgeTraceResults.size})
          </h3>
          <div className="flex space-x-2">
            <Button variant="outline" size="sm" onClick={exportAllEdgeTraces}>
              <Download className="w-4 h-4 mr-1" />
              Export all
            </Button>
            <Button variant="outline" size="sm" onClick={clearAllEdgeTraces}>
              <Trash2 className="w-4 h-4 mr-1" />
              Clear all
            </Button>
          </div>
        </div>
        
        <div className="space-y-2">
          {Array.from(edgeTraceResults.values()).map(result => {
            const isExpanded = expandedEdgeTraces.has(result.edgeKey);
            
            return (
              <div key={result.edgeKey} className="border rounded-lg overflow-hidden">
                <div
                  className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                  onClick={() => toggleEdgeTraceExpand(result.edgeKey)}
                >
                  <div className="flex items-center space-x-3">
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-gray-500" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-gray-500" />
                    )}
                    <Zap className="w-4 h-4 text-purple-500" />
                    <span className="font-mono font-medium">{result.move}</span>
                    <span className="text-xs text-gray-500">
                      {result.parentFen.split(' ')[0].slice(0, 15)}...
                    </span>
                            <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">
                      {result.traceResult?.nodes?.length || 0} nodes
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">
                      {new Date(result.timestamp).toLocaleTimeString()}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        const edge: SearchEdge = {
                          parent: result.parentFen,
                          child: result.childFen,
                          move: result.move,
                          score: 0,
                          q: 0,
                          u: 0,
                          m: 0,
                          visits: 0,
                          stage: '',
                          timestamp: 0,
                        };
                        openEdgeTraceDialog(edge);
                      }}
                    >
                      <Maximize2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                
                {isExpanded && (
                  <div className="p-4 border-t">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium text-gray-700">Parameters:</span>
                        <div className="mt-1 text-xs space-y-1">
                          <div>Side: {result.side}</div>
                          <div>Order Mode: {result.orderMode}</div>
                          <div>Max Nodes: {result.params.max_feature_nodes}</div>
                          <div>Node Threshold: {result.params.node_threshold}</div>
                          <div>Edge Threshold: {result.params.edge_threshold}</div>
                        </div>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">Summary:</span>
                        <div className="mt-1 text-xs space-y-1">
                          <div>Nodes: {result.traceResult?.nodes?.length || 0}</div>
                          <div>Links: {result.traceResult?.links?.length || 0}</div>
                          <div>Target move: {result.traceResult?.metadata?.target_move || result.move}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Error state
  if (error) {
    return (
      <div className="space-y-6">
        {/* Global BT4 SAE combo selection (Lorsa / Transcoder), shares backend cache and loading logs */}
        <SaeComboLoader />

        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">Failed to load</h3>
            <p className="text-gray-600">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        {/* Global BT4 SAE combo selection (Lorsa / Transcoder), shares backend cache and loading logs */}
        <SaeComboLoader />

        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading search trace data...</p>
          </div>
        </div>
      </div>
    );
  }

  // Initial state (no file uploaded yet)
  if (!searchData) {
    return (
      <div className="space-y-6">
        {/* Global BT4 SAE combo selection (Lorsa / Transcoder), shares backend cache and loading logs */}
        <SaeComboLoader />

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
                Upload search trace data
              </h3>
              <p className="text-gray-600 mb-4">
                Drag and drop a JSON file here, or click to select one
              </p>
              <input
                type="file"
                accept=".json"
                onChange={handleFileInput}
                className="hidden"
                id="search-file-upload"
              />
              <label
                htmlFor="search-file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 cursor-pointer transition-colors"
              >
                Select file
              </label>
            </div>
            <p className="text-sm text-gray-500">
              Supports search_trace_*.json formatted search trace files
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Main view (file loaded)
  return (
    <div className="space-y-6">
      {/* Global BT4 SAE combo selection (Lorsa / Transcoder), shares backend cache and loading logs */}
      <SaeComboLoader />

      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">File: {fileName}</span>
          <span className="px-2 py-1 bg-green-100 text-green-800 text-sm rounded">
            Best move: {searchData.metadata.search_results.best_move}
          </span>
          {edgeTraceResults.size > 0 && (
            <span className="px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded flex items-center">
              <Zap className="w-3 h-3 mr-1" />
              {edgeTraceResults.size} edges traced
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {/* Batch Trace button */}
          {isBatchTracing ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={abortBatchTrace}
            >
              <Square className="w-4 h-4 mr-1" />
              Stop batch Trace
            </Button>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={() => setShowBatchTraceDialog(true)}
              disabled={!isModelLoaded || isModelLoading}
              title={!isModelLoaded ? 'Please load the model first' : 'Batch Trace all edges'}
            >
              <Play className="w-4 h-4 mr-1" />
              Batch Trace all edges
            </Button>
          )}
          {/* Model loading status button */}
          <ModelLoadingStatus 
            showButton={true} 
            buttonVariant="outline" 
            buttonSize="sm"
            autoPreload={true}
          />
          <button
            onClick={() => {
              setSearchData(null);
              setSelectedNode(null);
              setSelectedNodeFen(null);
              setFileName("");
              setEdgeTraceResults(new Map());
              setExpandedEdgeTraces(new Set());
            }}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Upload new file
          </button>
        </div>
      </div>

      {/* Metadata cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Search parameters */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Search parameters</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Max playouts:</span>
              <span className="font-mono">{searchData.metadata.search_params.max_playouts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Max depth:</span>
              <span className="font-mono">{searchData.metadata.search_params.max_depth}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">CPUCT:</span>
              <span className="font-mono">{searchData.metadata.search_params.cpuct}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Minibatch size:</span>
              <span className="font-mono">{searchData.metadata.search_params.target_minibatch_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">FPU value:</span>
              <span className="font-mono">{searchData.metadata.search_params.fpu_value}</span>
            </div>
          </div>
        </div>

        {/* Search results */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Search results</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Best move:</span>
              <span className="font-mono font-bold text-green-600">{searchData.metadata.search_results.best_move}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Total playouts:</span>
              <span className="font-mono">{searchData.metadata.search_results.total_playouts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Reached depth:</span>
              <span className="font-mono">{searchData.metadata.search_results.max_depth}</span>
            </div>
          </div>
        </div>

        {/* Trace statistics */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Trace statistics</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Nodes:</span>
              <span className="font-mono">{searchData.nodes.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Edge records:</span>
              <span className="font-mono">{searchData.metadata.trace_stats.num_edge_records}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Expansion records:</span>
              <span className="font-mono">{searchData.metadata.trace_stats.num_expansion_records}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Exported at:</span>
              <span className="font-mono text-xs">{searchData.metadata.export_timestamp}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div className="flex gap-6 h-[700px]">
        {/* Left: search tree */}
        <div className="flex-1 bg-white rounded-lg border p-4 overflow-hidden flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Search tree</h3>
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  if (searchData) {
                    const allFens = searchData.nodes.map(n => n.fen);
                    setExpandedNodes(new Set(allFens));
                  }
                }}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                Expand all
              </button>
              <button
                onClick={() => {
                  if (searchData) {
                    setExpandedNodes(new Set([searchData.metadata.root_fen]));
                  }
                }}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                Collapse all
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {treeData && renderTreeNode(treeData, true)}
          </div>
        </div>

        {/* Right: node details */}
        <div className="w-[500px] bg-white rounded-lg border p-4 overflow-hidden flex flex-col">
          <h3 className="text-lg font-semibold mb-4">Node details</h3>
          {selectedNode ? (
            <div className="flex-1 overflow-y-auto space-y-4">
              {/* Board display */}
              <div className="flex justify-center">
                <ChessBoard
                  fen={selectedNode.fen}
                  size="medium"
                  showCoordinates={true}
                  move={searchData.metadata.search_results.best_move}
                />
              </div>

              {/* FEN string */}
              <div className="bg-gray-50 rounded p-3">
                <div className="text-sm font-medium text-gray-700 mb-1">FEN string:</div>
                <div className="font-mono text-xs break-all select-all">
                  {selectedNode.fen}
                </div>
              </div>

              {/* Incoming edges */}
              {getNodeInEdges(selectedNode.fen).length > 0 && (
                <div className="bg-blue-50 rounded p-3">
                  <div className="text-sm font-medium text-blue-700 mb-2">Incoming edges:</div>
                  <div className="space-y-2">
                    {getNodeInEdges(selectedNode.fen).slice(0, 5).map((edge, idx) => {
                      const edgeKey = getEdgeKey(edge);
                      const hasTrace = edgeTraceResults.has(edgeKey);
                      
                      return (
                        <div key={idx} className="text-xs bg-white p-2 rounded border">
                          <div className="flex justify-between items-center">
                            <span className={`font-mono font-bold ${hasTrace ? 'text-purple-600' : ''}`}>
                              {edge.move}
                              {hasTrace && <Zap className="inline w-3 h-3 ml-1" />}
                            </span>
                            <div className="flex items-center space-x-2">
                              <span className="text-gray-500">{edge.stage}</span>
                              <button
                                onClick={() => openEdgeTraceDialog(edge)}
                                className={`p-1 rounded ${hasTrace ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'} hover:opacity-80`}
                                title="Circuit Trace"
                              >
                                <Zap className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                          <div className="grid grid-cols-4 gap-1 mt-1 text-gray-600">
                            <span>Visits: {edge.visits}</span>
                            <span>Q: {edge.q.toFixed(3)}</span>
                            <span>U: {edge.u.toFixed(3)}</span>
                            <span>S: {edge.score.toFixed(3)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Available moves and policies */}
              {selectedNode.moves.length > 0 && (
                <div className="bg-green-50 rounded p-3">
                  <div className="text-sm font-medium text-green-700 mb-2">
                    Available moves ({selectedNode.moves.length}):
                  </div>
                  <div className="grid grid-cols-3 gap-2 max-h-40 overflow-y-auto">
                    {selectedNode.moves.map((move, idx) => {
                      const policy = selectedNode.policies[idx] || 0;
                      const isTopMove = idx < 3;
                      return (
                        <div
                          key={idx}
                          className={`text-xs p-2 rounded border ${
                            isTopMove ? 'bg-green-100 border-green-300' : 'bg-white'
                          }`}
                        >
                          <div className="font-mono font-bold">{move}</div>
                          <div className="text-gray-600">{(policy * 100).toFixed(1)}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Outgoing edges */}
              {getNodeOutEdges(selectedNode.fen).length > 0 && (
                <div className="bg-orange-50 rounded p-3">
                  <div className="text-sm font-medium text-orange-700 mb-2">
                    Outgoing edges ({getNodeOutEdges(selectedNode.fen).length}):
                  </div>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {getNodeOutEdges(selectedNode.fen).map((edge, idx) => {
                      const edgeKey = getEdgeKey(edge);
                      const hasTrace = edgeTraceResults.has(edgeKey);
                      
                      return (
                        <div
                          key={idx}
                          className="text-xs bg-white p-2 rounded border cursor-pointer hover:bg-orange-100"
                          onClick={() => selectNode(edge.child)}
                        >
                          <div className="flex justify-between items-center">
                            <span className={`font-mono font-bold ${hasTrace ? 'text-purple-600' : ''}`}>
                              {edge.move}
                              {hasTrace && <Zap className="inline w-3 h-3 ml-1" />}
                            </span>
                            <div className="flex items-center space-x-2">
                              <span className="text-orange-600">→ Go to child</span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openEdgeTraceDialog(edge);
                                }}
                                className={`p-1 rounded ${hasTrace ? 'bg-purple-100 text-purple-600' : 'bg-gray-100 text-gray-500'} hover:opacity-80`}
                                title="Circuit Trace"
                              >
                                <Zap className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                          <div className="grid grid-cols-4 gap-1 mt-1 text-gray-600">
                            <span>V: {edge.visits}</span>
                            <span>Q: {edge.q.toFixed(3)}</span>
                            <span>U: {edge.u.toFixed(3)}</span>
                            <span>S: {edge.score.toFixed(3)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                </svg>
                <p>Click a node in the search tree to view details.</p>
                <p className="text-sm mt-2">
                  Click the <Zap className="inline w-4 h-4 text-purple-500" /> button to run Circuit Trace on an edge.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Edge trace result list */}
      {renderEdgeTraceList()}

      {/* Root position board */}
      <div className="bg-white rounded-lg border p-4">
        <h3 className="text-lg font-semibold mb-4">Root position</h3>
        <div className="flex justify-center">
          <ChessBoard
            fen={searchData.metadata.root_fen}
            size="large"
            showCoordinates={true}
            move={searchData.metadata.search_results.best_move}
          />
        </div>
      </div>

      {/* Edge Circuit Trace dialog */}
      <Dialog open={showEdgeTraceDialog} onOpenChange={setShowEdgeTraceDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-purple-500" />
              <span>Edge Circuit Trace</span>
            </DialogTitle>
          </DialogHeader>
          
          {selectedEdgeForTrace && (
            <EdgeCircuitTracePanel
              edge={selectedEdgeForTrace}
              edgeKey={getEdgeKey(selectedEdgeForTrace)}
              existingResult={edgeTraceResults.get(getEdgeKey(selectedEdgeForTrace)) || null}
              onTraceComplete={handleEdgeTraceComplete}
              onClose={() => setShowEdgeTraceDialog(false)}
              isExpanded={true}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Batch Trace configuration dialog */}
      <Dialog open={showBatchTraceDialog} onOpenChange={setShowBatchTraceDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Settings className="w-5 h-5" />
              <span>Batch Circuit Trace configuration</span>
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            {/* Summary */}
            <Card>
              <CardContent className="pt-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Total unique edges:</span>
                    <span className="ml-2 font-mono">{getAllUniqueEdges().length}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Traced:</span>
                    <span className="ml-2 font-mono">{edgeTraceResults.size}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Pending:</span>
                    <span className="ml-2 font-mono text-blue-600">
                      {batchTraceParams.skipExisting 
                        ? getAllUniqueEdges().filter(e => !edgeTraceResults.has(getEdgeKey(e))).length
                        : getAllUniqueEdges().length}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Parameter settings */}
            <div className="space-y-3">
              <div className="flex items-center space-x-4">
                <div className="flex-1">
                  <Label htmlFor="batch-side">Side</Label>
                  <Select 
                    value={batchTraceParams.side} 
                    onValueChange={(v) => setBatchTraceParams(p => ({ ...p, side: v as 'q' | 'k' | 'both' }))}
                  >
                    <SelectTrigger id="batch-side">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="both">Both (Q+K)</SelectItem>
                      <SelectItem value="q">Q Side</SelectItem>
                      <SelectItem value="k">K Side</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-1">
                  <Label htmlFor="batch-order">Ordering mode</Label>
                  <Select 
                    value={batchTraceParams.orderMode} 
                    onValueChange={(v) => setBatchTraceParams(p => ({ ...p, orderMode: v as 'positive' | 'negative' }))}
                  >
                    <SelectTrigger id="batch-order">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="positive">Positive</SelectItem>
                      <SelectItem value="negative">Negative</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="batch-max-nodes">Max feature nodes</Label>
                  <Input
                    id="batch-max-nodes"
                    type="number"
                    value={batchTraceParams.max_feature_nodes}
                    onChange={(e) => setBatchTraceParams(p => ({ ...p, max_feature_nodes: parseInt(e.target.value) || 4096 }))}
                  />
                </div>
                <div>
                  <Label htmlFor="batch-max-act">Max activation times</Label>
                  <Input
                    id="batch-max-act"
                    type="number"
                    placeholder="Unlimited"
                    value={batchTraceParams.max_act_times || ''}
                    onChange={(e) => {
                      const val = e.target.value;
                      setBatchTraceParams(p => ({ 
                        ...p, 
                        max_act_times: val ? parseInt(val) : null 
                      }));
                    }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="batch-node-threshold">Node threshold</Label>
                  <Input
                    id="batch-node-threshold"
                    type="number"
                    step="0.01"
                    value={batchTraceParams.node_threshold}
                    onChange={(e) => setBatchTraceParams(p => ({ ...p, node_threshold: parseFloat(e.target.value) || 0.73 }))}
                  />
                </div>
                <div>
                  <Label htmlFor="batch-edge-threshold">Edge threshold</Label>
                  <Input
                    id="batch-edge-threshold"
                    type="number"
                    step="0.01"
                    value={batchTraceParams.edge_threshold}
                    onChange={(e) => setBatchTraceParams(p => ({ ...p, edge_threshold: parseFloat(e.target.value) || 0.57 }))}
                  />
                </div>
              </div>

              {/* Skip edges that already have results */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="batch-skip-existing"
                  checked={batchTraceParams.skipExisting}
                  onChange={(e) => setBatchTraceParams(p => ({ ...p, skipExisting: e.target.checked }))}
                  className="rounded border-gray-300"
                />
                <Label htmlFor="batch-skip-existing" className="text-sm">
                  Skip edges that already have trace results
                </Label>
              </div>
            </div>

            {/* Warning / notes */}
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-yellow-700 text-sm flex items-start">
              <AlertCircle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium">Notes</p>
                <ul className="mt-1 list-disc list-inside text-xs space-y-1">
                  <li>Batch trace will process edges one by one and may take a long time.</li>
                  <li>Each edge typically takes about 10–60 seconds, depending on parameters.</li>
                  <li>You can click "Stop" at any time to abort the remaining edges.</li>
                  <li>Results for already-processed edges are preserved.</li>
                </ul>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowBatchTraceDialog(false)}>
              Cancel
            </Button>
            <Button onClick={startBatchTrace} disabled={!isModelLoaded}>
              <Play className="w-4 h-4 mr-1" />
              Start batch Trace
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Batch Trace progress bar (fixed at bottom) */}
      {isBatchTracing && (
        <div className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg p-4 z-50">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-3">
                <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                <span className="font-medium">Batch Circuit Trace in progress...</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">
                  {batchTraceProgress.current} / {batchTraceProgress.total}
                </span>
                <Button variant="destructive" size="sm" onClick={abortBatchTrace}>
                  <Square className="w-4 h-4 mr-1" />
                  Stop
                </Button>
              </div>
            </div>
            <Progress 
              value={(batchTraceProgress.current / batchTraceProgress.total) * 100} 
              className="h-3"
            />
            <div className="mt-2 flex items-center justify-between text-sm text-gray-500">
              <span>Current: {batchTraceProgress.currentEdge}</span>
              <span>
                Estimated remaining: ~{Math.ceil((batchTraceProgress.total - batchTraceProgress.current) * 30 / 60)} minutes
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchCircuitsVisualization;
