import { useState, useCallback, useMemo } from "react";
import { ChessBoard } from "@/components/chess/chess-board";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Zap, ChevronDown, ChevronRight, Maximize2, Download, Trash2 } from "lucide-react";
import { EdgeCircuitTracePanel, EdgeCircuitTraceResult } from "./edge-circuit-trace-panel";
import { ModelLoadingStatus } from "@/components/shared/model-loading-status";

// 搜索追踪数据类型定义
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

// 用于可视化的树节点
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
  
  // Edge Circuit Trace 相关状态
  const [edgeTraceResults, setEdgeTraceResults] = useState<Map<string, EdgeCircuitTraceResult>>(new Map());
  const [selectedEdgeForTrace, setSelectedEdgeForTrace] = useState<SearchEdge | null>(null);
  const [showEdgeTraceDialog, setShowEdgeTraceDialog] = useState(false);
  const [expandedEdgeTraces, setExpandedEdgeTraces] = useState<Set<string>>(new Set());
  
  // 模型加载状态由 ModelLoadingStatus 组件处理，这里不需要额外的 hook 调用

  // 生成边的唯一键
  const getEdgeKey = useCallback((edge: SearchEdge) => {
    return `${edge.parent}__${edge.child}__${edge.move}`;
  }, []);

  // 构建树结构
  const treeData = useMemo(() => {
    if (!searchData) return null;

    const nodeMap = new Map<string, SearchNode>();
    searchData.nodes.forEach(node => {
      nodeMap.set(node.fen, node);
    });

    // 聚合边信息：对于相同的 (parent, child, move) 组合，取最后一条边的信息
    const edgeMap = new Map<string, SearchEdge>();
    searchData.edges.forEach(edge => {
      const key = `${edge.parent}__${edge.child}__${edge.move}`;
      edgeMap.set(key, edge);
    });

    // 构建子节点映射
    const childrenMap = new Map<string, { child: string; edge: SearchEdge }[]>();
    edgeMap.forEach((edge) => {
      if (!childrenMap.has(edge.parent)) {
        childrenMap.set(edge.parent, []);
      }
      childrenMap.get(edge.parent)!.push({ child: edge.child, edge });
    });

    // 递归构建树
    const buildTree = (fen: string, depth: number, visited: Set<string>): TreeNode | null => {
      if (visited.has(fen)) return null;
      visited.add(fen);

      const node = nodeMap.get(fen);
      if (!node) return null;

      const children: TreeNode[] = [];
      const childEdges = childrenMap.get(fen) || [];

      // 计算该节点的总访问次数
      let totalVisits = 0;
      childEdges.forEach(({ edge }) => {
        totalVisits = Math.max(totalVisits, edge.visits);
      });

      // 找出最佳移动（访问次数最多的）
      let bestMove: string | undefined;
      let maxVisits = 0;
      childEdges.forEach(({ edge }) => {
        if (edge.visits > maxVisits) {
          maxVisits = edge.visits;
          bestMove = edge.move;
        }
      });

      // 按访问次数排序子节点
      const sortedChildEdges = [...childEdges].sort((a, b) => b.edge.visits - a.edge.visits);

      sortedChildEdges.forEach(({ child, edge }) => {
        const childTree = buildTree(child, depth + 1, visited);
        if (childTree) {
          childTree.parentEdge = edge;
          children.push(childTree);
        }
      });

      // 创建简短的FEN用于显示
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

  // 处理文件上传
  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('请上传 JSON 文件');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const text = await file.text();
      const jsonData: SearchTraceData = JSON.parse(text);

      // 验证数据格式
      if (!jsonData.nodes || !jsonData.edges || !jsonData.metadata) {
        throw new Error('无效的搜索追踪 JSON 格式：缺少 nodes、edges 或 metadata 字段');
      }

      setSearchData(jsonData);
      setFileName(file.name);
      setSelectedNode(null);
      setSelectedNodeFen(null);
      setEdgeTraceResults(new Map());
      setExpandedEdgeTraces(new Set());
      
      // 默认展开根节点
      if (jsonData.metadata.root_fen) {
        setExpandedNodes(new Set([jsonData.metadata.root_fen]));
      }
    } catch (err) {
      console.error('加载搜索追踪数据失败:', err);
      setError(err instanceof Error ? err.message : '加载搜索追踪数据失败');
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

  // 切换节点展开/折叠
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

  // 选择节点
  const selectNode = useCallback((fen: string) => {
    if (!searchData) return;
    const node = searchData.nodes.find(n => n.fen === fen);
    setSelectedNode(node || null);
    setSelectedNodeFen(fen);
  }, [searchData]);

  // 获取节点的入边信息
  const getNodeInEdges = useCallback((fen: string) => {
    if (!searchData) return [];
    return searchData.edges.filter(e => e.child === fen);
  }, [searchData]);

  // 获取节点的出边信息
  const getNodeOutEdges = useCallback((fen: string) => {
    if (!searchData) return [];
    // 聚合相同 (parent, child, move) 的边
    const edgeMap = new Map<string, SearchEdge>();
    searchData.edges.filter(e => e.parent === fen).forEach(edge => {
      const key = `${edge.child}__${edge.move}`;
      edgeMap.set(key, edge);
    });
    return Array.from(edgeMap.values()).sort((a, b) => b.visits - a.visits);
  }, [searchData]);

  // 处理边的 Circuit Trace 完成
  const handleEdgeTraceComplete = useCallback((result: EdgeCircuitTraceResult) => {
    setEdgeTraceResults(prev => {
      const next = new Map(prev);
      next.set(result.edgeKey, result);
      return next;
    });
  }, []);

  // 切换边 trace 展开状态
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

  // 打开边 trace 对话框
  const openEdgeTraceDialog = useCallback((edge: SearchEdge) => {
    setSelectedEdgeForTrace(edge);
    setShowEdgeTraceDialog(true);
  }, []);

  // 导出所有边 trace 结果
  const exportAllEdgeTraces = useCallback(() => {
    if (edgeTraceResults.size === 0) {
      alert('没有可导出的边 Trace 结果');
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

  // 清除所有边 trace 结果
  const clearAllEdgeTraces = useCallback(() => {
    if (confirm('确定要清除所有边 Trace 结果吗？')) {
      setEdgeTraceResults(new Map());
      setExpandedEdgeTraces(new Set());
    }
  }, []);

  // 渲染树节点
  const renderTreeNode = useCallback((node: TreeNode, isRoot: boolean = false): React.ReactNode => {
    const isExpanded = expandedNodes.has(node.fen);
    const isSelected = selectedNodeFen === node.fen;
    const hasChildren = node.children.length > 0;

    // 根据访问次数确定颜色强度
    const maxVisits = treeData?.totalVisits || 1;
    const visitRatio = node.totalVisits / Math.max(maxVisits, 1);
    const bgOpacity = Math.max(0.1, Math.min(0.8, visitRatio));

    // 检查该节点的父边是否有 trace 结果
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
          {/* 展开/折叠按钮 */}
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

          {/* 节点信息 */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              {isRoot && (
                <span className="px-2 py-0.5 bg-green-500 text-white text-xs rounded-full">根节点</span>
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
              <span>深度: {node.depth}</span>
              <span>移动数: {node.moves.length}</span>
              {node.parentEdge && (
                <>
                  <span>访问: {node.parentEdge.visits}</span>
                  <span>Q: {node.parentEdge.q.toFixed(3)}</span>
                  <span>得分: {node.parentEdge.score.toFixed(3)}</span>
                </>
              )}
              {node.is_tt_hit !== null && (
                <span className={node.is_tt_hit ? 'text-green-600' : 'text-gray-400'}>
                  {node.is_tt_hit ? 'TT命中' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Circuit Trace 按钮 */}
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
              title="Circuit Trace 此边"
            >
              <Zap className="w-4 h-4" />
            </button>
          )}

          {/* 最佳移动标记 */}
          {node.bestMove && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded border border-yellow-300">
              最佳: {node.bestMove}
            </span>
          )}
        </div>

        {/* 子节点 */}
        {isExpanded && hasChildren && (
          <div className="border-l-2 border-gray-200 ml-3">
            {node.children.map(child => renderTreeNode(child, false))}
          </div>
        )}
      </div>
    );
  }, [expandedNodes, selectedNodeFen, selectNode, toggleNode, treeData, edgeTraceResults, getEdgeKey, openEdgeTraceDialog]);

  // 渲染边 trace 列表
  const renderEdgeTraceList = () => {
    if (edgeTraceResults.size === 0) return null;

    return (
      <div className="bg-white rounded-lg border p-4 mt-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center">
            <Zap className="w-5 h-5 mr-2 text-purple-500" />
            边 Circuit Trace 结果 ({edgeTraceResults.size})
          </h3>
          <div className="flex space-x-2">
            <Button variant="outline" size="sm" onClick={exportAllEdgeTraces}>
              <Download className="w-4 h-4 mr-1" />
              导出全部
            </Button>
            <Button variant="outline" size="sm" onClick={clearAllEdgeTraces}>
              <Trash2 className="w-4 h-4 mr-1" />
              清除全部
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
                      {result.traceResult?.nodes?.length || 0} 节点
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
                        // 找到对应的边并打开对话框
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
                        <span className="font-medium text-gray-700">参数:</span>
                        <div className="mt-1 text-xs space-y-1">
                          <div>Side: {result.side}</div>
                          <div>Order Mode: {result.orderMode}</div>
                          <div>Max Nodes: {result.params.max_feature_nodes}</div>
                          <div>Node Threshold: {result.params.node_threshold}</div>
                          <div>Edge Threshold: {result.params.edge_threshold}</div>
                        </div>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">结果统计:</span>
                        <div className="mt-1 text-xs space-y-1">
                          <div>节点数: {result.traceResult?.nodes?.length || 0}</div>
                          <div>连接数: {result.traceResult?.links?.length || 0}</div>
                          <div>目标移动: {result.traceResult?.metadata?.target_move || result.move}</div>
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

  // 错误状态
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-red-600 mb-2">加载失败</h3>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  // 加载状态
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">正在加载搜索追踪数据...</p>
        </div>
      </div>
    );
  }

  // 未上传文件状态
  if (!searchData) {
    return (
      <div className="space-y-6">
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
                上传搜索追踪数据
              </h3>
              <p className="text-gray-600 mb-4">
                拖拽 JSON 文件到此处，或点击选择文件
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
                选择文件
              </label>
            </div>
            <p className="text-sm text-gray-500">
              支持 search_trace_*.json 格式的搜索追踪文件
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 主视图
  return (
    <div className="space-y-6">
      {/* 头部信息 */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">文件: {fileName}</span>
          <span className="px-2 py-1 bg-green-100 text-green-800 text-sm rounded">
            最佳移动: {searchData.metadata.search_results.best_move}
          </span>
          {edgeTraceResults.size > 0 && (
            <span className="px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded flex items-center">
              <Zap className="w-3 h-3 mr-1" />
              {edgeTraceResults.size} 边已 Trace
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {/* 模型加载状态按钮 */}
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
            上传新文件
          </button>
        </div>
      </div>

      {/* 元数据卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 搜索参数 */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">搜索参数</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">最大 Playouts:</span>
              <span className="font-mono">{searchData.metadata.search_params.max_playouts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">最大深度:</span>
              <span className="font-mono">{searchData.metadata.search_params.max_depth}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">CPUCT:</span>
              <span className="font-mono">{searchData.metadata.search_params.cpuct}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Minibatch 大小:</span>
              <span className="font-mono">{searchData.metadata.search_params.target_minibatch_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">FPU 值:</span>
              <span className="font-mono">{searchData.metadata.search_params.fpu_value}</span>
            </div>
          </div>
        </div>

        {/* 搜索结果 */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">搜索结果</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">最佳移动:</span>
              <span className="font-mono font-bold text-green-600">{searchData.metadata.search_results.best_move}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">总 Playouts:</span>
              <span className="font-mono">{searchData.metadata.search_results.total_playouts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">达到深度:</span>
              <span className="font-mono">{searchData.metadata.search_results.max_depth}</span>
            </div>
          </div>
        </div>

        {/* 追踪统计 */}
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">追踪统计</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">节点数:</span>
              <span className="font-mono">{searchData.nodes.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">边记录数:</span>
              <span className="font-mono">{searchData.metadata.trace_stats.num_edge_records}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">扩展记录数:</span>
              <span className="font-mono">{searchData.metadata.trace_stats.num_expansion_records}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">导出时间:</span>
              <span className="font-mono text-xs">{searchData.metadata.export_timestamp}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 主要内容区域 */}
      <div className="flex gap-6 h-[700px]">
        {/* 左侧：搜索树 */}
        <div className="flex-1 bg-white rounded-lg border p-4 overflow-hidden flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">搜索树</h3>
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
                全部展开
              </button>
              <button
                onClick={() => {
                  if (searchData) {
                    setExpandedNodes(new Set([searchData.metadata.root_fen]));
                  }
                }}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                全部折叠
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {treeData && renderTreeNode(treeData, true)}
          </div>
        </div>

        {/* 右侧：节点详情 */}
        <div className="w-[500px] bg-white rounded-lg border p-4 overflow-hidden flex flex-col">
          <h3 className="text-lg font-semibold mb-4">节点详情</h3>
          {selectedNode ? (
            <div className="flex-1 overflow-y-auto space-y-4">
              {/* 棋盘显示 */}
              <div className="flex justify-center">
                <ChessBoard
                  fen={selectedNode.fen}
                  size="medium"
                  showCoordinates={true}
                  move={searchData.metadata.search_results.best_move}
                />
              </div>

              {/* FEN 字符串 */}
              <div className="bg-gray-50 rounded p-3">
                <div className="text-sm font-medium text-gray-700 mb-1">FEN 字符串:</div>
                <div className="font-mono text-xs break-all select-all">
                  {selectedNode.fen}
                </div>
              </div>

              {/* 入边信息 */}
              {getNodeInEdges(selectedNode.fen).length > 0 && (
                <div className="bg-blue-50 rounded p-3">
                  <div className="text-sm font-medium text-blue-700 mb-2">入边信息:</div>
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
                            <span>Visits:{edge.visits}</span>
                            <span>Q:{edge.q.toFixed(3)}</span>
                            <span>U:{edge.u.toFixed(3)}</span>
                            <span>S:{edge.score.toFixed(3)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* 可选移动和策略 */}
              {selectedNode.moves.length > 0 && (
                <div className="bg-green-50 rounded p-3">
                  <div className="text-sm font-medium text-green-700 mb-2">
                    可选移动 ({selectedNode.moves.length}):
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

              {/* 出边信息 */}
              {getNodeOutEdges(selectedNode.fen).length > 0 && (
                <div className="bg-orange-50 rounded p-3">
                  <div className="text-sm font-medium text-orange-700 mb-2">
                    出边信息 ({getNodeOutEdges(selectedNode.fen).length}):
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
                              <span className="text-orange-600">→ 跳转</span>
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
                            <span>V:{edge.visits}</span>
                            <span>Q:{edge.q.toFixed(3)}</span>
                            <span>U:{edge.u.toFixed(3)}</span>
                            <span>S:{edge.score.toFixed(3)}</span>
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
                <p>点击搜索树中的节点查看详情</p>
                <p className="text-sm mt-2">点击 <Zap className="inline w-4 h-4 text-purple-500" /> 按钮进行 Circuit Trace</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 边 Trace 结果列表 */}
      {renderEdgeTraceList()}

      {/* 根节点棋盘 */}
      <div className="bg-white rounded-lg border p-4">
        <h3 className="text-lg font-semibold mb-4">根节点局面</h3>
        <div className="flex justify-center">
          <ChessBoard
            fen={searchData.metadata.root_fen}
            size="large"
            showCoordinates={true}
            move={searchData.metadata.search_results.best_move}
          />
        </div>
      </div>

      {/* Edge Circuit Trace 对话框 */}
      <Dialog open={showEdgeTraceDialog} onOpenChange={setShowEdgeTraceDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-purple-500" />
              <span>边 Circuit Trace</span>
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
    </div>
  );
};

export default SearchCircuitsVisualization;
