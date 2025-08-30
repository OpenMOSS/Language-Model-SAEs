import React, { useEffect, useState, useMemo, useRef, Suspense, lazy } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, FileJson, Network, Activity } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useAsyncFn } from "react-use";



interface CircuitNode {
  id: string;
  type: string;
  sae?: string;
  featureIndex?: number;
  position?: number;
  activation?: number;
  maxActivation?: number;
  tokenId?: number;
  layer?: number;
  head?: number;
  query?: number;
  key?: number;
  pattern?: number;
}

interface CircuitEdge {
  source: string;
  target: string;
  attribution: number;
}

interface CircuitData {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  metadata?: {
    prompt_tokens?: string[];
  };
}

export const CircuitsPage: React.FC = () => {
  const [circuitData, setCircuitData] = useState<CircuitData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<CircuitNode | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; node: CircuitNode } | null>(null);
  const [highlightedEdges, setHighlightedEdges] = useState<Set<string>>(new Set());
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [highlightedInEdges, setHighlightedInEdges] = useState<Set<string>>(new Set());
  const [highlightedInNodes, setHighlightedInNodes] = useState<Set<string>>(new Set());
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null); // 新增：跟踪从节点列表选中的节点
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [nodeSamples, setNodeSamples] = useState<any[]>([]);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [scanning, setScanning] = useState<boolean>(false);
  const [miniScale, setMiniScale] = useState<number>(1);
  const [fitWidth, setFitWidth] = useState<boolean>(true);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeListRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(0);
  const [rememberedFile, setRememberedFile] = useState<string | null>(null);
  const [showInfluenceOnly, setShowInfluenceOnly] = useState<boolean>(false);
  const navigate = useNavigate();

  // 新增：归因输入状态
  const [fenInput, setFenInput] = useState<string>("");
  const [moveInput, setMoveInput] = useState<string>("");
  const [generating, setGenerating] = useState<boolean>(false);
  const [sideInput, setSideInput] = useState<string>("k");

  // 日志流
  const [logs, setLogs] = useState<Array<{ ts: number; level: string; msg: string }>>([]);
  const [logStreaming, setLogStreaming] = useState<boolean>(false);
  const evtSourceRef = useRef<EventSource | null>(null);

  const startLogStream = () => {
    if (evtSourceRef.current) return;
    const backendBase = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/$/, "") || "";
    const url = `${backendBase}/logs/stream`;
    const es = new EventSource(url);
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setLogs((prev) => [...prev.slice(-499), { ts: data.ts, level: data.level, msg: data.msg }]);
      } catch {}
    };
    es.onerror = () => {
      es.close();
      evtSourceRef.current = null;
      setLogStreaming(false);
    };
    evtSourceRef.current = es;
    setLogStreaming(true);
  };

  const stopLogStream = () => {
    if (evtSourceRef.current) {
      evtSourceRef.current.close();
      evtSourceRef.current = null;
    }
    setLogStreaming(false);
  };

  useEffect(() => {
    return () => stopLogStream();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setLoading(true);
      setError(null);
      setSelectedFileName(file.name);
      const text = await file.text();
      const json = JSON.parse(text);
      const transformed = transformCircuitData(json);
      setCircuitData(transformed);
      setSelectedNode(null);
      setSelectedNodeId(null);
      
      // 保存本地文件选择到本地存储（使用文件名作为标识）
      const localFileKey = `local_file_${file.name}`;
      localStorage.setItem('lastSelectedCircuitFile', localFileKey);
      setRememberedFile(localFileKey);
      console.log('💾 保存本地文件选择到本地存储:', localFileKey);
    } catch (err) {
      console.error("本地文件解析失败:", err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateFromFEN = async () => {
    if (!fenInput || fenInput.trim().length === 0) return;
    try {
      setGenerating(true);
      setError(null);
      const backendBase = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/$/, "") || "";
      const resp = await fetch(`${backendBase}/circuits/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          fen: fenInput.trim(),
          move_uci: moveInput.trim() || undefined,
          side: sideInput,
          node_threshold: 0.5,
          edge_threshold: 0.3,
          sae_series: undefined,
          save: false, // 不保存到后端文件系统
        }),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${txt}`);
      }
      const data = await resp.json();
      const graph = data?.graph ?? data;
      const transformed = transformCircuitData(graph);
      setCircuitData(transformed);
      setSelectedNode(null);
      setSelectedNodeId(null);
      // 不再保存服务器生成的文件路径，避免跨机访问失败
    } catch (err) {
      console.error("生成电路失败:", err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  };

  // 扫描 /circuits 下可用的 JSON 文件
  const scanCircuits = async (): Promise<string[]> => {
    setScanning(true);
    try {
      // 直接读取 index.json 文件
      try {
        const response = await fetch("/circuits/index.json", { 
          headers: { Accept: "application/json" } 
        });
        if (response.ok) {
          const files = await response.json();
          if (Array.isArray(files)) {
            // 规范化为以 /circuits/ 开头的路径
            const normalized = files
              .map((p) => (p.startsWith("/circuits/") ? p : p.startsWith("/") ? p : `/circuits/${p}`))
              .filter((p) => p.endsWith(".json"));
            return Array.from(new Set(normalized));
          }
        }
      } catch (error) {
        console.error("读取 index.json 失败:", error);
      }

      // 如果 index.json 读取失败，使用兜底探测
      const probe = async (p: string) => {
        try {
          const r = await fetch(p, { method: "HEAD" });
          return r.ok;
        } catch {
          return false;
        }
      };
      
      // 从 index.json 中获取文件列表进行探测
      const indexFiles = [
        "win_or_go_home_k_1024.json",
        "win_or_go_home_q_1024.json",
        "win_or_go_home_k_1024_encoder_demean.json",
        "win_or_go_home_incorrect_k_negative_1024.json",
        "sacrifice_the_queen_k_1024.json",
        "sacrifice_the_queen_k_4096_B.json",
        "sacrifice_the_queen_in3_k_4096_B.json",
        "win_or_go_home_k_move_pair_1024.json",
        "win_or_go_home_k_group_1024.json",
        "win_or_go_home_k_2048_G.json",
        "win_or_go_home_k_4096_G.json",
        "win_or_go_home_k_4096_B.json"
        // "carlson_badmove_k_negative_1024.json",
      ];
      
      const found: string[] = [];
      await Promise.all(
        indexFiles.map(async (file) => {
          const path = `/circuits/${file}`;
          if (await probe(path)) found.push(path);
        })
      );
      return found;
    } finally {
      setScanning(false);
    }
  };

  const loadCircuitFromPath = async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const tryFetch = async (url: string) => {
        const res = await fetch(url, { headers: { Accept: "application/json" } });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const first = text.trim().charAt(0);
        if (!(first === "{" || first === "[")) throw new Error("非JSON响应");
        return JSON.parse(text);
      };

      let json: any = null;
      try {
        json = await tryFetch(path);
      } catch (e1) {
        // 如果是以 /circuits/ 开头的本地静态路径，避免去请求后端，重新扫描并随机选择一个可用文件
        if (path.startsWith('/circuits/')) {
          try {
            // 清除失效的本地记忆，避免下次仍然尝试相同文件
            const last = localStorage.getItem('lastSelectedCircuitFile');
            if (last === path) localStorage.removeItem('lastSelectedCircuitFile');
            // 重新扫描以获取真实可用列表
            const fresh = await scanCircuits();
            setAvailableFiles(fresh);
            if (fresh.length > 0) {
              const randomIdx = Math.floor(Math.random() * fresh.length);
              const fallbackPath = fresh[randomIdx];
              console.warn(`无法加载 ${path}，改为随机加载: ${fallbackPath}`);
              json = await tryFetch(fallbackPath);
              path = fallbackPath; // 将当前路径更新为成功加载的路径
            } else {
              throw e1;
            }
          } catch (e2) {
            throw e1;
          }
        } else {
          // 其他情况再尝试后端兜底
          const backendBase = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/$/, "");
          const backendUrl = backendBase ? `${backendBase}${path}` : null;
          if (!backendUrl) throw e1;
          json = await tryFetch(backendUrl);
        }
      }
      const transformed = transformCircuitData(json);
      setCircuitData(transformed);
      setSelectedNode(null);
      setSelectedNodeId(null);
      
      // 保存选择的文件到本地存储（仅保存路径，不触发跨机访问）
      localStorage.setItem('lastSelectedCircuitFile', path);
      setRememberedFile(path);
      console.log('💾 保存选择的电路文件到本地存储:', path);
      
      // 保存显示影响状态到本地存储
      localStorage.setItem('circuitsShowInfluenceOnly', showInfluenceOnly.toString());
      console.log('💾 保存显示影响状态到本地存储:', showInfluenceOnly);
      
      // 如果有选中的节点，也保存到本地存储
      if (selectedNode) {
        localStorage.setItem('circuitsSelectedNodeId', selectedNode.id);
        console.log('💾 保存选中节点ID到本地存储:', selectedNode.id);
      }
    } catch (err) {
      console.error("加载电路文件失败:", err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  // 首次扫描并加载默认文件
  useEffect(() => {
    (async () => {
      const files = await scanCircuits();
      setAvailableFiles(files);
      
      // 设置记住的文件状态
      const lastSelectedFile = localStorage.getItem('lastSelectedCircuitFile');
      setRememberedFile(lastSelectedFile);
      
      // 从本地存储恢复显示影响状态
      const savedShowInfluenceOnly = localStorage.getItem('circuitsShowInfluenceOnly');
      if (savedShowInfluenceOnly !== null) {
        setShowInfluenceOnly(savedShowInfluenceOnly === 'true');
        console.log('🔄 从本地存储恢复显示影响状态:', savedShowInfluenceOnly === 'true');
      }
      
      // 从本地存储恢复选中的节点ID（但需要等circuitData加载后再恢复节点）
      const savedSelectedNodeId = localStorage.getItem('circuitsSelectedNodeId');
      if (savedSelectedNodeId) {
        console.log('🔄 从本地存储恢复选中节点ID:', savedSelectedNodeId);
      }
      
      // 检查是否有返回状态，优先加载返回的文件
      const urlParams = new URLSearchParams(window.location.search);
      const returnStateParam = urlParams.get('returnState');
      
      let fileToLoad = null;
      if (returnStateParam) {
        try {
          const returnState = JSON.parse(decodeURIComponent(returnStateParam));
          console.log('🔄 检测到返回状态，优先加载文件:', returnState.circuitFile);
          
          if (returnState.circuitFile && files.includes(returnState.circuitFile)) {
            fileToLoad = returnState.circuitFile;
          }
        } catch (error) {
          console.error('解析返回状态失败:', error);
        }
      }
      
      // 如果没有返回文件，尝试从本地存储加载上次选择的文件
      if (!fileToLoad) {
        if (lastSelectedFile) {
          // 检查是否是本地文件
          if (lastSelectedFile.startsWith('local_file_')) {
            console.log('🔄 检测到上次选择的是本地文件:', lastSelectedFile);
            // 不自动加载本地文件，因为文件可能不在当前会话中
          } else if (files.includes(lastSelectedFile)) {
            console.log('🔄 从本地存储加载上次选择的文件:', lastSelectedFile);
            fileToLoad = lastSelectedFile;
          }
        }
      }
      
      // 如果本地存储也没有或不可用，随机选择一个已有文件
      if (!fileToLoad) {
        if (files.length > 0) {
          const randomIdx = Math.floor(Math.random() * files.length);
          fileToLoad = files[randomIdx];
          console.log('🔄 未找到记忆文件，随机加载:', fileToLoad);
        } else {
          fileToLoad = null;
        }
      }
      
      if (fileToLoad) {
        setCurrentFile(fileToLoad);
        await loadCircuitFromPath(fileToLoad);
      } else {
        // 若未发现任何文件，保持现有状态（可能用户稍后手动选择本地文件）
      }
      
      // 如果没有返回状态，尝试从本地存储恢复选中的节点
      if (!returnStateParam && savedSelectedNodeId) {
        // 等待circuitData加载完成后再恢复节点
        setTimeout(() => {
          if (circuitData) {
            const node = circuitData.nodes.find(n => n.id === savedSelectedNodeId);
            if (node) {
              setSelectedNode(node);
              setSelectedNodeId(node.id);
              console.log('✅ 从本地存储恢复选中节点:', node.id);
            }
          }
        }, 100);
      }
    })();
  }, []);

  // 监听树状图容器宽度变化，用于"适应宽度"
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        setContainerWidth(w);
      }
    });
    ro.observe(el);
    // 初始化
    setContainerWidth(el.clientWidth);
    return () => ro.disconnect();
  }, [containerRef.current]);

  // 缩放或容器变化时水平居中显示（当内容宽度大于容器宽度时）
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const contentW = (fitWidth && containerWidth ? containerWidth : miniWidth * miniScale);
    const viewportW = el.clientWidth;
    if (contentW > viewportW) {
      el.scrollLeft = Math.max(0, (contentW - viewportW) / 2);
    } else {
      el.scrollLeft = 0; // 居中由 margin:auto 处理
    }
  }, [miniScale, fitWidth, containerWidth]);

  // 转换电路数据格式
  const transformCircuitData = (json: any): CircuitData => {
    // 如果已经是正确的格式，直接返回
    if (json.nodes && json.edges) {
      return json;
    }

    // 如果是nodes和links格式，转换links为edges
    if (json.nodes && json.links) {
      const parseFeatureIndex = (n: any): number | undefined => {
        if (typeof n?.node_id === 'string') {
          const parts = n.node_id.split('_');
          if (parts.length >= 2) {
            const idx = parseInt(parts[1], 10);
            if (!Number.isNaN(idx)) return idx;
          }
        }
        if (typeof n?.feature === 'number') return n.feature;
        return undefined;
      };

      const mapType = (n: any): string => {
        if (n?.is_target_logit) return 'logits';
        return (n?.feature_type || 'feature').toLowerCase();
      };

      const nodes: CircuitNode[] = json.nodes.map((n: any) => ({
        id: n.node_id || n.id,
        type: mapType(n),
        sae: n.feature_type,
        featureIndex: parseFeatureIndex(n),
        position: n.ctx_idx,
        activation: n.activation ?? undefined,
        maxActivation: (typeof n.activation === 'number' ? Math.max(n.activation, 1) : 1),
        tokenId: n.ctx_idx,
        layer: n.layer,
        head: undefined,
        query: undefined,
        key: undefined,
        pattern: n.influence ?? undefined,
      }));

      const edges: CircuitEdge[] = json.links.map((link: any) => ({
        source: link.source,
        target: link.target,
        attribution: link.weight ?? 0,
      }));

      return { nodes, edges, metadata: json.metadata };
    }

    // 如果是旧的tracings格式，转换
    if (json.tracings && Array.isArray(json.tracings)) {
      const nodes: CircuitNode[] = [];
      const edges: CircuitEdge[] = [];
      const nodeMap = new Map<string, CircuitNode>();

      json.tracings.forEach((tracing: any) => {
        // 添加主节点
        const mainNode: CircuitNode = {
          id: tracing.node.id,
          type: tracing.node.type,
          sae: tracing.node.sae,
          featureIndex: tracing.node.featureIndex,
          position: tracing.node.position,
          activation: tracing.node.activation ?? undefined,
          maxActivation: tracing.node.maxActivation ?? 1,
          tokenId: tracing.node.tokenId,
          layer: tracing.node.layer,
          head: tracing.node.head,
          query: tracing.node.query,
          key: tracing.node.key,
          pattern: tracing.node.pattern ?? undefined,
        };
        nodes.push(mainNode);
        nodeMap.set(mainNode.id, mainNode);

        // 添加贡献者节点和边
        tracing.contributors.forEach((contributor: any) => {
          const contributorNode: CircuitNode = {
            id: contributor.node.id,
            type: contributor.node.type,
            sae: contributor.node.sae,
            featureIndex: contributor.node.featureIndex,
            position: contributor.node.position,
            activation: contributor.node.activation ?? undefined,
            maxActivation: contributor.node.maxActivation ?? 1,
            tokenId: contributor.node.tokenId,
            layer: contributor.node.layer,
            head: contributor.node.head,
            query: contributor.node.query,
            key: contributor.node.key,
            pattern: contributor.node.pattern ?? undefined,
          };

          if (!nodeMap.has(contributorNode.id)) {
            nodes.push(contributorNode);
            nodeMap.set(contributorNode.id, contributorNode);
          }

          edges.push({
            source: contributor.node.id,
            target: tracing.node.id,
            attribution: contributor.attribution ?? 0,
          });
        });
      });

      return { nodes, edges, metadata: json.metadata };
    }

    // 如果是其他格式，尝试适配
    return {
      nodes: [],
      edges: [],
      metadata: { prompt_tokens: [] },
    };
  };

  // 获取节点标签
  const getNodeLabel = (node: CircuitNode): string => {
    if (node.type === "feature" || node.type === "transcoder") {
      return `${node.position}.${node.sae}.${node.featureIndex}`;
    } else if (node.type === "attn-score") {
      return `L${node.layer}H${node.head}Q${node.query}K${node.key}`;
    } else if (node.type === "logits") {
      return `${node.position}.logits.${node.tokenId}`;
    } else if (node.type === "embedding") {
      return `E_${node.position}`;
    } else if (node.type === "error") {
      return `Error_${node.position}`;
    }
    return node.id;
  };

  // 获取节点颜色
  const getNodeColor = (node: CircuitNode): string => {
    if (node.type === "feature" || node.type === "transcoder") {
      return "bg-red-500"; // 所有feature节点恒为红色
    } else if (node.type === "logits") {
      return "bg-purple-500";
    } else if (node.type === "attn-score") {
      return "bg-green-500";
    } else if (node.type === "embedding") {
      return "bg-indigo-500";
    } else if (node.type === "error") {
      return "bg-gray-400"; // 只有error为灰色
    }
    return "bg-blue-400";
  };

  // 小图用的颜色（SVG fill）
  const getNodeFill = (node: CircuitNode): string => {
    if (node.type === "feature" || node.type === "transcoder") {
      // 使用节点影响力（pattern 字段承载 influence）映射红色深浅
      const influence = typeof node.pattern === 'number' ? node.pattern : 0;
      const s = Math.max(0, Math.min(1, influence));
      if (s > 0.8) return "#b91c1c"; // red-700 深红
      if (s > 0.6) return "#dc2626"; // red-600
      if (s > 0.4) return "#ef4444"; // red-500
      if (s > 0.2) return "#f87171"; // red-400
      return "#fecaca"; // red-200 浅红
    }
    if (node.type === "logits") return "#a855f7"; // purple-500
    if (node.type === "attn-score") return "#22c55e"; // green-500
    if (node.type === "embedding") return "#6366f1"; // indigo-500
    if (node.type === "error") return "#9ca3af"; // gray-400
    return "#9ca3af"; // gray-400
  };

  const isFeatureNode = (n: CircuitNode) => n.type === "feature" || n.type === "transcoder";
  
  // 计算子图：包括选定节点及其上游影响的节点和边（保留完整子图结构）
  const calculateSubgraph = (selectedNode: CircuitNode | null): { nodes: CircuitNode[], edges: CircuitEdge[] } => {
    if (!selectedNode || !circuitData) {
      return { nodes: [], edges: [] };
    }
    
    const subgraphNodes = new Set<string>([selectedNode.id]);
    const queue = [selectedNode.id];
    
    // 第一步：广度优先搜索，找到所有影响选定节点的上游节点
    while (queue.length > 0) {
      const currentNodeId = queue.shift()!;
      
      // 找到所有指向当前节点的边（上游影响）
      const incomingEdges = circuitData.edges.filter(edge => edge.target === currentNodeId);
      
      for (const edge of incomingEdges) {
        // 添加源节点（上游节点）
        const sourceNodeId = edge.source;
        if (!subgraphNodes.has(sourceNodeId)) {
          subgraphNodes.add(sourceNodeId);
          queue.push(sourceNodeId);
        }
      }
    }
    
    // 第二步：保留子图中所有节点之间的完整连接关系
    const subgraphEdges = circuitData.edges.filter(edge => 
      subgraphNodes.has(edge.source) && subgraphNodes.has(edge.target)
    );
    
    // 返回完整的子图
    const nodes = circuitData.nodes.filter(node => subgraphNodes.has(node.id));
    const edges = subgraphEdges;
    
    return { nodes, edges };
  };
  
  // 统计信息
  const stats = circuitData ? {
    totalNodes: showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).nodes.length : circuitData.nodes.length,
    totalEdges: showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).edges.length : circuitData.edges.length,
    nodeTypes: (showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).nodes : circuitData.nodes).reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>),
    uniqueSAEs: new Set((showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).nodes : circuitData.nodes).filter(n => n.sae).map(n => n.sae)).size,
  } : null;
  
  // 计算节点的高亮边和目标节点
  const calculateHighlights = (node: CircuitNode) => {
    if (!circuitData) return { 
      edges: new Set<string>(), 
      nodes: new Set<string>(),
      inEdges: new Set<string>(),
      inNodes: new Set<string>()
    };
    
    // 只对 logits 和 feature 节点进行高亮
    if (node.type !== 'logits' && !isFeatureNode(node)) {
      return { 
        edges: new Set<string>(), 
        nodes: new Set<string>(),
        inEdges: new Set<string>(),
        inNodes: new Set<string>()
      };
    }
    
    // 找出从这个节点出发的边（下游影响）
    const outgoingEdges = circuitData.edges.filter(edge => edge.source === node.id);
    
    // 找出指向这个节点的边（上游影响）
    const incomingEdges = circuitData.edges.filter(edge => edge.target === node.id);
    
    // 按影响权重排序，取前5条
    const sortedOutEdges = outgoingEdges
      .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
      .slice(0, 5);
    
    const sortedInEdges = incomingEdges
      .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
      .slice(0, 5);
    
    const edgeIds = new Set(sortedOutEdges.map(edge => `${edge.source}-${edge.target}`));
    const targetNodeIds = new Set(sortedOutEdges.map(edge => edge.target));
    
    const inEdgeIds = new Set(sortedInEdges.map(edge => `${edge.source}-${edge.target}`));
    const sourceNodeIds = new Set(sortedInEdges.map(edge => edge.source));
    
    return { 
      edges: edgeIds, 
      nodes: targetNodeIds,
      inEdges: inEdgeIds,
      inNodes: sourceNodeIds
    };
  };

  const navigateToFeature = (node: CircuitNode) => {
    if (!isFeatureNode(node) || node.featureIndex === undefined) return;
    const series = (circuitData?.metadata as any)?.sae_series || node.sae || "lc0";
    const base = typeof series === 'string' && series.includes('-') ? series.split('-')[0] : (series || 'lc0');
    if (typeof node.layer !== 'number') return;
    const dict = `${base}_L${node.layer - 1}M_16x_k30_lr2e-03_auxk_sparseadam`;
    
    // 构建返回状态信息
    const returnState = {
      from: 'circuits',
      circuitFile: currentFile,
      selectedNodeId: node.id,
      hoveredNodeId: hoveredNodeId,
      miniScale: miniScale,
      fitWidth: fitWidth,
      showInfluenceOnly: showInfluenceOnly, // 记忆显示影响状态
      // 记录当前容器与列表的滚动位置
      containerScrollLeft: containerRef.current?.scrollLeft ?? 0,
      containerScrollTop: containerRef.current?.scrollTop ?? 0,
      nodeListScrollTop: nodeListRef.current?.scrollTop ?? 0,
      nodeInfo: {
        id: node.id,
        type: node.type,
        layer: node.layer,
        position: node.position,
        featureIndex: node.featureIndex,
        sae: node.sae,
        pattern: node.pattern
      }
    };
    
    // 使用URLSearchParams构建URL，确保参数正确编码
    const params = new URLSearchParams();
    params.set('from', 'circuits');
    params.set('dictionary', dict);
    params.set('featureIndex', node.featureIndex.toString());
    params.set('returnState', JSON.stringify(returnState));
    
    const jumpUrl = `/features?${params.toString()}`;
    console.log('🔄 跳转到feature页面:', jumpUrl);
    console.log('🔄 URL参数详情:', {
      from: params.get('from'),
      dictionary: params.get('dictionary'),
      featureIndex: params.get('featureIndex'),
      returnState: params.get('returnState')
    });
    navigate(jumpUrl);
  };

  // 获取节点的激活样本
  const [samplesState, fetchNodeSamples] = useAsyncFn(async (node: CircuitNode) => {
    if (!isFeatureNode(node) || node.featureIndex === undefined) return;
    
    const series = (circuitData?.metadata as any)?.sae_series || node.sae || "lc0";
    const base = typeof series === 'string' && series.includes('-') ? series.split('-')[0] : (series || 'lc0');
    if (typeof node.layer !== 'number') return;
    const dict = `${base}_L${node.layer - 1}M_16x_k30_lr2e-03_auxk_sparseadam`;
    
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dict}/features/${node.featureIndex}/samples?limit=4`,
      {
        headers: {
          Accept: "application/x-msgpack",
        },
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const decoded = decode(new Uint8Array(arrayBuffer));
    const samples = camelcaseKeys(decoded as any, {
      deep: true,
      stopPaths: ["context"],
    });

    setNodeSamples(samples);
    return samples;
  });

  const { miniNodes, miniEdges, miniWidth, miniHeight } = useMemo(() => {
    // 根据子图模式选择要显示的数据
    const displayData = showInfluenceOnly && selectedNode 
      ? calculateSubgraph(selectedNode) 
      : circuitData;
    
    if (!displayData || displayData.nodes.length === 0) {
      return { 
        miniNodes: [] as Array<{ id: string; x: number; y: number; n: CircuitNode }>, 
        miniEdges: [] as Array<{ x1: number; y1: number; x2: number; y2: number; source: string; target: string; attribution: number }>, 
        miniWidth: 0, 
        miniHeight: 0 
      };
    }

    const getOrder = (n: CircuitNode): number => {
      const t = (n.type || '').toLowerCase();
      const isLogits = t === 'logits' || t === 'logit';
      if (isLogits) return 1000;
      if (typeof n.layer === 'number') return n.layer;
      // embeddings 或未知，排到底部
      return -10;
    };

    // 分层：按 order 降序（logits 最上，随后 layer 从高到低）
    const layerGroups = new Map<number, CircuitNode[]>();
    displayData.nodes.forEach(n => {
      const o = getOrder(n);
      if (!layerGroups.has(o)) layerGroups.set(o, []);
      layerGroups.get(o)!.push(n);
    });
    const orders = Array.from(layerGroups.keys()).sort((a,b)=>b-a);

    // 根据层级与每层节点数动态计算画布尺寸
    const paddingX = 24;
    const paddingY = 24;
    const rows = Math.max(orders.length, 1);
    const maxNodesPerRow = Math.max(
      ...orders.map(ord => (layerGroups.get(ord)?.length || 0)),
      1
    );
    const rowGap = 100; // 每层垂直间距
    const colGap = 80;  // 同层节点水平间距
    const height = paddingY * 2 + (rows - 1) * rowGap;
    const width = paddingX * 2 + (maxNodesPerRow - 1) * colGap;
    const verticalStep = rows > 1 ? rowGap : 0;

    // 计算每层的水平分布
    const pos = new Map<string, { x: number; y: number }>();
    orders.forEach((ord, rowIndex) => {
      const ids = layerGroups.get(ord)!.map(n => n.id).sort();
      const count = Math.max(ids.length, 1);
      const horizontalStep = count > 1 ? (width - 2 * paddingX) / (count - 1) : 0;
      const y = paddingY + rowIndex * verticalStep; // 顶部为 logits
      ids.forEach((id, i) => {
        const x = paddingX + i * horizontalStep;
        pos.set(id, { x, y });
      });
    });

    const miniNodes = displayData.nodes
      .filter(n => true)
      .map(n => ({ id: n.id, x: pos.get(n.id)?.x ?? paddingX, y: pos.get(n.id)?.y ?? (height - paddingY), n }));
    const miniEdges = displayData.edges
      .filter(e => pos.has(e.source) && pos.has(e.target))
      .map(e => ({ 
        x1: pos.get(e.source)!.x, 
        y1: pos.get(e.source)!.y, 
        x2: pos.get(e.target)!.x, 
        y2: pos.get(e.target)!.y,
        source: e.source,
        target: e.target,
        attribution: e.attribution
      }));

    return { miniNodes, miniEdges, miniWidth: width, miniHeight: height };
  }, [circuitData, showInfluenceOnly, selectedNode]);

  const scaledHeight = miniHeight * miniScale;

  // 自动滚动到悬停的节点
  useEffect(() => {
    if (!hoveredNodeId || !nodeListRef.current) return;
    
    const nodeElement = nodeListRef.current.querySelector(`[data-node-id="${hoveredNodeId}"]`) as HTMLElement;
    if (nodeElement) {
      nodeElement.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center',
        inline: 'nearest'
      });
    }
  }, [hoveredNodeId]);

  // 当选中节点变化时，获取样本
  useEffect(() => {
    if (selectedNode && isFeatureNode(selectedNode)) {
      fetchNodeSamples(selectedNode);
    } else {
      setNodeSamples([]);
    }
  }, [selectedNode, fetchNodeSamples]);

  // 状态恢复逻辑 - 当circuitData加载完成后恢复状态
  useEffect(() => {
    if (!circuitData) return;
    
    const urlParams = new URLSearchParams(window.location.search);
    const returnStateParam = urlParams.get('returnState');
    
    if (returnStateParam) {
      try {
        const returnState = JSON.parse(decodeURIComponent(returnStateParam));
        console.log('🔄 检测到返回状态，恢复Circuit页面状态:', returnState);
        
        // 恢复缩放和宽度设置
        if (returnState.miniScale !== undefined) {
          setMiniScale(returnState.miniScale);
        }
        if (returnState.fitWidth !== undefined) {
          setFitWidth(returnState.fitWidth);
        }
        
        // 恢复显示影响状态
        if (returnState.showInfluenceOnly !== undefined) {
          setShowInfluenceOnly(returnState.showInfluenceOnly);
          console.log('✅ 恢复显示影响状态:', returnState.showInfluenceOnly);
        }
        
        // 恢复选中的节点
        if (returnState.selectedNodeId) {
          const node = circuitData.nodes.find(n => n.id === returnState.selectedNodeId);
          if (node) {
            setSelectedNode(node);
            setSelectedNodeId(node.id);
            // 计算并设置高亮效果
            const highlights = calculateHighlights(node);
            setHighlightedEdges(highlights.edges);
            setHighlightedNodes(highlights.nodes);
            setHighlightedInEdges(highlights.inEdges);
            setHighlightedInNodes(highlights.inNodes);
            console.log('✅ 恢复选中节点:', node.id, '显示影响模式:', returnState.showInfluenceOnly);
            
            // 如果显示影响模式开启，确保子图正确显示
            if (returnState.showInfluenceOnly) {
              console.log('🔄 显示影响模式已开启，子图应该显示');
            }
          }
        }
        
        // 恢复悬停的节点
        if (returnState.hoveredNodeId) {
          setHoveredNodeId(returnState.hoveredNodeId);
          console.log('✅ 恢复悬停节点:', returnState.hoveredNodeId);
        }
        
        // 恢复滚动位置（等待布局渲染）
        setTimeout(() => {
          const c = containerRef.current;
          if (c && (returnState.containerScrollLeft !== undefined)) {
            c.scrollLeft = returnState.containerScrollLeft;
          }
          if (c && (returnState.containerScrollTop !== undefined)) {
            c.scrollTop = returnState.containerScrollTop;
          }
          const nl = nodeListRef.current;
          if (nl && (returnState.nodeListScrollTop !== undefined)) {
            nl.scrollTop = returnState.nodeListScrollTop;
          }
          console.log('✅ 恢复滚动位置');
        }, 50);
        
        console.log('✅ Circuit页面状态恢复完成');
        
        // 清除URL中的返回状态参数
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.delete('returnState');
        window.history.replaceState({}, '', newUrl.toString());
        
      } catch (error) {
        console.error('恢复状态失败:', error);
      }
    }
  }, [circuitData, calculateHighlights]);

  // 清理超时
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  // 简化的棋盘组件
  const SimpleChessBoard = ({ fen, activation }: { fen: string; activation: number }) => {
    const [board, setBoard] = useState<string[][]>([]);
    
    useEffect(() => {
      // 解析FEN字符串
      const parts = fen.split(' ');
      const boardPart = parts[0];
      const rows = boardPart.split('/');
      
      const newBoard: string[][] = [];
      for (const row of rows) {
        const boardRow: string[] = [];
        for (const char of row) {
          if (/\d/.test(char)) {
            // 数字表示空位
            for (let i = 0; i < parseInt(char); i++) {
              boardRow.push('');
            }
          } else {
            // 棋子
            boardRow.push(char);
          }
        }
        newBoard.push(boardRow);
      }
      setBoard(newBoard);
    }, [fen]);

    const getPieceSymbol = (piece: string) => {
      const symbols: { [key: string]: string } = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
      };
      return symbols[piece] || piece;
    };

    return (
      <div className="border rounded-lg p-4 bg-white">
        <div className="text-sm font-medium mb-2">激活值: {activation !== null ? activation.toFixed(3) : '0.000'}</div>
        <div className="grid grid-cols-8 gap-0 border-2 border-gray-800">
          {board.map((row, rowIndex) =>
            row.map((piece, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`w-8 h-8 flex items-center justify-center text-lg ${
                  (rowIndex + colIndex) % 2 === 0 ? 'bg-gray-200' : 'bg-gray-400'
                }`}
              >
                {piece && getPieceSymbol(piece)}
              </div>
            ))
          )}
        </div>
        <div className="text-xs text-gray-500 mt-2 truncate">{fen}</div>
      </div>
    );
  };

  if (error) {
    return (
      <div className="pt-4 pb-20 px-6">
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-red-700">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span className="font-medium">加载失败:</span> {error}
            </div>
            <Button 
              onClick={() => window.location.reload()}
              className="mt-4"
              variant="outline"
            >
              重试
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="pt-4 pb-20 px-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-center gap-3 text-blue-600">
              <Loader2 className="h-6 w-6 animate-spin" />
              <span>正在加载电路数据...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!circuitData || circuitData.nodes.length === 0) {
    return (
      <div className="pt-4 pb-20 px-6">
        <Card className="text-center py-12">
          <CardContent>
            <FileJson className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">没有可展示的电路数据</h3>
            <p className="text-gray-500">当前文件不包含有效的电路信息</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="pt-4 pb-20 px-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
            <Network className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Circuits Viewer
            </h1>
            <p className="text-gray-600 mt-1">可视化神经网络电路和特征归因</p>
          </div>
        </div>

        {/* 日志面板 */}
        <Card className="mb-4">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">后端日志</CardTitle>
              <div className="flex items-center gap-2">
                <Button size="sm" variant={logStreaming ? "default" : "outline"} onClick={() => (logStreaming ? stopLogStream() : startLogStream())}>
                  {logStreaming ? "停止" : "开始"}
                </Button>
                <Button size="sm" variant="outline" onClick={async () => {
                  try {
                    const backendBase = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.replace(/\/$/, "") || "";
                    const r = await fetch(`${backendBase}/logs/recent?limit=200`);
                    if (r.ok) {
                      const j = await r.json();
                      setLogs((j.logs || []).map((x: any) => ({ ts: x.ts, level: x.level, msg: x.msg })));
                    }
                  } catch {}
                }}>刷新</Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-40 overflow-auto text-xs font-mono bg-gray-50 p-2 rounded">
              {logs.length === 0 ? (
                <div className="text-gray-500">无日志</div>
              ) : (
                logs.map((l, idx) => (
                  <div key={idx} className="whitespace-pre-wrap">
                    [{new Date(l.ts * 1000).toLocaleTimeString()}] {l.level}: {l.msg}
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Prompt Board Display */}
      {circuitData?.metadata?.prompt_tokens && circuitData.metadata.prompt_tokens.length > 0 && (
        <Card className="mb-6 shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              当前局面
            </CardTitle>
            <CardDescription>
              显示当前电路分析对应的棋盘局面
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-700 mb-2">FEN字符串:</div>
                <div className="text-sm text-gray-600 font-mono bg-gray-50 p-2 rounded">
                  {circuitData.metadata.prompt_tokens.join(' ')}
                </div>
              </div>
              <div className="flex-shrink-0">
                <SimpleChessBoard 
                  fen={circuitData.metadata.prompt_tokens.join(' ')} 
                  activation={1.0}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics */}
      {stats && (
        <Card className="mb-6 shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">电路统计</CardTitle>
              <div className="flex items-center gap-3">
                {/* 远端文件选择 */}
                <div className="flex items-center gap-2">
                  <Select
                    value={currentFile || undefined}
                    onValueChange={async (value) => {
                      setCurrentFile(value);
                      await loadCircuitFromPath(value);
                    }}
                  >
                    <SelectTrigger className="w-[260px] bg-white">
                      <SelectValue placeholder={scanning ? "扫描中..." : availableFiles.length ? "选择电路文件" : "未发现文件"} />
                    </SelectTrigger>
                    <SelectContent>
                      {availableFiles.map((p) => {
                        const name = p.split("/").pop();
                        return (
                          <SelectItem key={p} value={p}>
                            {name}
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={scanning}
                    onClick={async () => {
                      const files = await scanCircuits();
                      setAvailableFiles(files);
                      if (files.length && (!currentFile || !files.includes(currentFile))) {
                        setCurrentFile(files[0]);
                        await loadCircuitFromPath(files[0]);
                      }
                    }}
                  >
                    {scanning ? "扫描中" : "重新扫描"}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      localStorage.removeItem('lastSelectedCircuitFile');
                      localStorage.removeItem('circuitsShowInfluenceOnly');
                      localStorage.removeItem('circuitsSelectedNodeId');
                      setRememberedFile(null);
                      setShowInfluenceOnly(false);
                      setSelectedNode(null);
                      setSelectedNodeId(null);
                      console.log('🗑️ 清除本地存储的电路文件选择、显示影响状态和选中节点');
                      // 重新加载默认文件
                      const defaultFile = availableFiles.find((f) => f.includes("win_or_go_home_k_group_1024")) || availableFiles[0];
                      if (defaultFile) {
                        setCurrentFile(defaultFile);
                        loadCircuitFromPath(defaultFile);
                      }
                    }}
                    title="清除记住的文件选择，恢复默认"
                  >
                    重置默认
                  </Button>
                </div>
                {/* 本地文件选择 */}
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={handleFileChange}
                  className="text-sm"
                />
                {selectedFileName && (
                  <span className="text-xs text-gray-500">已选择: {selectedFileName}</span>
                )}
                {rememberedFile && (
                  <span className="text-xs text-blue-500">
                    记住: {rememberedFile.startsWith('local_file_') 
                      ? '本地文件' 
                      : rememberedFile.split('/').pop()}
                  </span>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {/* 新增：FEN/Move 归因表单 */}
            <div className="mb-4 p-4 border rounded-lg bg-gray-50">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
                <div className="md:col-span-2">
                  <div className="text-sm mb-1">FEN</div>
                  <input
                    className="w-full border rounded px-3 py-2 text-sm"
                    placeholder="输入FEN字符串"
                    value={fenInput}
                    onChange={(e) => setFenInput(e.target.value)}
                  />
                </div>
                <div>
                  <div className="text-sm mb-1">Move (UCI，可选)</div>
                  <input
                    className="w-full border rounded px-3 py-2 text-sm"
                    placeholder="如 e2e4，不填则自动推理"
                    value={moveInput}
                    onChange={(e) => setMoveInput(e.target.value)}
                  />
                </div>
                <div>
                  <div className="text-sm mb-1">Side</div>
                  <Select value={sideInput} onValueChange={setSideInput}>
                    <SelectTrigger className="w-full bg-white">
                      <SelectValue placeholder="选择侧" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="k">k</SelectItem>
                      <SelectItem value="q">q</SelectItem>
                      <SelectItem value="both">both</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="mt-3">
                <Button onClick={handleGenerateFromFEN} disabled={generating || !fenInput.trim()}>
                  {generating ? "生成中..." : "生成电路图"}
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{stats.totalNodes}</div>
                <div className="text-sm text-gray-600">总节点数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{stats.totalEdges}</div>
                <div className="text-sm text-gray-600">总边数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{stats.uniqueSAEs}</div>
                <div className="text-sm text-gray-600">SAE模型数</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {Object.keys(stats.nodeTypes).length}
                </div>
                <div className="text-sm text-gray-600">节点类型数</div>
              </div>
            </div>
            
            <div className="mt-4">
              <div className="text-sm font-medium text-gray-700 mb-2">节点类型分布:</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(stats.nodeTypes).map(([type, count]) => (
                  <Badge key={type} variant="outline" className="bg-blue-50">
                    {type}: {count}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Circuit Visualization */}
      <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Network className="h-5 w-5 text-blue-500" />
                电路可视化
              </CardTitle>
              <CardDescription>
                显示 {showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).nodes.length : circuitData.nodes.length} 个节点和 {showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).edges.length : circuitData.edges.length} 条边的电路关系
                {showInfluenceOnly && selectedNode && (
                  <span className="text-orange-600 font-medium"> (完整影响子图模式)</span>
                )}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={showInfluenceOnly ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  if (showInfluenceOnly) {
                    setShowInfluenceOnly(false);
                    localStorage.setItem('circuitsShowInfluenceOnly', 'false');
                  } else if (selectedNode) {
                    setShowInfluenceOnly(true);
                    localStorage.setItem('circuitsShowInfluenceOnly', 'true');
                  }
                }}
                disabled={!selectedNode && !showInfluenceOnly}
                title={!selectedNode ? "请先选择一个节点" : showInfluenceOnly ? "显示完整电路" : "显示影响选中节点的完整子图"}
              >
                {showInfluenceOnly ? "显示完整" : "显示影响"}
              </Button>
              {showInfluenceOnly && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setShowInfluenceOnly(false);
                    localStorage.setItem('circuitsShowInfluenceOnly', 'false');
                  }}
                  title="退出完整影响子图模式"
                >
                  退出影响
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 节点列表 */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">节点列表</h3>
                  {selectedNode && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSelectedNode(null);
                        setSelectedNodeId(null);
                        setHighlightedEdges(new Set());
                        setHighlightedNodes(new Set());
                        setHighlightedInEdges(new Set());
                        setHighlightedInNodes(new Set());
                      }}
                    >
                      清除选中
                    </Button>
                  )}
                </div>
                <div ref={nodeListRef} className="space-y-2 h-96 overflow-y-auto">
                  {(showInfluenceOnly && selectedNode ? calculateSubgraph(selectedNode).nodes : circuitData.nodes).map((node) => (
                    <div
                      key={node.id}
                      data-node-id={node.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                        selectedNode?.id === node.id ? 'ring-2 ring-blue-500' : ''
                      } ${
                        hoveredNodeId === node.id ? 'ring-2 ring-orange-400 bg-orange-50' : ''
                      }`}
                      onClick={() => {
                        setSelectedNode(node);
                        setSelectedNodeId(node.id);
                        
                        // 计算并设置高亮效果
                        const highlights = calculateHighlights(node);
                        setHighlightedEdges(highlights.edges);
                        setHighlightedNodes(highlights.nodes);
                        setHighlightedInEdges(highlights.inEdges);
                        setHighlightedInNodes(highlights.inNodes);
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-4 h-4 rounded-full ${getNodeColor(node)}`}></div>
                        <div className="flex-1">
                          <div className="font-medium">{getNodeLabel(node)}</div>
                          <div className="text-sm text-gray-500">
                            {node.type} • ID: {node.id}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>


              </div>

                            {/* 选中节点详情 */}
              <div>
                <h3 className="text-lg font-semibold mb-4">节点详情</h3>
                <div className="h-[600px] overflow-y-auto">
                {(() => {
                  const displayNode = selectedNode || (hoveredNodeId ? circuitData.nodes.find(n => n.id === hoveredNodeId) : null);
                  if (!displayNode) {
                    return (
                      <div className="p-4 border rounded-lg bg-gray-50 text-gray-500">
                        请选择一个节点查看详情
                      </div>
                    );
                  }
                  return (
                    <div className="p-4 border rounded-lg bg-gray-50">
                      <div className="space-y-2">
                        <div><strong>ID:</strong> {displayNode.id}</div>
                        <div><strong>类型:</strong> {displayNode.type}</div>
                        {displayNode.sae && <div><strong>SAE:</strong> {displayNode.sae}</div>}
                        {displayNode.featureIndex !== undefined && <div><strong>特征索引:</strong> {displayNode.featureIndex}</div>}
                        {displayNode.position !== undefined && <div><strong>位置:</strong> {displayNode.position}</div>}
                        {displayNode.activation !== undefined && displayNode.activation !== null && <div><strong>激活值:</strong> {displayNode.activation.toFixed(3)}</div>}
                        {displayNode.maxActivation !== undefined && displayNode.maxActivation !== null && <div><strong>最大激活:</strong> {displayNode.maxActivation.toFixed(3)}</div>}
                        {displayNode.tokenId !== undefined && <div><strong>Token ID:</strong> {displayNode.tokenId}</div>}
                        {displayNode.layer !== undefined && <div><strong>层:</strong> {displayNode.layer}</div>}
                        {displayNode.head !== undefined && <div><strong>头:</strong> {displayNode.head}</div>}
                        {displayNode.query !== undefined && <div><strong>Query:</strong> {displayNode.query}</div>}
                        {displayNode.key !== undefined && <div><strong>Key:</strong> {displayNode.key}</div>}
                        {displayNode.pattern !== undefined && displayNode.pattern !== null && <div><strong>Pattern:</strong> {displayNode.pattern.toFixed(3)}</div>}
                      </div>

                      {/* 相关边 */}
                      <div className="mt-4">
                        <h4 className="font-medium mb-2">相关连接:</h4>
                        <div className="space-y-1">
                          {circuitData.edges
                            .filter(edge => edge.source === displayNode.id || edge.target === displayNode.id)
                            .map((edge, index) => (
                              <div key={index} className="text-sm text-gray-600">
                                {edge.source === displayNode.id ? '输出到' : '来自'} {edge.source === displayNode.id ? edge.target : edge.source} 
                                (权重: {edge.attribution !== null ? edge.attribution.toFixed(3) : '0.000'})
                              </div>
                            ))}
                        </div>
                      </div>

                      {/* 跳转到 Feature 页 */}
                      {isFeatureNode(displayNode) && displayNode.featureIndex !== undefined && (
                        <div className="mt-4">
                          <Button onClick={() => navigateToFeature(displayNode)} variant="default">
                            查看该 feature
                          </Button>
                        </div>
                      )}

                      {/* 激活样本棋盘 */}
                      {isFeatureNode(displayNode) && (
                        <div className="mt-6">
                          <h4 className="font-medium mb-3">Top 激活样本</h4>
                          {samplesState.loading ? (
                            <div className="flex items-center justify-center py-8">
                              <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
                              <span className="ml-2 text-gray-600">加载样本中...</span>
                            </div>
                          ) : nodeSamples.length > 0 ? (
                            <div className="grid grid-cols-2 gap-4">
                              {nodeSamples.slice(0, 4).map((sample, index) => {
                                // 从样本文本中提取FEN
                                const lines = sample.text?.split('\n') || [];
                                let fen = '';
                                for (const line of lines) {
                                  const trimmed = line.trim();
                                  const parts = trimmed.split(/\s+/);
                                  if (parts.length >= 6) {
                                    const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
                                    const boardRows = boardPart.split('/');
                                    if (boardRows.length === 8 && /^[rnbqkpRNBQKP1-8]+$/.test(boardPart) && /^[wb]$/.test(activeColor)) {
                                      fen = trimmed;
                                      break;
                                    }
                                  }
                                }
                                
                                if (!fen) return null;
                                
                                const maxActivation = sample.featureActs && sample.featureActs.length > 0 ? Math.max(...sample.featureActs) : 0;
                                return (
                                  <SimpleChessBoard
                                    key={index}
                                    fen={fen}
                                    activation={maxActivation}
                                  />
                                );
                              })}
                            </div>
                          ) : (
                            <div className="text-gray-500 text-sm py-4 text-center">
                              暂无激活样本
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })()}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Full-width Tree View */}
      {miniNodes.length > 0 && (
        <Card className="mt-6 shadow-xl border-0 bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>电路树状图（顶部 logits → 底部低层）</span>
              <span className="flex items-center gap-3">
                <label className="flex items-center gap-1 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    className="mr-1"
                    checked={fitWidth}
                    onChange={(e) => setFitWidth(e.target.checked)}
                  />
                  适应宽度
                </label>
                <Button variant="outline" size="icon" onClick={() => setMiniScale(s => Math.max(0.25, s - 0.1))}>-</Button>
                <span className="text-xs w-12 text-center">{Math.round(miniScale * 100)}%</span>
                <Button variant="outline" size="icon" onClick={() => setMiniScale(s => Math.min(5, s + 0.1))}>+</Button>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* 图例 */}
            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <div className="text-sm font-medium mb-2">高亮图例:</div>
              <div className="flex flex-wrap gap-4 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-yellow-400 border-2 border-orange-500"></div>
                  <span>选中节点</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-200 border-2 border-red-500"></div>
                  <span>下游影响节点</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-200 border-2 border-blue-500"></div>
                  <span>上游影响节点</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-4 bg-red-500"></div>
                  <span>下游影响边</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-4 bg-blue-500"></div>
                  <span>上游影响边</span>
                </div>
              </div>
            </div>
            
            <div ref={containerRef} className="relative overflow-x-auto overflow-y-visible">
              <div style={{ width: (fitWidth && containerWidth ? containerWidth : miniWidth * miniScale), margin: '0 auto' }}>
                <svg
                  ref={svgRef}
                  width={(fitWidth && containerWidth ? containerWidth : miniWidth * miniScale)}
                  height={scaledHeight}
                  viewBox={`0 0 ${miniWidth} ${miniHeight}`}>
                  <g transform={`scale(${miniScale})`}>
                    {miniEdges.map((l, i) => {
                      const edgeId = `${l.source}-${l.target}`;
                      const isHighlighted = highlightedEdges.has(edgeId);
                      const isInHighlighted = highlightedInEdges.has(edgeId);
                      return (
                        <line 
                          key={i} 
                          x1={l.x1} 
                          y1={l.y1} 
                          x2={l.x2} 
                          y2={l.y2} 
                          stroke={
                            isHighlighted ? "#ef4444" : // 下游影响 - 红色
                            isInHighlighted ? "#3b82f6" : // 上游影响 - 蓝色
                            "#cbd5e1" // 普通边 - 灰色
                          } 
                          strokeWidth={
                            isHighlighted || isInHighlighted ? 3 / miniScale : 1 / miniScale
                          }
                          opacity={
                            isHighlighted || isInHighlighted ? 0.8 : 0.6
                          }
                        />
                      );
                    })}
                    {miniNodes.map((m) => {
                      const isHighlighted = highlightedNodes.has(m.id);
                      const isInHighlighted = highlightedInNodes.has(m.id);
                      const isSelected = selectedNodeId === m.id; // 新增：检查是否是从节点列表选中的节点
                      return (
                      <g key={m.id} transform={`translate(${m.x}, ${m.y})`}>
                        <circle
                          r={5 / miniScale}
                          fill={
                            isSelected ? "#fbbf24" : // 选中的节点 - 黄色
                            isHighlighted ? "#fecaca" : // 下游目标节点 - 浅红色
                            isInHighlighted ? "#dbeafe" : // 上游源节点 - 浅蓝色
                            getNodeFill(m.n) // 普通节点 - 原色
                          }
                          stroke={
                            isSelected ? "#d97706" : // 选中的节点 - 橙色边框
                            isHighlighted ? "#ef4444" : // 下游目标节点 - 红色边框
                            isInHighlighted ? "#3b82f6" : // 上游源节点 - 蓝色边框
                            "none" // 普通节点 - 无边框
                          }
                          strokeWidth={
                            isSelected ? 3 / miniScale : // 选中的节点 - 更粗的边框
                            isHighlighted || isInHighlighted ? 2 / miniScale : 0
                          }
                          onClick={() => isFeatureNode(m.n) && navigateToFeature(m.n)}
                          onMouseEnter={(e) => {
                            const crect = containerRef.current?.getBoundingClientRect();
                            if (!crect) return;
                            setHoverInfo({ x: e.clientX - crect.left + 8, y: e.clientY - crect.top + 8, node: m.n });
                            
                            // 清除之前的超时
                            if (hoverTimeoutRef.current) {
                              clearTimeout(hoverTimeoutRef.current);
                            }
                            
                            // 设置新的超时，延迟更新悬停状态
                            hoverTimeoutRef.current = setTimeout(() => {
                              // 计算并设置高亮
                              const highlights = calculateHighlights(m.n);
                              setHighlightedEdges(highlights.edges);
                              setHighlightedNodes(highlights.nodes);
                              setHighlightedInEdges(highlights.inEdges);
                              setHighlightedInNodes(highlights.inNodes);
                              setHoveredNodeId(m.id);
                              // 悬停时清除选中状态
                              setSelectedNodeId(null);
                            }, 100); // 100ms 延迟
                          }}
                          onMouseLeave={() => {
                            // 清除超时
                            if (hoverTimeoutRef.current) {
                              clearTimeout(hoverTimeoutRef.current);
                              hoverTimeoutRef.current = null;
                            }
                            
                            setHoverInfo(null);
                            setHighlightedEdges(new Set());
                            setHighlightedNodes(new Set());
                            setHighlightedInEdges(new Set());
                            setHighlightedInNodes(new Set());
                            setHoveredNodeId(null);
                            // 鼠标离开时恢复选中状态（如果有的话）
                            if (selectedNode) {
                              setSelectedNodeId(selectedNode.id);
                              const highlights = calculateHighlights(selectedNode);
                              setHighlightedEdges(highlights.edges);
                              setHighlightedNodes(highlights.nodes);
                              setHighlightedInEdges(highlights.inEdges);
                              setHighlightedInNodes(highlights.inNodes);
                            }
                          }}
                          onMouseMove={(e) => {
                            const crect = containerRef.current?.getBoundingClientRect();
                            if (!crect) return;
                            setHoverInfo({ x: e.clientX - crect.left + 8, y: e.clientY - crect.top + 8, node: m.n });
                          }}
                          style={{ cursor: isFeatureNode(m.n) ? 'pointer' as const : 'default' }}
                        />
                      </g>
                    );
                    })}
                  </g>
                </svg>
              </div>
              {hoverInfo && (
                <div
                  className="absolute text-xs bg-white border border-gray-300 rounded shadow p-2 pointer-events-none"
                  style={{ left: hoverInfo.x, top: hoverInfo.y }}
                >
                  <div><span className="text-gray-500">层</span>: <span className="font-medium">{hoverInfo.node.layer ?? '-'}</span></div>
                  <div><span className="text-gray-500">位置</span>: <span className="font-medium">{hoverInfo.node.position ?? '-'}</span></div>
                  <div><span className="text-gray-500">特征</span>: <span className="font-medium">{hoverInfo.node.featureIndex ?? '-'}</span></div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}; 