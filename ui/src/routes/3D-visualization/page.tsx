import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { forceSimulation, forceManyBody, forceCenter, forceCollide } from 'd3-force';

// 图数据接口定义
interface GraphNode {
  node_id: string;
  feature?: number;
  layer?: number;
  ctx_idx?: number;
  feature_type?: string;
  label?: string;
  type?: string;
  meta?: any;
  weight?: number;
  influence?: number;
  activation?: number;
}

interface GraphLink {
  source: string;
  target: string;
  weight: number;
}

interface GraphData {
  metadata?: {
    slug?: string;
    prompt?: string;
  };
  nodes: GraphNode[];
  links: GraphLink[];
  label?: string;
}

interface ProcessedNode extends GraphNode {
  x: number;
  y: number;
  z: number;
  graphIds: number[];
  isShared: boolean;
  color: string;
}

interface ProcessedGraph {
  data: GraphData;
  color: string;
  visible: boolean;
  highlighted: boolean;
  id: number;
}

// 颜色配置
const GRAPH_COLORS = [
  '#ff6b6b',
  '#4ecdc4',
  '#45b7d1',
  '#96ceb4',
  '#ffeaa7',
];

const FEATURE_TYPE_COLORS = {
  lorsa: '#7a4cff',
  'cross layer transcoder': '#69b3a2',
  logit: '#ff6b6b',
  embedding: '#69b3a2',
  default: '#95a5a6',
  'mlp reconstruction error': '#95a5a6',
  'lorsa error': '#95a5a6',
  error: '#95a5a6'
};

export default function ThreeDVisualPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const animatingRef = useRef<boolean>(false);
  const lastRenderRef = useRef<number>(0);
  const lastRaycastRef = useRef<number>(0);
  
  const [graphs, setGraphs] = useState<ProcessedGraph[]>([]);
  const [processedNodes, setProcessedNodes] = useState<ProcessedNode[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 控制面板状态
  const [showUniqueNodes, setShowUniqueNodes] = useState(true);
  const [nodeSize, setNodeSize] = useState(1.0);
  const [edgeThickness, setEdgeThickness] = useState(1.0);
  const [performanceMode, setPerformanceMode] = useState(true);
  const [chessMode, setChessMode] = useState(false);
  const chessBoardGroupRef = useRef<THREE.Group | null>(null);
  const chessPiecesGroupRef = useRef<THREE.Group | null>(null);
  
  // 拖拽状态（右侧面板）
  const [panelPos, setPanelPos] = useState<{x: number; y: number}>({ x: 24, y: 24 });
  const [panelDragging, setPanelDragging] = useState(false);
  const dragOffsetRef = useRef<{x: number; y: number}>({ x: 0, y: 0 });

  // Three.js 对象引用
  const nodesGroupRef = useRef<THREE.Group | null>(null);
  const linksGroupRef = useRef<THREE.Group | null>(null);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);
  const mouseRef = useRef({ x: 0, y: 0 });



  // 初始化Three.js场景（本地依赖）
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, rect.width / rect.height, 0.1, 1000);
    camera.position.set(0, 0, 50);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: !performanceMode, powerPreference: 'high-performance', alpha: false, stencil: false, depth: true });
    renderer.setSize(rect.width, rect.height);
    renderer.setPixelRatio(performanceMode ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 1);
    renderer.shadowMap.enabled = false;
    rendererRef.current = renderer;

    // 星空背景（低分辨率）
    const starsGeometry = new THREE.BufferGeometry();
    const starsCount = performanceMode ? 500 : 2000;
    const positions = new Float32Array(starsCount * 3);
    for (let i = 0; i < starsCount * 3; i++) positions[i] = (Math.random() - 0.5) * 400;
    starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: performanceMode ? 0.6 : 0.5, transparent: true, opacity: 0.75, sizeAttenuation: true });
    const stars = new THREE.Points(starsGeometry, starsMaterial);
    stars.frustumCulled = true;
    scene.add(stars);

    // 控制器
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = performanceMode ? 0.06 : 0.05;
    controlsRef.current = controls;

    // 光照
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 10, 5);
    scene.add(dirLight);

    // 射线检测
    raycasterRef.current = new THREE.Raycaster();

    // 分组
    nodesGroupRef.current = new THREE.Group();
    linksGroupRef.current = new THREE.Group();
    chessBoardGroupRef.current = new THREE.Group();
    chessPiecesGroupRef.current = new THREE.Group();
    scene.add(nodesGroupRef.current);
    scene.add(linksGroupRef.current);
    scene.add(chessBoardGroupRef.current);
    scene.add(chessPiecesGroupRef.current);

    const onResize = () => {
      if (!rendererRef.current || !cameraRef.current || !containerRef.current) return;
      const r = containerRef.current.getBoundingClientRect();
      rendererRef.current.setSize(r.width, r.height);
      cameraRef.current.aspect = r.width / r.height;
      cameraRef.current.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    const animate = (t: number) => {
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
      animatingRef.current = true;
      requestAnimationFrame(animate);
      // 帧率限制（性能模式30fps，否则60fps）
      const now = performance.now();
      const minDelta = performanceMode ? (1000 / 30) : (1000 / 60);
      if (now - lastRenderRef.current < minDelta) return;
      lastRenderRef.current = now;
      controlsRef.current?.update();
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };
    if (!animatingRef.current) requestAnimationFrame(animate);

    const onMouseMove = (event: MouseEvent) => {
      if (!canvasRef.current || !cameraRef.current || !raycasterRef.current || !nodesGroupRef.current) return;
      // 射线节流
      const now = performance.now();
      const rayDelta = performanceMode ? 120 : 60;
      if (now - lastRaycastRef.current < rayDelta) return;
      lastRaycastRef.current = now;

      const rect = canvasRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycasterRef.current.setFromCamera(mouseRef.current as any, cameraRef.current);
      const intersects = raycasterRef.current.intersectObjects(nodesGroupRef.current.children, false);
      if (intersects.length > 0) {
        const nodeData = intersects[0].object.userData;
        showTooltip(event, nodeData);
      } else {
        hideTooltip();
      }
    };
    canvas.addEventListener('mousemove', onMouseMove);



    return () => {
      window.removeEventListener('resize', onResize);
      canvas.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
      animatingRef.current = false;
    };
  // 仅在首次和performanceMode变化时重建渲染器与背景
  }, [performanceMode]);

  // 构建棋盘与棋子
  const buildChessBoard = (ctxList: number[]) => {
    if (!chessBoardGroupRef.current || !chessPiecesGroupRef.current) return;
    chessBoardGroupRef.current.clear();
    chessPiecesGroupRef.current.clear();

    // 仅单图可用
    if (graphs.length !== 1) return;

    // 计算embedding层的实际y高度（与布局一致）
    let minLayer = Number.POSITIVE_INFINITY;
    let maxLayer = Number.NEGATIVE_INFINITY;
    processedNodes.forEach(n => {
      const parts = (n.node_id || '').split('_');
      const rawLayer = Number(parts[0]);
      const layerFromId = Number.isFinite(rawLayer) ? Math.floor(rawLayer / 2) : 0;
      if (layerFromId < minLayer) minLayer = layerFromId;
      if (layerFromId > maxLayer) maxLayer = layerFromId;
    });
    if (!Number.isFinite(minLayer) || !Number.isFinite(maxLayer)) return;
    const ySpacingLocal = 18;
    const layerToYLocal = (layer: number) => (layer - minLayer) * ySpacingLocal;
    const yCenterOffsetLocal = -((maxLayer - minLayer) * ySpacingLocal) / 2;
    const embeddingY = layerToYLocal(minLayer) + yCenterOffsetLocal;

    // 棋盘参数
    const gridSize = 8; // 8x8
    const cellSize = 5.0; // 放大格子边长
    const boardSize = gridSize * cellSize; // 棋盘总宽

    // 将ctx映射到8x8网格（按升序截断/填充）
    const sortedCtx = [...ctxList].sort((a,b)=>a-b);
    const maxCells = gridSize * gridSize;
    const mappedCtx = sortedCtx.slice(0, maxCells);

    // 棋盘底板
    const boardGeo = new THREE.PlaneGeometry(boardSize, boardSize);
    const boardMat = new THREE.MeshBasicMaterial({ color: 0x222222, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });
    const boardMesh = new THREE.Mesh(boardGeo, boardMat);
    boardMesh.rotation.x = -Math.PI / 2; // 水平放置
    boardMesh.position.set(0, embeddingY, 0);
    chessBoardGroupRef.current.add(boardMesh);

    // 画格子
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        const isDark = (r + c) % 2 === 1;
        const tileGeo = new THREE.PlaneGeometry(cellSize, cellSize);
        const tileMat = new THREE.MeshBasicMaterial({ color: isDark ? 0x3b3b3b : 0xcccccc, side: THREE.DoubleSide, transparent: true, opacity: 0.85, depthWrite: false });
        const tile = new THREE.Mesh(tileGeo, tileMat);
        tile.rotation.x = -Math.PI / 2;
        const x = (c - (gridSize - 1) / 2) * cellSize;
        const z = (r - (gridSize - 1) / 2) * cellSize;
        tile.position.set(x, embeddingY + 0.01, z);
        chessBoardGroupRef.current.add(tile);
      }
    }

    // 从FEN生成棋子
    try {
      const fen = graphs[0]?.data?.metadata?.prompt || '';
      const boardPart = String(fen).split(' ')[0] || '';
      const ranks = boardPart.split('/');
      if (ranks.length === 8) {
        for (let r = 0; r < 8; r++) {
          const rankStr = ranks[r];
          let file = 0;
          for (const ch of rankStr) {
            if (/^[1-8]$/.test(ch)) {
              file += Number(ch);
            } else {
              if (file >= 0 && file < 8) {
                const c = file;
                const tileX = (c - (gridSize - 1) / 2) * cellSize;
                const tileZ = (r - (gridSize - 1) / 2) * cellSize;
                const isWhite = ch === ch.toUpperCase();
                const color = isWhite ? 0xffffff : 0x333333;
                const height = 3.2; // 增加高度
                const radialSegments = performanceMode ? 10 : 18;
                const geo = new THREE.CylinderGeometry(1.3, 1.3, height, radialSegments);
                const mat = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 1.0, emissive: 0x111111, emissiveIntensity: 0.2 });
                const piece = new THREE.Mesh(geo, mat);
                piece.position.set(tileX, embeddingY + height / 2, tileZ);
                chessPiecesGroupRef.current!.add(piece);
              }
              file += 1;
            }
          }
        }
      }
    } catch {}

    // 参考柱：将上方feature的同ctx列x位置做参考（保持与棋盘r/c一致）
    const allCtxSet = new Set<number>(ctxList);
    const allCtxArr = Array.from(allCtxSet.values()).sort((a,b)=>a-b);
    const ctxToX = new Map<number, number>();
    const xGroupSpacing = 10;
    allCtxArr.forEach((ctx, idx) => {
      ctxToX.set(ctx, (idx - (allCtxArr.length - 1)/2) * xGroupSpacing);
    });

    mappedCtx.forEach((ctx, i) => {
      const r = Math.floor(i / gridSize);
      const c = i % gridSize;
      const tileZ = (r - (gridSize - 1) / 2) * cellSize;
      const refX = ctxToX.get(ctx) ?? 0;
      const refMat = new THREE.MeshBasicMaterial({ color: 0xffcc00, transparent: true, opacity: 0.5 });
      const refGeo = new THREE.CylinderGeometry(0.05, 0.05, 40, 6);
      const ref = new THREE.Mesh(refGeo, refMat);
      ref.position.set(refX, embeddingY + 20, tileZ);
      chessPiecesGroupRef.current!.add(ref);
    });
  };

  // 在数据或模式变化时重建棋盘
  useEffect(() => {
    if (!chessMode) {
      chessBoardGroupRef.current?.clear();
      chessPiecesGroupRef.current?.clear();
      return;
    }
    if (graphs.length !== 1) return; // 仅单图
    // 收集embedding层的ctx（按node_id解析layer/ctx，layer最小的一层视为embedding层）
    const embCtxSet = new Set<number>();
    processedNodes.forEach(n => {
      const parts = (n.node_id || '').split('_');
      const rawLayer = Number(parts[0]);
      const layerFromId = Number.isFinite(rawLayer) ? Math.floor(rawLayer / 2) : 0;
      if (layerFromId === 0) { // 视为embedding层
        const ctxFromId = Number(parts[2]);
        const ctx = (n as any).ctx_idx ?? (Number.isFinite(ctxFromId) ? ctxFromId : 0);
        embCtxSet.add(ctx);
      }
    });
    buildChessBoard(Array.from(embCtxSet.values()));
  }, [chessMode, graphs, processedNodes, performanceMode]);

  const showTooltip = (event: MouseEvent, nodeData: any) => {
    if (!tooltipRef.current) return;
    const tooltip = tooltipRef.current;
    tooltip.style.display = 'block';
    tooltip.style.left = `${event.clientX + 10}px`;
    tooltip.style.top = `${event.clientY + 10}px`;
    const graphNames = nodeData.graphIds?.map((id: number) => graphs[id]?.data.label || graphs[id]?.data.metadata?.slug || `Graph ${id + 1}`).join(', ') || '';
    tooltip.innerHTML = `
      <div class="bg-gray-900 text-white p-2 rounded shadow-lg text-sm">
        <div><strong>ID:</strong> ${nodeData.node_id}</div>
        ${nodeData.label ? `<div><strong>Label:</strong> ${nodeData.label}</div>` : ''}
        ${nodeData.feature_type ? `<div><strong>Type:</strong> ${nodeData.feature_type}</div>` : ''}
        <div><strong>Graphs:</strong> ${graphNames}</div>
      </div>
    `;
  };

  const hideTooltip = () => {
    if (tooltipRef.current) tooltipRef.current.style.display = 'none';
  };

  // 文件上传处理
  const handleFileUpload = async (files: FileList) => {
    if (files.length < 1 || files.length > 5) {
      setError('Please upload 1-5 JSON files');
      return;
    }

    const filesArr = Array.from(files);
    console.log('📦 本次将解析文件数量:', filesArr.length, filesArr.map(f=>f.name));

    setIsLoading(true);
    setError(null);

    try {
      const loadedGraphs: ProcessedGraph[] = [];
      const failures: { name: string; reason: string }[] = [];

      for (let i = 0; i < filesArr.length; i++) {
        const file = filesArr[i];
        console.log(`⏳ 准备读取文件[${i}/${filesArr.length}]: ${file.name}, size=${file.size}`);
        try {
          let text: string | null = null;
          // 路径1：直接使用 File.text()
          try {
            text = await file.text();
            console.log(`📄 text()成功[${i}]: bytes=${text.length}`);
          } catch (e1: any) {
            console.warn(`⚠️ text()失败[${i}]: ${file.name}, 尝试FileReader`, e1?.message || e1);
            // 路径2：使用 FileReader 读取
            text = await new Promise<string>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(String(reader.result || ''));
              reader.onerror = () => reject(reader.error || new Error('FileReader error'));
              reader.readAsText(file);
            });
            console.log(`📄 FileReader成功[${i}]: bytes=${text.length}`);
          }

          const data: GraphData = JSON.parse(text!);
          const nodeCount = Array.isArray((data as any).nodes) ? (data as any).nodes.length : 0;
          const linkCount = Array.isArray((data as any).links) ? (data as any).links.length : 0;
          const label = data.label || data.metadata?.slug || file.name.replace(/\.json$/,'');
          console.log(`✅ 解析成功[${i}]: label=${label}, nodes=${nodeCount}, links=${linkCount}`);

          loadedGraphs.push({
            data,
            color: GRAPH_COLORS[i % GRAPH_COLORS.length],
            visible: true,
            highlighted: false,
            id: i
          });
          console.log(`📌 已加入loadedGraphs: index=${i}, 当前总数=${loadedGraphs.length}`);
        } catch (e: any) {
          console.error(`❌ 解析失败[${i}]: ${file.name}`, e?.message || e);
          failures.push({ name: file.name, reason: e?.message || 'unknown' });
        }
      }

      console.log('✅ 已加载图:', loadedGraphs.map(g => g.data.label || g.data.metadata?.slug || `Graph ${g.id + 1}`));
      console.log('📊 图数量:', loadedGraphs.length, '，失败数量:', failures.length);
      if (failures.length > 0) {
        console.warn('⚠️ 失败的文件:', failures);
      }

      if (loadedGraphs.length === 0) {
        setError('No valid JSON graphs loaded');
        setIsLoading(false);
        return;
      }

      setGraphs(loadedGraphs);
      console.log('🚀 调用processGraphs, 传入图数量=', loadedGraphs.length);
      processGraphs(loadedGraphs);
    } catch (err) {
      setError('Failed to load graph files: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setIsLoading(false);
    }
  };

  // 处理图数据
  const processGraphs = (loadedGraphs: ProcessedGraph[]) => {
    console.log('🔧 开始处理图数据，图数量=', loadedGraphs.length);
    loadedGraphs.forEach((g, idx) => {
      const nodeCount = Array.isArray((g.data as any).nodes) ? (g.data as any).nodes.length : 0;
      const linkCount = Array.isArray((g.data as any).links) ? (g.data as any).links.length : 0;
      console.log(`  • 图[${idx}] ${g.data.label || g.data.metadata?.slug || idx}: nodes=${nodeCount}, links=${linkCount}`);
    });

    const nodeMap = new Map<string, ProcessedNode>();
    loadedGraphs.forEach((graph, graphIndex) => {
      graph.data.nodes?.forEach(node => {
        const nodeId = node.node_id;
        if (!nodeId) return;

        // 对于logit节点：不同graph视为不同点（不合并）
        const isLogit = (node as any).is_target_logit === true 
          || (node.type && /logit/i.test(node.type))
          || (node.feature_type && /logit/i.test(node.feature_type));
        const mapKey = isLogit ? `${nodeId}::g${graphIndex}` : nodeId;

        if (nodeMap.has(mapKey)) {
          const existingNode = nodeMap.get(mapKey)!;
          if (!existingNode.graphIds.includes(graphIndex)) existingNode.graphIds.push(graphIndex);
          existingNode.isShared = existingNode.graphIds.length > 1;
        } else {
          const color = node.feature_type ? (FEATURE_TYPE_COLORS as any)[node.feature_type] || FEATURE_TYPE_COLORS.default : graph.color;
          nodeMap.set(mapKey, {
            ...node,
            node_id: mapKey, // 保证后续渲染与链接查找一致
            x: 0,
            y: 0,
            z: 0,
            graphIds: [graphIndex],
            isShared: false,
            color
          });
        }
      });
    });

    const totalNodes = nodeMap.size;
    const sharedCount = Array.from(nodeMap.values()).filter(n => n.isShared).length;
    const uniqueCount = totalNodes - sharedCount;
    console.log(`📈 聚合节点: total=${totalNodes}, shared=${sharedCount}, unique=${uniqueCount}`);

    layoutNodes(Array.from(nodeMap.values()), loadedGraphs);
    setProcessedNodes(Array.from(nodeMap.values()));
  };

  // 节点布局（共享节点2D，独特节点多平面）
  const layoutNodes = (nodes: ProcessedNode[], loadedGraphs: ProcessedGraph[]) => {
    // 解析基础信息
    const parsedNodes = nodes.map(n => {
      const parts = (n.node_id || '').split('_');
      const rawLayer = Number(parts[0]);
      const layerFromId = Number.isFinite(rawLayer) ? Math.floor(rawLayer / 2) : undefined;
      const ctxFromId = Number(parts[2]);
      const layer = n.layer ?? layerFromId ?? 0;
      const ctx = (n as any).ctx_idx ?? (Number.isFinite(ctxFromId) ? ctxFromId : 0);
      return { n, layer, ctx };
    });

    // 统计所有层与所有ctx
    const layerSet = new Set<number>();
    const ctxSet = new Set<number>();
    parsedNodes.forEach(p => { layerSet.add(p.layer); ctxSet.add(p.ctx); });
    const layers = Array.from(layerSet.values()).sort((a,b)=>a-b);
    if (layers.length === 0) return;
    const ctxList = Array.from(ctxSet.values()).sort((a,b)=>a-b);

    // 垂直方向：logit在最上，embedding在最下 => y随层号增大而上移（反转）
    const minLayer = layers[0];
    const maxLayer = layers[layers.length - 1];
    const ySpacing = 18;
    const layerToY = (layer: number) => (layer - minLayer) * ySpacing; 
    const yCenterOffset = -((maxLayer - minLayer) * ySpacing) / 2;

    // 水平方向：按全局ctx列对齐
    const xGroupSpacing = 10;
    const xInGroupSpacing = 2.2;
    const ctxToX = new Map<number, number>();
    ctxList.forEach((ctx, idx) => {
      const x = (idx - (ctxList.length - 1)/2) * xGroupSpacing;
      ctxToX.set(ctx, x);
    });

    // 如果是Chess模式（单图），准备ctx->棋盘tile坐标映射
    let ctxToTileXZ: Map<number, {x: number; z: number}> | null = null;
    if (chessMode && graphs.length === 1) {
      const gridSize = 8;
      const cellSize = 5.0; // 与棋盘一致
      const sortedCtx = [...ctxList].sort((a,b)=>a-b).slice(0, gridSize * gridSize);
      ctxToTileXZ = new Map<number, {x: number; z: number}>();
      sortedCtx.forEach((ctx, i) => {
        const r = Math.floor(i / gridSize);
        const c = i % gridSize;
        const tileX = (c - (gridSize - 1) / 2) * cellSize;
        const tileZ = (r - (gridSize - 1) / 2) * cellSize;
        ctxToTileXZ!.set(ctx, { x: tileX, z: tileZ });
      });
    }

    // 先按层与ctx聚合，保证相同ctx的X一致
    const mapKey = (layer: number, ctx: number) => `${layer}__${ctx}`;
    const bucket = new Map<string, ProcessedNode[]>();
    parsedNodes.forEach(({ n, layer, ctx }) => {
      const key = mapKey(layer, ctx);
      if (!bucket.has(key)) bucket.set(key, []);
      bucket.get(key)!.push(n);
    });

    // 基础平面位置（未应用多平面变换）
    const basePos = new Map<string, { x: number; y: number }[]>();
    const orderScore = (ft?: string) => {
      const t = (ft||'').toLowerCase();
      if (t === 'lorsa') return 0;
      if (t === 'cross layer transcoder') return 1;
      return 2;
    };

    bucket.forEach((group, key) => {
      const [layerStr, ctxStr] = key.split('__');
      const layer = Number(layerStr);
      const ctx = Number(ctxStr);
      const baseY = layerToY(layer) + yCenterOffset;
      group.sort((a,b)=> orderScore(a.feature_type) - orderScore(b.feature_type));
      const arr: { x: number; y: number }[] = [];
      group.forEach((node, idx) => {
        let x: number;
        let z: number | undefined;
        if (chessMode && graphs.length === 1 && ctxToTileXZ && ctxToTileXZ.has(ctx)) {
          const tile = ctxToTileXZ.get(ctx)!;
          x = tile.x;
          z = tile.z;
        } else {
          x = (ctxToX.get(ctx) || 0) + (idx - (group.length - 1)/2) * xInGroupSpacing;
        }
        const y = baseY;
        arr.push({ x, y });
        // 直接写入node的z（在后续统一写入时使用映射）
        if (z !== undefined) (node as any).__tileZ = z;
      });
      basePos.set(key, arr);
    });

    // 为不同子集分配平面（非chess模式仅用y分层；chess模式x/z来自tile，z再按子集扇区叠加）
    const nGraphs = Math.max(1, loadedGraphs.length);
    const baseAngles: number[] = Array.from({ length: nGraphs }, (_, i) => (2 * Math.PI * i) / nGraphs);
    const depthUnit = 6;

    const angleForSubset = (ids: number[]): number | null => {
      if (ids.length === nGraphs) return null;
      if (ids.length === 1) return baseAngles[ids[0] % nGraphs];
      if (ids.length === 2) {
        const a = baseAngles[ids[0] % nGraphs];
        const b = baseAngles[ids[1] % nGraphs];
        let mid = (a + b) / 2;
        if (Math.abs(a - b) > Math.PI) mid = ((a + b + 2 * Math.PI) / 2) % (2 * Math.PI);
        return mid;
      }
      const avg = ids.reduce((acc, g) => acc + baseAngles[g % nGraphs], 0) / ids.length;
      return avg;
    };

    const zOffsetForSubset = (ids: number[]): number => {
      if (ids.length === nGraphs) return 0;
      if (ids.length === 1) return depthUnit;
      if (ids.length === 2) return depthUnit * 2;
      return depthUnit;
    };

    bucket.forEach((group, key) => {
      const baseArr = basePos.get(key)!;
      const [layerStr, ctxStr] = key.split('__');
      const ctx = Number(ctxStr);
      group.forEach((node, idx) => {
        const bx = baseArr[idx].x;
        const by = baseArr[idx].y;
        let xFinal = bx;
        let zBaseFromTile: number | null = null;
        if (chessMode && graphs.length === 1 && (node as any).__tileZ !== undefined) {
          zBaseFromTile = (node as any).__tileZ as number;
        }
        const subset = [...node.graphIds].sort((a,b)=>a-b);
        const phi = angleForSubset(subset);
        const zOffset = zOffsetForSubset(subset);

        if (phi === null) {
          // 全共享：中央扇区
          node.x = xFinal;
          node.y = by;
          node.z = (zBaseFromTile !== null) ? zBaseFromTile : 0;
        } else {
          // 绕Y轴旋转 + 偏移（仅当非棋盘时才旋转x；棋盘模式保持tile x，不再旋转x）
          if (zBaseFromTile !== null) {
            node.x = xFinal;
            node.y = by;
            node.z = zBaseFromTile + zOffset;
          } else {
            const xRot = xFinal * Math.cos(phi);
            const zRot = -xFinal * Math.sin(phi);
            node.x = xRot;
            node.y = by;
            node.z = zRot + zOffset;
          }
        }
        delete (node as any).__tileZ;
      });
    });

    // 完成后渲染
    renderNodes();
    renderLinks(loadedGraphs);
  };

  const layoutUniqueNodes = (uniqueNodes: ProcessedNode[], graphCount: number) => {
    const layerSpacing = 20;
    uniqueNodes.forEach(node => {
      const graphId = node.graphIds[0];
      const z = (graphId - graphCount / 2) * layerSpacing;
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 30 + 10;
      node.x = Math.cos(angle) * radius;
      node.y = Math.sin(angle) * radius;
      node.z = z;
    });
  };

  // 渲染节点
  const renderNodes = () => {
    if (!nodesGroupRef.current || !sceneRef.current) return;
    nodesGroupRef.current.clear();
    processedNodes.forEach(node => {
      const graph = graphs[node.graphIds[0]];
      if (!graph?.visible && !node.isShared) return;
      if (!showUniqueNodes && !node.isShared) return;

      // 检测是否为logit节点：优先用标志，其次用type/feature_type关键词
      const isLogit = (node as any).is_target_logit === true 
        || (node.type && /logit/i.test(node.type))
        || (node.feature_type && /logit/i.test(node.feature_type));

      const widthSegments = performanceMode ? 8 : 16;
      const heightSegments = performanceMode ? 8 : 16;
      const geometry = new THREE.SphereGeometry(nodeSize, widthSegments, heightSegments);
      let material: THREE.Material;
      if (isLogit) {
        material = new THREE.MeshLambertMaterial({ color: 0xff4d4f, transparent: true, opacity: 0.95 });
      } else if (node.isShared) {
        material = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
      } else {
        material = new THREE.MeshLambertMaterial({ color: node.color, transparent: true, opacity: graph?.highlighted ? 1.0 : 0.7 });
      }
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(node.x, node.y, node.z);
      (mesh as any).userData = node;

      if (node.isShared) {
        const thetaSegments = performanceMode ? 8 : 16;
        const ringGeometry = new THREE.RingGeometry(nodeSize * 1.2, nodeSize * 1.5, thetaSegments);
        const ringColors = node.graphIds.map(id => graphs[id]?.color || '#ffffff');
        ringColors.forEach((color, index) => {
          const ringMaterial = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
          const ring = new THREE.Mesh(ringGeometry, ringMaterial);
          ring.position.copy(mesh.position);
          ring.rotation.x = Math.PI / 2;
          ring.rotation.z = (index / ringColors.length) * Math.PI * 2;
          nodesGroupRef.current!.add(ring);
        });
      }
      nodesGroupRef.current!.add(mesh);
    });
  };

  // 渲染连接
  const renderLinks = (loadedGraphs: ProcessedGraph[]) => {
    if (!linksGroupRef.current) return;
    linksGroupRef.current.clear();
    const nodeMap = new Map(processedNodes.map(node => [node.node_id, node]));
    loadedGraphs.forEach((graph) => {
      if (!graph.visible) return;
      graph.data.links?.forEach(link => {
        // 适配logit节点拆分：优先尝试graph特定key
        const srcKeyGraph = `${link.source}::g${graph.id}`;
        const tgtKeyGraph = `${link.target}::g${graph.id}`;
        const sourceNode = nodeMap.get(srcKeyGraph) || nodeMap.get(link.source);
        const targetNode = nodeMap.get(tgtKeyGraph) || nodeMap.get(link.target);
        if (!sourceNode || !targetNode) return;
        const geometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z),
          new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z)
        ]);
        const material = new THREE.LineBasicMaterial({ color: graph.color, transparent: true, opacity: graph.highlighted ? 0.8 : 0.4 });
        const line = new THREE.Line(geometry, material);
        linksGroupRef.current!.add(line);
      });
    });
  };

  // 重新渲染场景（当控制开关变化时）
  useEffect(() => {
    if (processedNodes.length > 0) {
      renderNodes();
      renderLinks(graphs);
    }
  }, [graphs, processedNodes, showUniqueNodes, nodeSize, edgeThickness]);

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => { setIsDragOver(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    const files = e.dataTransfer.files; if (files.length > 0) handleFileUpload(files);
  };
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) { handleFileUpload(e.target.files); e.currentTarget.value = ''; }
  };

  const rotateCamera = (axis: 'x' | 'y', direction: number) => {
    if (!cameraRef.current || !controlsRef.current) return;
    const angle = (15 * Math.PI) / 180 * direction;
    const camera = cameraRef.current;
    if (axis === 'x') {
      const newY = camera.position.y * Math.cos(angle) - camera.position.z * Math.sin(angle);
      const newZ = camera.position.y * Math.sin(angle) + camera.position.z * Math.cos(angle);
      camera.position.y = newY; camera.position.z = newZ;
    } else {
      const newX = camera.position.x * Math.cos(angle) + camera.position.z * Math.sin(angle);
      const newZ = -camera.position.x * Math.sin(angle) + camera.position.z * Math.cos(angle);
      camera.position.x = newX; camera.position.z = newZ;
    }
    camera.lookAt(0, 0, 0); controlsRef.current.update();
  };

  const resetView = () => { if (!cameraRef.current || !controlsRef.current) return; cameraRef.current.position.set(0, 0, 50); cameraRef.current.lookAt(0, 0, 0); controlsRef.current.reset(); };
  const toggleGraphVisibility = (graphId: number) => setGraphs(prev => prev.map(graph => graph.id === graphId ? { ...graph, visible: !graph.visible } : graph));
  const toggleGraphHighlight = (graphId: number) => setGraphs(prev => prev.map(graph => graph.id === graphId ? { ...graph, highlighted: !graph.highlighted } : graph));

  // 面板拖拽
  const onPanelMouseDown = (e: React.MouseEvent) => { setPanelDragging(true); dragOffsetRef.current = { x: e.clientX - panelPos.x, y: e.clientY - panelPos.y }; };
  const onPanelMouseMove = (e: React.MouseEvent) => { if (!panelDragging) return; setPanelPos({ x: e.clientX - dragOffsetRef.current.x, y: e.clientY - dragOffsetRef.current.y }); };
  const onPanelMouseUp = () => setPanelDragging(false);



  return (
    <div className="flex h-screen bg-gray-100" onMouseMove={onPanelMouseMove} onMouseUp={onPanelMouseUp}>
      <div ref={containerRef} className="flex-1 relative">
        {graphs.length === 0 ? (
          <div 
            className={`h-full flex items-center justify-center border-2 border-dashed transition-colors ${
              isDragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="text-center">
              <div className="mb-4">
                <svg className="w-16 h-16 text-gray-400 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-700 mb-2">Upload Graph Files</h3>
              <p className="text-gray-500 mb-4">Drag and drop 1-5 JSON files here, or click to browse</p>
              <input ref={fileInputRef} type="file" accept=".json" multiple onChange={handleFileInput} className="hidden" />
              <Button className="cursor-pointer" onClick={() => fileInputRef.current?.click()} aria-label="Choose Files">Choose Files</Button>
              {error && (<div className="mt-4 text-red-600 text-sm">{error}</div>)}
            </div>
          </div>
        ) : (
          <>
            <canvas ref={canvasRef} className="w-full h-full" />
            <div className="absolute top-4 left-4 flex flex-col space-y-2">
              <div className="flex space-x-2">
                <Button size="sm" onClick={() => rotateCamera('x', -1)}>↑</Button>
                <Button size="sm" onClick={() => rotateCamera('x', 1)}>↓</Button>
                <Button size="sm" onClick={() => rotateCamera('y', -1)}>←</Button>
                <Button size="sm" onClick={() => rotateCamera('y', 1)}>→</Button>
              </div>
              <Button size="sm" onClick={resetView}>Reset View</Button>
            </div>
            {isLoading && (
              <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="text-white text-lg">Loading graphs...</div>
              </div>
            )}
          </>
        )}
      </div>

      {/* 可拖拽的右侧控制面板 */}
      <div
        className="fixed w-80 bg-white border border-gray-200 shadow-lg rounded-md p-4"
        style={{ left: panelPos.x, top: panelPos.y, cursor: panelDragging ? 'grabbing' : 'grab', userSelect: 'none' }}
      >
        <div className="flex items-center justify-between mb-2" onMouseDown={onPanelMouseDown}>
          <h2 className="text-lg font-semibold">Graph Controls</h2>
          <span className="text-xs text-gray-500">Drag</span>
        </div>
        {graphs.length > 0 && (
          <>
            <Card className="mb-4">
              <CardHeader><CardTitle className="text-sm">Graphs</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                {graphs.map((graph) => (
                  <div key={graph.id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 rounded" style={{ backgroundColor: graph.color }} />
                      <span className="text-sm">{graph.data.label || graph.data.metadata?.slug || `Graph ${graph.id + 1}`}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input type="checkbox" checked={graph.visible} onChange={() => toggleGraphVisibility(graph.id)} className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded" />
                      <Button size="sm" variant={graph.highlighted ? 'default' : 'outline'} onClick={() => toggleGraphHighlight(graph.id)}>Highlight</Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="mb-4">
              <CardHeader><CardTitle className="text-sm">Display Options</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Show Unique Nodes</span>
                  <Switch checked={showUniqueNodes} onCheckedChange={setShowUniqueNodes} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Chess-board Mode (single graph)</span>
                  <Switch checked={chessMode} onCheckedChange={(v)=> graphs.length===1 ? setChessMode(v) : null} />
                </div>
                <div>
                  <label className="text-sm font-medium">Node Size: {nodeSize.toFixed(1)}</label>
                  <input type="range" min="0.5" max="3.0" step="0.1" value={nodeSize} onChange={(e) => setNodeSize(parseFloat(e.target.value))} className="w-full mt-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                </div>
                <div>
                  <label className="text-sm font-medium">Edge Thickness: {edgeThickness.toFixed(1)}</label>
                  <input type="range" min="0.5" max="3.0" step="0.1" value={edgeThickness} onChange={(e) => setEdgeThickness(parseFloat(e.target.value))} className="w-full mt-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Performance Mode</span>
                  <Switch checked={performanceMode} onCheckedChange={setPerformanceMode} />
                </div>
                {graphs.length !== 1 && chessMode && (
                  <div className="text-xs text-orange-600">仅在单图时可启用棋盘模式</div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle className="text-sm">Statistics</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="text-sm"><span className="font-medium">Total Nodes:</span> {processedNodes.length}</div>
                <div className="text-sm"><span className="font-medium">Shared Nodes:</span> {processedNodes.filter(n => n.isShared).length}</div>
                <div className="text-sm"><span className="font-medium">Unique Nodes:</span> {processedNodes.filter(n => !n.isShared).length}</div>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      <div ref={tooltipRef} className="absolute pointer-events-none z-50" style={{ display: 'none' }} />
    </div>
  );
}

export { ThreeDVisualPage };
