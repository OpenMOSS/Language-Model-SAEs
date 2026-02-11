import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ReactFlow,
  Node as ReactFlowNode,
  Edge as ReactFlowEdge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Loader2, Trash2, Edit2, Save, X } from 'lucide-react';
import { SaeComboLoader } from '@/components/common/SaeComboLoader';
import { ChessBoard } from '@/components/chess/chess-board';
import { GetFeatureFromFen } from '@/components/feature/get-feature-from-fen';
import {
  listCircuitAnnotations,
  getCircuitAnnotation,
  CircuitAnnotation,
  CircuitFeature,
  addEdgeToCircuit,
  removeEdgeFromCircuit,
  updateEdgeWeight,
  setFeatureLevel,
  createCircuitAnnotation,
  addFeatureToCircuit,
  removeFeatureFromCircuit,
} from '@/utils/api';

interface GlobalWeightData {
  feature_type: string;
  layer_idx: number;
  feature_idx: number;
  feature_name: string;
  features_in: Array<{ name: string; weight: number }>;
  features_out: Array<{ name: string; weight: number }>;
}

interface BetweenFeaturesResult {
  featureA: {
    name: string;
    layer: number;
    featureIndex: number;
    featureType: string;
    featuresIn: Array<{ name: string; weight: number }>;
    featuresOut: Array<{ name: string; weight: number }>;
  };
  featureB: {
    name: string;
    layer: number;
    featureIndex: number;
    featureType: string;
    featuresIn: Array<{ name: string; weight: number }>;
    featuresOut: Array<{ name: string; weight: number }>;
  };
  rankAInB: number | null; // Feature A 在 Feature B 的 features_in 中的排名
  rankBInA: number | null; // Feature B 在 Feature A 的 features_out 中的排名
  weightAInB: number | null; // Feature A 在 Feature B 的 features_in 中的权重
  weightBInA: number | null; // Feature B 在 Feature A 的 features_out 中的权重
}

interface TopActivationData {
  fen: string;
  activationStrength: number;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  contextId?: number;
  sampleIndex?: number;
}

// Custom node data type
type FeatureNodeData = Record<string, unknown> & {
  featureId: string;
  featureType: string;
  layer: number;
  featureIndex: number;
  saeName: string;
  saeSeries: string;
  interpretation?: string;
  level?: number;
  nodeColor: string;
  globalWeight?: number;
};

// Custom node component for features
const FeatureNode = ({ data, selected }: { 
  data: FeatureNodeData & { isSecondSelection?: boolean; isSteeringNode?: boolean; isTargetNode?: boolean }; 
  selected?: boolean;
}) => {
  const isSecondSelection = data.isSecondSelection || false;
  const isSteeringNode = data.isSteeringNode || false;
  const isTargetNode = data.isTargetNode || false;
  
  // Priority: target > steering > second selection > normal selection
  const borderColor = isTargetNode
    ? 'border-green-600 border-2'
    : isSteeringNode
    ? 'border-blue-600 border-2'
    : isSecondSelection
    ? 'border-green-500' 
    : selected 
    ? 'border-blue-500' 
    : 'border-gray-300';
  const bgColor = isTargetNode
    ? 'bg-green-100'
    : isSteeringNode
    ? 'bg-blue-100'
    : isSecondSelection
    ? 'bg-green-50'
    : selected
    ? 'bg-blue-50'
    : 'bg-white';

  return (
    <div className={`px-4 py-3 ${bgColor} border-2 ${borderColor} rounded-lg shadow-md min-w-[200px]`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: data.nodeColor || '#95a5a6' }}
          />
          <span className="text-xs font-semibold text-gray-700">
            {data.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L{data.layer} #{data.featureIndex}
          </span>
        </div>
        {data.level !== undefined && (
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
            Level {data.level}
          </span>
        )}
      </div>
      {data.interpretation && (
        <div className="text-xs text-gray-600 mt-1 line-clamp-2">
          {data.interpretation}
        </div>
      )}
      {data.globalWeight !== undefined && (
        <div className="text-xs text-purple-600 mt-1 font-medium">
          Global Weight: {data.globalWeight.toFixed(4)}
        </div>
      )}
    </div>
  );
};

const nodeTypes: NodeTypes = {
  feature: FeatureNode,
};

export const FunctionalMicrocircuitVisualization: React.FC = () => {
  const [circuits, setCircuits] = useState<CircuitAnnotation[]>([]);
  const [selectedCircuitId, setSelectedCircuitId] = useState<string | null>(null);
  const [selectedCircuit, setSelectedCircuit] = useState<CircuitAnnotation | null>(null);
  const [loadingCircuits, setLoadingCircuits] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState<ReactFlowNode<FeatureNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<ReactFlowEdge>([]);

  // Interaction state
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedNodeId2, setSelectedNodeId2] = useState<string | null>(null); // Second node for between-features analysis
  const [selectedSteeringNodeIds, setSelectedSteeringNodeIds] = useState<Set<string>>(new Set()); // Multiple steering nodes
  const [selectedTargetNodeIds, setSelectedTargetNodeIds] = useState<Set<string>>(new Set()); // Multiple target nodes
  const [nodeSelectionMode, setNodeSelectionMode] = useState<"steering" | "target" | null>(null); // Selection mode
  const [draggingFrom, setDraggingFrom] = useState<string | null>(null);

  // Global Weight state
  const [loadingGlobalWeights, setLoadingGlobalWeights] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [betweenFeaturesResult, setBetweenFeaturesResult] = useState<BetweenFeaturesResult | null>(null);

  // Top Activation state
  const [topActivations, setTopActivations] = useState<TopActivationData[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);

  // Edge editing state
  const [editingEdge, setEditingEdge] = useState<{ id: string; weight: number } | null>(null);
  const [edgeWeightInput, setEdgeWeightInput] = useState<string>('0');

  // Level editing state
  const [editingLevel, setEditingLevel] = useState<{ featureId: string; level: number } | null>(null);
  const [levelInput, setLevelInput] = useState<string>('0');

  // Node interaction state
  const [steeringScale, setSteeringScale] = useState<string>('2.0');
  const [interactionResult, setInteractionResult] = useState<any>(null);
  const [analyzingInteraction, setAnalyzingInteraction] = useState(false);
  const [interactionFen, setInteractionFen] = useState<string>('8/p3kpp1/8/3R1r2/8/4P1Q1/PPr4n/6KR b - - 9 32'); // FEN for interaction analysis
  const [steeringNodePositions, setSteeringNodePositions] = useState<Map<string, number>>(new Map()); // Position for each steering node
  const [targetNodePositions, setTargetNodePositions] = useState<Map<string, number>>(new Map()); // Position for each target node
  const [nodeActivations, setNodeActivations] = useState<Map<string, number>>(new Map()); // Activation value for each node at its position
  const [loadingActivations, setLoadingActivations] = useState<Set<string>>(new Set()); // Loading state for activation fetching

  // Feature from FEN state
  const [fenForFeatureSelection, setFenForFeatureSelection] = useState<string>('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [layerForFeatureSelection, setLayerForFeatureSelection] = useState<number>(0);
  const [positionForFeatureSelection, setPositionForFeatureSelection] = useState<number>(0);
  const [componentTypeForFeatureSelection, setComponentTypeForFeatureSelection] = useState<"attn" | "mlp">("attn");
  const [autoCreateEdges, setAutoCreateEdges] = useState<boolean>(false); // 默认不自动创建边

  // Get SAE combo ID (use useState to avoid re-reading on every render)
  const [saeComboId, setSaeComboId] = useState<string | undefined>(() => {
    if (typeof window !== 'undefined') {
      return window.localStorage.getItem('bt4_sae_combo_id') || undefined;
    }
    return undefined;
  });

  // Listen for localStorage changes
  useEffect(() => {
    const handleStorageChange = () => {
      if (typeof window !== 'undefined') {
        const newComboId = window.localStorage.getItem('bt4_sae_combo_id') || undefined;
        setSaeComboId(newComboId);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Load all circuits
  const loadCircuits = useCallback(async () => {
    if (!saeComboId) return; // Don't load if no combo ID
    
    setLoadingCircuits(true);
    setError(null);
    try {
      const result = await listCircuitAnnotations(saeComboId, 100, 0);
      setCircuits(result.circuits);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load circuits');
      console.error('Failed to load circuits:', err);
    } finally {
      setLoadingCircuits(false);
    }
  }, [saeComboId]);

  // Load selected circuit
  const loadCircuit = useCallback(async (circuitId: string) => {
    setError(null);
    try {
      const circuit = await getCircuitAnnotation(circuitId);
      setSelectedCircuit(circuit);
      setSelectedCircuitId(circuitId);
      
      // Convert circuit to graph data
      convertCircuitToGraph(circuit);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load circuit');
      console.error('Failed to load circuit:', err);
    }
  }, []);

  // Convert circuit annotation to graph nodes and edges
  const convertCircuitToGraph = useCallback((circuit: CircuitAnnotation) => {
    if (!circuit.features || circuit.features.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    // Create nodes from features
    const graphNodes: ReactFlowNode<FeatureNodeData>[] = circuit.features.map((feature, index) => {
      const featureId = feature.feature_id || `feature_${index}`;
      const nodeId = `node_${featureId}`;
      
      // Generate node color based on feature type
      const nodeColor = feature.feature_type === 'lorsa' ? '#7a4cff' : '#4ecdc4';
      
      // Calculate position based on level (if available) or layer
      const level = feature.level !== undefined ? feature.level : feature.layer;
      const x = level * 300 + 100; // Horizontal spacing by level
      const y = (index % 5) * 150 + 100; // Vertical spacing

      return {
        id: nodeId,
        type: 'feature',
        position: { x, y },
        data: {
          featureId,
          featureType: feature.feature_type,
          layer: feature.layer,
          featureIndex: feature.feature_index,
          saeName: feature.sae_name,
          saeSeries: feature.sae_series,
          interpretation: feature.interpretation,
          level: feature.level,
          nodeColor,
        },
      };
    });

    // Create edges from circuit edges
    const graphEdges: ReactFlowEdge[] = (circuit.edges || [])
      .map((edge, index) => {
        const sourceNode = graphNodes.find(n => n.data.featureId === edge.source_feature_id);
        const targetNode = graphNodes.find(n => n.data.featureId === edge.target_feature_id);
        
        if (!sourceNode || !targetNode) return null;

        const edgeData: Record<string, unknown> = {
          sourceFeatureId: edge.source_feature_id,
          targetFeatureId: edge.target_feature_id,
          weight: edge.weight,
          interpretation: edge.interpretation,
        };

        return {
          id: `edge_${index}`,
          source: sourceNode.id,
          target: targetNode.id,
          type: 'smoothstep' as const,
          animated: Math.abs(edge.weight) > 0.1,
          style: {
            strokeWidth: Math.max(1, Math.min(5, Math.abs(edge.weight) * 10)),
            stroke: edge.weight > 0 ? '#4CAF50' : '#F44336',
          },
          label: edge.weight.toFixed(3),
          markerEnd: {
            type: MarkerType.ArrowClosed,
          },
          data: edgeData,
        } as ReactFlowEdge;
      })
      .filter((edge): edge is ReactFlowEdge => edge !== null);

    setNodes(graphNodes);
    setEdges(graphEdges);
  }, [setNodes, setEdges]);

  // Load circuits on mount (only once)
  // 使用 ref 来跟踪是否已经加载过，避免重复加载
  const hasLoadedCircuitsRef = useRef(false);
  
  useEffect(() => {
    if (saeComboId && !hasLoadedCircuitsRef.current) {
      hasLoadedCircuitsRef.current = true;
      loadCircuits();
    } else if (!saeComboId) {
      // 如果 saeComboId 被清空，重置标志
      hasLoadedCircuitsRef.current = false;
    }
    // 只在 saeComboId 变化时重新加载，避免频繁调用
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saeComboId]);

  // Convert circuit when selected circuit changes
  useEffect(() => {
    if (selectedCircuit) {
      convertCircuitToGraph(selectedCircuit);
    }
  }, [selectedCircuit, convertCircuitToGraph]);

  // Handle node drag start (for creating edges)
  const onNodeDragStart = useCallback((_event: React.MouseEvent, node: ReactFlowNode<FeatureNodeData>) => {
    setDraggingFrom(node.id);
  }, []);

  // Handle node drag stop (for creating edges)
  const onNodeDragStop = useCallback((_event: React.MouseEvent, node: ReactFlowNode<FeatureNodeData>) => {
    if (draggingFrom && draggingFrom !== node.id) {
      // Check if edge already exists
      const existingEdge = edges.find(
        e => (e.source === draggingFrom && e.target === node.id) ||
             (e.source === node.id && e.target === draggingFrom)
      );

      if (!existingEdge && selectedCircuit) {
        const sourceNode = nodes.find(n => n.id === draggingFrom);
        const targetNode = nodes.find(n => n.id === node.id);
        
        if (sourceNode && targetNode) {
          // Create new edge
          const newEdge: ReactFlowEdge = {
            id: `edge_${Date.now()}`,
            source: draggingFrom,
            target: node.id,
            type: 'smoothstep',
            style: {
              strokeWidth: 2,
              stroke: '#4CAF50',
            },
            label: '0.000',
            markerEnd: {
              type: MarkerType.ArrowClosed,
            },
            data: {
              sourceFeatureId: sourceNode.data.featureId,
              targetFeatureId: targetNode.data.featureId,
              weight: 0,
            },
          };

          setEdges((eds) => [...eds, newEdge]);
          
          // Save to backend
          addEdgeToCircuit(
            selectedCircuit.circuit_id,
            sourceNode.data.featureId,
            targetNode.data.featureId,
            0.0
          ).catch(err => {
            console.error('Failed to add edge:', err);
            // Revert edge on error
            setEdges((eds) => eds.filter(e => e.id !== newEdge.id));
          });
        }
      }
    }
    setDraggingFrom(null);
  }, [draggingFrom, edges, nodes, selectedCircuit, setEdges]);

  // Handle node click
  const onNodeClick = useCallback((_event: React.MouseEvent, node: ReactFlowNode<FeatureNodeData>) => {
    // If Ctrl/Cmd is pressed, select as second node (for backward compatibility)
    if (_event.ctrlKey || _event.metaKey && !nodeSelectionMode) {
      setSelectedNodeId2(node.id === selectedNodeId2 ? null : node.id);
    } else if (nodeSelectionMode === "steering") {
      // Steering node selection mode
      setSelectedSteeringNodeIds(prev => {
        const newSet = new Set(prev);
        if (newSet.has(node.id)) {
          newSet.delete(node.id);
          // Remove position when node is deselected
          setSteeringNodePositions(prevPos => {
            const newPos = new Map(prevPos);
            newPos.delete(node.id);
            return newPos;
          });
    } else {
          newSet.add(node.id);
          // Don't set default position - user must specify it
        }
        return newSet;
      });
    } else if (nodeSelectionMode === "target") {
      // Target node selection mode
      setSelectedTargetNodeIds(prev => {
        const newSet = new Set(prev);
        if (newSet.has(node.id)) {
          newSet.delete(node.id);
          // Remove position when node is deselected
          setTargetNodePositions(prevPos => {
            const newPos = new Map(prevPos);
            newPos.delete(node.id);
            return newPos;
          });
        } else {
          newSet.add(node.id);
          // Don't set default position - user must specify it
        }
        return newSet;
      });
    } else {
      // Default behavior: single node selection
      setSelectedNodeId(node.id === selectedNodeId ? null : node.id);
      // Clear second selection if clicking first node
      if (node.id === selectedNodeId) {
        setSelectedNodeId2(null);
      }
    }
  }, [selectedNodeId, selectedNodeId2, nodeSelectionMode]);

  // Handle node hover (for future use)
  const onNodeMouseEnter = useCallback((_event: React.MouseEvent, _node: ReactFlowNode<FeatureNodeData>) => {
    // Can be used for hover effects in the future
  }, []);

  const onNodeMouseLeave = useCallback(() => {
    // Can be used for hover effects in the future
  }, []);

  // Preload models before fetching global weight
  const preloadModels = useCallback(async (comboId: string): Promise<void> => {
    setLoadingMessage('正在检查模型加载状态...');
    
    // Check loading status
    const checkLoadingStatus = async (): Promise<{ isLoading: boolean; logs?: Array<{ timestamp: number; message: string }> }> => {
      try {
        const logParams = new URLSearchParams({
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: comboId,
        });
        const logRes = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/loading_logs?${logParams.toString()}`);
        if (logRes.ok) {
          const logData: { is_loading?: boolean; logs?: Array<{ timestamp: number; message: string }> } = await logRes.json();
          return { isLoading: logData.is_loading ?? false, logs: logData.logs };
        }
      } catch (err) {
        console.warn('Failed to check loading status:', err);
      }
      return { isLoading: false };
    };

    // If already loading, wait for completion
    let status = await checkLoadingStatus();
    if (status.isLoading) {
      setLoadingMessage('检测到模型正在加载中，等待加载完成...');
      const maxWaitTime = 300000; // 5 minutes max
      const startTime = Date.now();
      let lastLogCount = status.logs?.length ?? 0;
      while (status.isLoading && Date.now() - startTime < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        status = await checkLoadingStatus();
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`加载中: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
      }
      if (status.isLoading) {
        throw new Error('模型加载超时，请稍后重试');
      }
      setLoadingMessage('模型加载完成');
      return;
    }

    // Call preload endpoint
    setLoadingMessage('开始预加载模型...');
    const preloadRes = await fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit/preload_models`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_name: 'lc0/BT4-1024x15x32h',
        sae_combo_id: comboId,
      }),
    });

    if (!preloadRes.ok) {
      const errorText = await preloadRes.text();
      throw new Error(`预加载失败: HTTP ${preloadRes.status}: ${errorText}`);
    }

    const preloadData = await preloadRes.json();
    
    if (preloadData.status === 'already_loaded') {
      setLoadingMessage('模型已加载');
      return;
    }

    if (preloadData.status === 'loaded' || preloadData.status === 'loading') {
      setLoadingMessage('等待模型加载完成...');
      const maxWaitTime = 300000;
      const startTime = Date.now();
      let lastLogCount = 0;
      while (Date.now() - startTime < maxWaitTime) {
        status = await checkLoadingStatus();
        if (status.logs && status.logs.length > lastLogCount) {
          const lastLog = status.logs[status.logs.length - 1];
          setLoadingMessage(`加载中: ${lastLog.message}`);
          lastLogCount = status.logs.length;
        }
        if (!status.isLoading) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          setLoadingMessage('模型加载完成');
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      throw new Error('模型加载超时，请稍后重试');
    }
  }, []);

  // Helper function to build feature name
  const buildFeatureName = useCallback((featureType: string, layer: number, featureIndex: number): string => {
    const typePrefix = featureType === 'lorsa' ? 'BT4_lorsa' : 'BT4_tc';
    const layerSuffix = featureType === 'lorsa' ? 'A' : 'M';
    // For k_30_e_16, add suffix
    if (saeComboId === 'k_30_e_16') {
      return `${typePrefix}_L${layer}${layerSuffix}_k30_e16#${featureIndex}`;
    } else if (saeComboId === 'k_64_e_32') {
      return `${typePrefix}_L${layer}${layerSuffix}_k64_e32#${featureIndex}`;
    } else if (saeComboId === 'k_128_e_64') {
      return `${typePrefix}_L${layer}${layerSuffix}_k128_e64#${featureIndex}`;
    } else if (saeComboId === 'k_256_e_128') {
      return `${typePrefix}_L${layer}${layerSuffix}_k256_e128#${featureIndex}`;
    } else {
      return `${typePrefix}_L${layer}${layerSuffix}#${featureIndex}`;
    }
  }, [saeComboId]);

  // Helper function to find feature rank in a list
  const findFeatureRank = useCallback((featureName: string, featureList: Array<{ name: string; weight: number }>): { rank: number | null; weight: number | null } => {
    for (let i = 0; i < featureList.length; i++) {
      if (featureList[i].name === featureName) {
        return { rank: i + 1, weight: featureList[i].weight };
      }
    }
    return { rank: null, weight: null };
  }, []);

  // Fetch Global Weight between two features
  const fetchGlobalWeightBetweenFeatures = useCallback(async (
    layerA: number, featureIndexA: number, featureTypeA: string,
    layerB: number, featureIndexB: number, featureTypeB: string
  ) => {
    if (!saeComboId) {
      alert('请先选择 SAE 组合');
      return;
    }

    setLoadingGlobalWeights(true);
    setLoadingMessage(null);
    try {
      // Preload models first
      await preloadModels(saeComboId);

      // Fetch global weight for Feature A
      setLoadingMessage('正在计算 Feature A 的全局权重...');
      const paramsA = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureTypeA === 'lorsa' ? 'lorsa' : 'tc',
        layer_idx: layerA.toString(),
        feature_idx: featureIndexA.toString(),
        k: '100',
        sae_combo_id: saeComboId,
      });

      let responseA = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${paramsA.toString()}`);
      if (responseA.status === 503) {
        setLoadingMessage('模型未加载，正在自动加载...');
        await preloadModels(saeComboId);
        setLoadingMessage('正在计算 Feature A 的全局权重...');
        responseA = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${paramsA.toString()}`);
      }
      if (!responseA.ok) {
        throw new Error(`Failed to fetch Feature A: HTTP ${responseA.status}`);
      }
      const dataA: GlobalWeightData = await responseA.json();

      // Fetch global weight for Feature B
      setLoadingMessage('正在计算 Feature B 的全局权重...');
      const paramsB = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureTypeB === 'lorsa' ? 'lorsa' : 'tc',
        layer_idx: layerB.toString(),
        feature_idx: featureIndexB.toString(),
        k: '100',
        sae_combo_id: saeComboId,
      });

      let responseB = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${paramsB.toString()}`);
      if (responseB.status === 503) {
        setLoadingMessage('模型未加载，正在自动加载...');
        await preloadModels(saeComboId);
        setLoadingMessage('正在计算 Feature B 的全局权重...');
        responseB = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${paramsB.toString()}`);
      }
      if (!responseB.ok) {
        throw new Error(`Failed to fetch Feature B: HTTP ${responseB.status}`);
      }
      const dataB: GlobalWeightData = await responseB.json();

      // Build feature names
      const featureAName = buildFeatureName(featureTypeA, layerA, featureIndexA);
      const featureBName = buildFeatureName(featureTypeB, layerB, featureIndexB);

      // Find ranks
      // Feature A 在 Feature B 的 features_in 中的排名（A 影响 B）
      const rankAInB = findFeatureRank(featureAName, dataB.features_in);
      // Feature B 在 Feature A 的 features_out 中的排名（A 影响 B）
      const rankBInA = findFeatureRank(featureBName, dataA.features_out);

      setBetweenFeaturesResult({
        featureA: {
          name: featureAName,
          layer: layerA,
          featureIndex: featureIndexA,
          featureType: featureTypeA,
          featuresIn: dataA.features_in,
          featuresOut: dataA.features_out,
        },
        featureB: {
          name: featureBName,
          layer: layerB,
          featureIndex: featureIndexB,
          featureType: featureTypeB,
          featuresIn: dataB.features_in,
          featuresOut: dataB.features_out,
        },
        rankAInB: rankAInB.rank,
        rankBInA: rankBInA.rank,
        weightAInB: rankAInB.weight,
        weightBInA: rankBInA.weight,
      });

      setLoadingMessage(null);
    } catch (err) {
      console.error('Failed to fetch global weight between features:', err);
      alert('获取两个特征之间的全局权重失败: ' + (err instanceof Error ? err.message : '未知错误'));
      setBetweenFeaturesResult(null);
    } finally {
      setLoadingGlobalWeights(false);
      setLoadingMessage(null);
    }
  }, [saeComboId, preloadModels, buildFeatureName, findFeatureRank]);

  // Fetch activation value for a node at a specific position
  const fetchNodeActivation = useCallback(async (nodeId: string, position: number) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node || !interactionFen || !saeComboId) {
      return null;
    }

    setLoadingActivations(prev => new Set(prev).add(nodeId));
    try {
      const componentType = node.data.featureType === 'lorsa' ? 'attn' : 'mlp';
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/activation/get_features_at_position`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          fen: interactionFen.trim(),
          layer: node.data.layer,
          pos: position,
          component_type: componentType,
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: saeComboId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      const features = componentType === 'attn' ? data.attn_features : data.mlp_features;
      const feature = features?.find((f: any) => f.feature_index === node.data.featureIndex);
      
      if (feature) {
        setNodeActivations(prev => {
          const newMap = new Map(prev);
          newMap.set(nodeId, feature.activation_value);
          return newMap;
        });
        return feature.activation_value;
      }
      return 0;
    } catch (error) {
      console.error('Failed to fetch node activation:', error);
      return null;
    } finally {
      setLoadingActivations(prev => {
        const newSet = new Set(prev);
        newSet.delete(nodeId);
        return newSet;
      });
    }
  }, [nodes, interactionFen, saeComboId]);

  // Analyze interaction between selected nodes
  const analyzeNodeInteraction = useCallback(async () => {
    const fen = interactionFen || (selectedCircuit as any)?.original_fen || '8/p3kpp1/8/3R1r2/8/4P1Q1/PPr4n/6KR b - - 9 32';
    
    if (!fen) {
      alert('请先输入 FEN 字符串');
      return;
    }

    // Check if using new multi-node selection mode
    const useMultiNodeMode = selectedSteeringNodeIds.size > 0 && selectedTargetNodeIds.size > 0;
    
    if (!useMultiNodeMode) {
      // Fallback to old two-node mode for backward compatibility
    if (!selectedNodeId || !selectedNodeId2 || !selectedCircuit) {
        alert('请选择两个节点来进行交互分析，或者使用多节点选择模式');
      return;
    }

    const nodeA = nodes.find(n => n.id === selectedNodeId);
    const nodeB = nodes.find(n => n.id === selectedNodeId2);

    if (!nodeA || !nodeB) {
      alert('找不到选中的节点');
      return;
    }

      // Check positions
      const nodeAPos = steeringNodePositions.get(selectedNodeId);
      const nodeBPos = targetNodePositions.get(selectedNodeId2);
      
      if (nodeAPos === undefined) {
        alert(`请为 steering node (${nodeA.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeA.data?.layer} #${nodeA.data?.featureIndex}) 指定 position`);
        return;
      }
      if (nodeBPos === undefined) {
        alert(`请为 target node (${nodeB.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeB.data?.layer} #${nodeB.data?.featureIndex}) 指定 position`);
        return;
      }

      // Check activation values
      const nodeAActivation = nodeActivations.get(selectedNodeId);
      const nodeBActivation = nodeActivations.get(selectedNodeId2);
      
      if (nodeAActivation === undefined) {
        // Fetch activation
        const activation = await fetchNodeActivation(selectedNodeId, nodeAPos);
        if (activation === null || activation === 0) {
          alert(`Steering node (${nodeA.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeA.data?.layer} #${nodeA.data?.featureIndex}) 在 position ${nodeAPos} 的激活值为 0，无法进行分析`);
          return;
        }
      } else if (nodeAActivation === 0) {
        alert(`Steering node (${nodeA.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeA.data?.layer} #${nodeA.data?.featureIndex}) 在 position ${nodeAPos} 的激活值为 0，无法进行分析`);
        return;
      }

      if (nodeBActivation === undefined) {
        // Fetch activation
        const activation = await fetchNodeActivation(selectedNodeId2, nodeBPos);
        if (activation === null || activation === 0) {
          alert(`Target node (${nodeB.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeB.data?.layer} #${nodeB.data?.featureIndex}) 在 position ${nodeBPos} 的激活值为 0，无法进行分析`);
          return;
        }
      } else if (nodeBActivation === 0) {
        alert(`Target node (${nodeB.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${nodeB.data?.layer} #${nodeB.data?.featureIndex}) 在 position ${nodeBPos} 的激活值为 0，无法进行分析`);
        return;
      }

    setAnalyzingInteraction(true);
    try {
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/interaction/analyze_node_interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: saeComboId,
            fen: fen,
            steering_nodes: [{
            feature_type: nodeA.data.featureType,
            layer: nodeA.data.layer,
            feature: nodeA.data.featureIndex,
              pos: nodeAPos,
            }],
            target_nodes: [{
            feature_type: nodeB.data.featureType,
            layer: nodeB.data.layer,
            feature: nodeB.data.featureIndex,
              pos: nodeBPos,
            }],
          steering_scale: parseFloat(steeringScale),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      setInteractionResult(result);
    } catch (error) {
      console.error('Failed to analyze node interaction:', error);
      alert(`分析节点交互失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setAnalyzingInteraction(false);
    }
      return;
    }

    // New multi-node mode
    if (!selectedCircuit) {
      alert('请先选择一个 Circuit');
      return;
    }

    const steeringNodes = Array.from(selectedSteeringNodeIds)
      .map(id => nodes.find(n => n.id === id))
      .filter((n): n is ReactFlowNode<FeatureNodeData> => n !== undefined);
    
    const targetNodes = Array.from(selectedTargetNodeIds)
      .map(id => nodes.find(n => n.id === id))
      .filter((n): n is ReactFlowNode<FeatureNodeData> => n !== undefined);

    if (steeringNodes.length === 0 || targetNodes.length === 0) {
      alert('请至少选择一个 steering node 和一个 target node');
      return;
    }

    // Check all positions are specified
    for (const node of steeringNodes) {
      const pos = steeringNodePositions.get(node.id);
      if (pos === undefined) {
        alert(`请为 steering node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 指定 position`);
        return;
      }
    }

    for (const node of targetNodes) {
      const pos = targetNodePositions.get(node.id);
      if (pos === undefined) {
        alert(`请为 target node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 指定 position`);
        return;
      }
    }

    // Check all activation values are non-zero
    for (const node of steeringNodes) {
      const pos = steeringNodePositions.get(node.id)!;
      const activation = nodeActivations.get(node.id);
      
      if (activation === undefined) {
        // Fetch activation
        const fetchedActivation = await fetchNodeActivation(node.id, pos);
        if (fetchedActivation === null || fetchedActivation === 0) {
          alert(`Steering node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 在 position ${pos} 的激活值为 0，无法进行分析`);
          return;
        }
      } else if (activation === 0) {
        alert(`Steering node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 在 position ${pos} 的激活值为 0，无法进行分析`);
        return;
      }
    }

    for (const node of targetNodes) {
      const pos = targetNodePositions.get(node.id)!;
      const activation = nodeActivations.get(node.id);
      
      if (activation === undefined) {
        // Fetch activation
        const fetchedActivation = await fetchNodeActivation(node.id, pos);
        if (fetchedActivation === null || fetchedActivation === 0) {
          alert(`Target node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 在 position ${pos} 的激活值为 0，无法进行分析`);
          return;
        }
      } else if (activation === 0) {
        alert(`Target node (${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}) 在 position ${pos} 的激活值为 0，无法进行分析`);
        return;
      }
    }

    setAnalyzingInteraction(true);
    try {
      // Get positions for each node
      const steeringNodesWithPos = steeringNodes.map(node => ({
        feature_type: node.data.featureType,
        layer: node.data.layer,
        feature: node.data.featureIndex,
        pos: steeringNodePositions.get(node.id)!,
      }));
      
      const targetNodesWithPos = targetNodes.map(node => ({
        feature_type: node.data.featureType,
        layer: node.data.layer,
        feature: node.data.featureIndex,
        pos: targetNodePositions.get(node.id)!,
      }));
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/interact/analyze_node_interaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: 'lc0/BT4-1024x15x32h',
          sae_combo_id: saeComboId,
          fen: fen,
          steering_nodes: steeringNodesWithPos,
          target_nodes: targetNodesWithPos,
          steering_scale: parseFloat(steeringScale),
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      setInteractionResult(result);
    } catch (error) {
      console.error('Failed to analyze node interaction:', error);
      alert(`分析节点交互失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      setAnalyzingInteraction(false);
    }
  }, [
    selectedNodeId,
    selectedNodeId2,
    selectedSteeringNodeIds,
    selectedTargetNodeIds,
    nodes,
    selectedCircuit,
    saeComboId,
    steeringScale,
    interactionFen,
    steeringNodePositions,
    targetNodePositions,
    nodeActivations,
    fetchNodeActivation,
  ]);

  // Fetch Top Activations for a feature
  const fetchTopActivations = useCallback(async (layer: number, featureIndex: number, featureType: string) => {
    if (!saeComboId) {
      alert('请先选择 SAE 组合');
      return;
    }

    setLoadingTopActivations(true);
    let dictionary: string = '';
    
    try {
      // 优先从 circuit annotation 的 feature 中获取 sae_name
      let foundDictionary: string | undefined = undefined;
      
      if (selectedCircuit) {
        const feature = selectedCircuit.features.find(
          f => f.layer === layer && 
               f.feature_index === featureIndex && 
               f.feature_type === featureType
        );
        if (feature?.sae_name) {
          foundDictionary = feature.sae_name;
        }
      }
      
      // 如果从 circuit 中找不到，根据 sae_combo_id 生成
      if (!foundDictionary) {
        const isLorsa = featureType === 'lorsa';
        // 根据 sae_combo_id 生成正确的 dictionary 名称
        // 对于 k_30_e_16: BT4_tc_L{layer}M_k30_e16 或 BT4_lorsa_L{layer}A_k30_e16
        // 对于 k_128_e_128 (默认): BT4_tc_L{layer}M 或 BT4_lorsa_L{layer}A
        if (saeComboId === 'k_30_e_16') {
          foundDictionary = isLorsa 
            ? `BT4_lorsa_L${layer}A_k30_e16`
            : `BT4_tc_L${layer}M_k30_e16`;
        } else if (saeComboId === 'k_64_e_32') {
          foundDictionary = isLorsa 
            ? `BT4_lorsa_L${layer}A_k64_e32`
            : `BT4_tc_L${layer}M_k64_e32`;
        } else if (saeComboId === 'k_128_e_64') {
          foundDictionary = isLorsa 
            ? `BT4_lorsa_L${layer}A_k128_e64`
            : `BT4_tc_L${layer}M_k128_e64`;
        } else if (saeComboId === 'k_256_e_128') {
          foundDictionary = isLorsa 
            ? `BT4_lorsa_L${layer}A_k256_e128`
            : `BT4_tc_L${layer}M_k256_e128`;
        } else {
          // 默认组合 k_128_e_128
          foundDictionary = isLorsa 
            ? `BT4_lorsa_L${layer}A`
            : `BT4_tc_L${layer}M`;
        }
      }
      
      dictionary = foundDictionary;
      
      console.log('🔍 获取 Top Activation 数据:', {
        layer,
        featureIndex,
        featureType,
        saeComboId,
        dictionary,
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
      
      console.log('📡 Top Activation API 请求:', {
        url: `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        status: response.status,
        statusText: response.statusText,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const decoded = await import('@msgpack/msgpack').then(m => m.decode(new Uint8Array(arrayBuffer)));
      const camelcaseKeys = await import('camelcase-keys').then(m => m.default);

      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ['sample_groups.samples.context'],
      }) as any;

      // Extract chess samples
      const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
      const allSamples: any[] = [];
      
      for (const group of sampleGroups) {
        if (group.samples && Array.isArray(group.samples)) {
          allSamples.push(...group.samples);
        }
      }

      const chessSamples: TopActivationData[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
                  // Validate board format
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
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0;
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      activationsArray = new Array(64).fill(0);
                      
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation,
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0,
                    });
                    
                    break; // Found valid FEN, move to next sample
                  }
                }
              }
            }
          }
        }
      }
      
      console.log('🔍 Top Activations提取结果:', {
        totalSamples: allSamples.length,
        chessSamplesFound: chessSamples.length,
        dictionary,
        layer,
        featureIndex,
        featureType,
      });

      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);

      console.log('✅ Top Activations最终结果:', {
        topSamplesCount: topSamples.length,
        samples: topSamples.map(s => ({
          fen: s.fen.substring(0, 20) + '...',
          activationStrength: s.activationStrength,
          hasActivations: !!s.activations,
        })),
      });

      setTopActivations(topSamples);
    } catch (err) {
      console.error('❌ Failed to fetch top activations:', err);
      const errorMessage = err instanceof Error ? err.message : '未知错误';
      alert(`获取Top Activations失败: ${errorMessage}\n\n使用的字典名称: ${dictionary || '未确定'}\n请检查字典名称是否正确。`);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [saeComboId, selectedCircuit]);

  // Handle edge click (for editing weight)
  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: ReactFlowEdge) => {
    const edgeData = edge.data as { weight?: number } | undefined;
    setEditingEdge({
      id: edge.id,
      weight: edgeData?.weight || 0,
    });
    setEdgeWeightInput((edgeData?.weight || 0).toString());
  }, []);

  // Save edge weight
  const handleSaveEdgeWeight = useCallback(async () => {
    if (!editingEdge || !selectedCircuit) return;

    const edge = edges.find(e => e.id === editingEdge.id);
    if (!edge) return;
    
    const edgeData = edge.data as { sourceFeatureId?: string; targetFeatureId?: string; weight?: number } | undefined;
    if (!edgeData || !edgeData.sourceFeatureId || !edgeData.targetFeatureId) return;

    try {
      await updateEdgeWeight(
        selectedCircuit.circuit_id,
        edgeData.sourceFeatureId,
        edgeData.targetFeatureId,
        parseFloat(edgeWeightInput) || 0
      );

      // Update edge in graph
      setEdges((eds) => eds.map(e => {
        if (e.id === editingEdge.id) {
          const weight = parseFloat(edgeWeightInput) || 0;
          return {
            ...e,
            label: weight.toFixed(3),
            style: {
              ...e.style,
              strokeWidth: Math.max(1, Math.min(5, Math.abs(weight) * 10)),
              stroke: weight > 0 ? '#4CAF50' : '#F44336',
            },
            data: {
              ...(e.data || {}),
              weight,
            },
          };
        }
        return e;
      }));

      setEditingEdge(null);
      setEdgeWeightInput('0');
    } catch (err) {
      console.error('Failed to update edge weight:', err);
      alert('Failed to update edge weight: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }, [editingEdge, edgeWeightInput, selectedCircuit, edges, setEdges]);

  // Delete edge
  const handleDeleteEdge = useCallback(async (edgeId: string) => {
    if (!selectedCircuit) return;

    const edge = edges.find(e => e.id === edgeId);
    if (!edge) return;
    
    const edgeData = edge.data as { sourceFeatureId?: string; targetFeatureId?: string } | undefined;
    if (!edgeData || !edgeData.sourceFeatureId || !edgeData.targetFeatureId) return;

    try {
      await removeEdgeFromCircuit(
        selectedCircuit.circuit_id,
        edgeData.sourceFeatureId,
        edgeData.targetFeatureId
      );

      setEdges((eds) => eds.filter(e => e.id !== edgeId));
    } catch (err) {
      console.error('Failed to delete edge:', err);
      alert('Failed to delete edge: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }, [selectedCircuit, edges, setEdges]);

  // Save feature level
  const handleSaveLevel = useCallback(async () => {
    if (!editingLevel || !selectedCircuit) return;

    try {
      await setFeatureLevel(
        selectedCircuit.circuit_id,
        editingLevel.featureId,
        parseInt(levelInput) || 0
      );

      // Update node in graph
      setNodes((nds) => nds.map(node => {
        if (node.data?.featureId === editingLevel.featureId) {
          const newLevel = parseInt(levelInput) || 0;
          return {
            ...node,
            position: {
              x: newLevel * 300 + 100,
              y: node.position.y,
            },
            data: {
              ...node.data,
              level: newLevel,
            },
          };
        }
        return node;
      }));

      // Update selected circuit
      setSelectedCircuit(prev => {
        if (!prev) return null;
        return {
          ...prev,
          features: prev.features.map(f => 
            f.feature_id === editingLevel.featureId
              ? { ...f, level: parseInt(levelInput) || 0 }
              : f
          ),
        };
      });

      setEditingLevel(null);
      setLevelInput('0');
    } catch (err) {
      console.error('Failed to set feature level:', err);
      alert('Failed to set feature level: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }, [editingLevel, levelInput, selectedCircuit, setNodes]);

  // Get selected nodes
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodes.find(n => n.id === selectedNodeId);
  }, [selectedNodeId, nodes]);

  const selectedNode2 = useMemo(() => {
    if (!selectedNodeId2) return null;
    return nodes.find(n => n.id === selectedNodeId2);
  }, [selectedNodeId2, nodes]);

  // Helper function to build dictionary name
  const buildDictionaryName = useCallback((layer: number, featureType: string): string => {
    const isLorsa = featureType === 'lorsa';
    const lorsaSuffix = isLorsa ? 'A' : 'M';
    const baseDict = `BT4_${isLorsa ? 'lorsa' : 'tc'}_L${layer}${lorsaSuffix}`;
    
    if (saeComboId && saeComboId !== 'k_128_e_128') {
      const comboParts = saeComboId.replace(/k_(\d+)_e_(\d+)/, 'k$1_e$2');
      return `${baseDict}_${comboParts}`;
    }
    return baseDict;
  }, [saeComboId]);

  // Handle adding feature to circuit
  const handleAddFeatureToCircuit = useCallback(async (
    featureIndex: number
  ) => {
    if (!saeComboId) {
      alert('请先选择 SAE 组合');
      return;
    }

    try {
      const featureType = componentTypeForFeatureSelection === 'attn' ? 'lorsa' : 'transcoder';
      const dictionaryName = buildDictionaryName(layerForFeatureSelection, featureType);
      
      // Build feature object
      const newFeatureId = `feature_${layerForFeatureSelection}_${featureIndex}_${featureType}_${Date.now()}`;
      const newFeature: CircuitFeature = {
        sae_name: dictionaryName,
        sae_series: 'BT4-exp128',
        layer: layerForFeatureSelection,
        feature_index: featureIndex,
        feature_type: featureType,
        feature_id: newFeatureId,
      };

      if (selectedCircuit) {
        // Add to existing circuit
        await addFeatureToCircuit(selectedCircuit.circuit_id, newFeature);
        
        // If auto-create edges is enabled, create edges to all existing features
        if (autoCreateEdges && selectedCircuit.features && selectedCircuit.features.length > 0) {
          const edgesToCreate: Array<{ source: string; target: string }> = [];
          
          // Create edges from new feature to all existing features
          for (const existingFeature of selectedCircuit.features) {
            if (existingFeature.feature_id) {
              edgesToCreate.push({
                source: newFeatureId,
                target: existingFeature.feature_id,
              });
            }
          }
          
          // Create edges from all existing features to new feature
          for (const existingFeature of selectedCircuit.features) {
            if (existingFeature.feature_id) {
              edgesToCreate.push({
                source: existingFeature.feature_id,
                target: newFeatureId,
              });
            }
          }
          
          // Add all edges with weight 0
          for (const edge of edgesToCreate) {
            try {
              await addEdgeToCircuit(
                selectedCircuit.circuit_id,
                edge.source,
                edge.target,
                0.0
              );
            } catch (err) {
              console.warn(`Failed to create edge ${edge.source} -> ${edge.target}:`, err);
            }
          }
        }
        
        // Reload circuit to get updated data
        await loadCircuit(selectedCircuit.circuit_id);
        
        alert(`Feature L${layerForFeatureSelection} #${featureIndex} 已添加到 circuit${autoCreateEdges ? '（已自动创建边）' : ''}`);
      } else {
        // Create new circuit
        const circuitInterpretation = prompt('请输入新 Circuit 的名称/解释:');
        if (!circuitInterpretation) {
          return; // User cancelled
        }

        const newCircuit = await createCircuitAnnotation(
          circuitInterpretation,
          saeComboId,
          [newFeature],
          [], // No edges by default
          {
            original_fen: fenForFeatureSelection,
            created_from: 'feature_selection',
          }
        );

        // Load the new circuit
        await loadCircuit(newCircuit.circuit_id);
        await loadCircuits(); // Refresh circuit list
        
        alert(`已创建新 Circuit: ${circuitInterpretation}`);
      }
    } catch (err) {
      console.error('Failed to add feature to circuit:', err);
      alert(`添加 Feature 失败: ${err instanceof Error ? err.message : '未知错误'}`);
    }
  }, [
    saeComboId,
    selectedCircuit,
    layerForFeatureSelection,
    componentTypeForFeatureSelection,
    fenForFeatureSelection,
    autoCreateEdges,
    buildDictionaryName,
    loadCircuit,
    loadCircuits,
  ]);

  // Handle removing feature from circuit
  const handleRemoveFeatureFromCircuit = useCallback(async (
    featureIndex: number,
    layer: number,
    componentType: "attn" | "mlp"
  ) => {
    if (!selectedCircuit) {
      alert('请先选择一个 Circuit');
      return;
    }

    if (!saeComboId) {
      alert('请先选择 SAE 组合');
      return;
    }

    if (!confirm(`确定要从 Circuit 中删除 Feature L${layer} #${featureIndex} 吗？`)) {
      return;
    }

    try {
      const featureType = componentType === 'attn' ? 'lorsa' : 'transcoder';
      const dictionaryName = buildDictionaryName(layer, featureType);
      
      // Find sae_series from circuit features or use default
      const circuitFeature = selectedCircuit.features?.find(
        f => f.layer === layer && 
        f.feature_index === featureIndex && 
        f.feature_type === featureType
      );
      
      const saeSeries = circuitFeature?.sae_series || 'BT4-exp128';

      await removeFeatureFromCircuit(
        selectedCircuit.circuit_id,
        dictionaryName,
        saeSeries,
        layer,
        featureIndex,
        featureType
      );
      
      // Reload circuit to get updated data
      await loadCircuit(selectedCircuit.circuit_id);
      
      alert(`Feature L${layer} #${featureIndex} 已从 circuit 中删除`);
    } catch (err) {
      console.error('Failed to remove feature from circuit:', err);
      alert(`删除 Feature 失败: ${err instanceof Error ? err.message : '未知错误'}`);
    }
  }, [
    selectedCircuit,
    saeComboId,
    buildDictionaryName,
    loadCircuit,
  ]);

  return (
    <div className="space-y-6">
      <SaeComboLoader />
      
      <Card>
        <CardHeader>
          <CardTitle>Functional Microcircuit Visualization</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Circuit Selection */}
          <div className="flex items-center space-x-4">
            <Label htmlFor="circuit-select" className="w-32">Select Circuit:</Label>
            <Select
              value={selectedCircuitId || ''}
              onValueChange={(value) => {
                if (value) {
                  loadCircuit(value);
                } else {
                  setSelectedCircuit(null);
                  setSelectedCircuitId(null);
                  setNodes([]);
                  setEdges([]);
                }
              }}
            >
              <SelectTrigger id="circuit-select" className="flex-1">
                <SelectValue placeholder="Select a circuit interpretation..." />
              </SelectTrigger>
              <SelectContent>
                {loadingCircuits ? (
                  <div className="p-4 text-center">
                    <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                  </div>
                ) : circuits.length === 0 ? (
                  <div className="p-4 text-center text-gray-500">No circuits found</div>
                ) : (
                  circuits.map((circuit) => (
                    <SelectItem key={circuit.circuit_id} value={circuit.circuit_id}>
                      {circuit.circuit_interpretation || circuit.circuit_id}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            <Button onClick={loadCircuits} variant="outline" size="sm">
              Refresh
            </Button>
          </div>

          {error && (
            <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}

          {selectedCircuit && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm font-medium text-blue-900">Circuit Interpretation:</p>
              <p className="text-sm text-blue-700">{selectedCircuit.circuit_interpretation}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Graph Visualization */}
      {selectedCircuit && (
        <Card>
          <CardHeader>
            <CardTitle>Circuit Graph</CardTitle>
            <p className="text-sm text-gray-600">
              Drag nodes to create edges. Click nodes to view details. Click edges to edit weight.
              <br />
              <strong>For node interaction analysis:</strong> Click first node, then Ctrl+click second node.
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-[600px] border rounded-lg">
              <ReactFlow
                nodes={nodes.map(node => ({
                  ...node,
                  selected: node.id === selectedNodeId || 
                           node.id === selectedNodeId2 || 
                           selectedSteeringNodeIds.has(node.id) ||
                           selectedTargetNodeIds.has(node.id),
                  data: {
                    ...node.data,
                    isSecondSelection: node.id === selectedNodeId2,
                    isSteeringNode: selectedSteeringNodeIds.has(node.id),
                    isTargetNode: selectedTargetNodeIds.has(node.id),
                  },
                }))}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={onNodeClick}
                onNodeMouseEnter={onNodeMouseEnter}
                onNodeMouseLeave={onNodeMouseLeave}
                onNodeDragStart={onNodeDragStart}
                onNodeDragStop={onNodeDragStop}
                onEdgeClick={onEdgeClick}
                nodeTypes={nodeTypes}
                fitView
                attributionPosition="bottom-left"
              >
                <Background />
                <Controls />
                <MiniMap />
              </ReactFlow>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Global Weight Between Two Features */}
          <Card>
            <CardHeader>
              <CardTitle>Global Weight Analysis (Between Two Features)</CardTitle>
              <p className="text-sm text-gray-600 mt-2">
                点击节点选择第一个特征，按住 Ctrl/Cmd 点击选择第二个特征
                <br />
                <em>选择两个节点后，还可以进行节点交互分析（查看因果影响）</em>
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Selected Features Display */}
                <div className="space-y-2">
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded">
                    <p className="text-xs font-medium text-blue-900">Feature A (第一个):</p>
                    <p className="text-sm text-blue-700">
                      {selectedNode.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L{selectedNode.data?.layer} #{selectedNode.data?.featureIndex}
                    </p>
                  </div>
                  {selectedNode2 ? (
                    <div className="p-2 bg-green-50 border border-green-200 rounded">
                      <p className="text-xs font-medium text-green-900">Feature B (第二个):</p>
                      <p className="text-sm text-green-700">
                        {selectedNode2.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L{selectedNode2.data?.layer} #{selectedNode2.data?.featureIndex}
                      </p>
                    </div>
                  ) : (
                    <div className="p-2 bg-gray-50 border border-gray-200 rounded">
                      <p className="text-xs text-gray-500">按住 Ctrl/Cmd 点击另一个节点选择 Feature B</p>
                    </div>
                  )}
                </div>

                {/* Calculate Button */}
                <Button
                  onClick={() => {
                    if (!selectedNode2) {
                      alert('请先选择第二个特征（按住 Ctrl/Cmd 点击节点）');
                      return;
                    }
                    const nodeDataA = selectedNode.data;
                    const nodeDataB = selectedNode2.data;
                    if (nodeDataA && nodeDataB) {
                      fetchGlobalWeightBetweenFeatures(
                        nodeDataA.layer,
                        nodeDataA.featureIndex,
                        nodeDataA.featureType,
                        nodeDataB.layer,
                        nodeDataB.featureIndex,
                        nodeDataB.featureType
                      );
                    }
                  }}
                  disabled={loadingGlobalWeights || !selectedNode2}
                  size="sm"
                  className="w-full"
                >
                  {loadingGlobalWeights ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      {loadingMessage || '计算中...'}
                    </>
                  ) : (
                    '计算两个特征之间的全局权重'
                  )}
                </Button>

                {loadingMessage && (
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                    {loadingMessage}
                  </div>
                )}

                {/* Results Display */}
                {betweenFeaturesResult && (
                  <div className="space-y-3 mt-4">
                    <div className="p-3 bg-purple-50 border border-purple-200 rounded">
                      <p className="text-sm font-medium text-purple-900 mb-2">分析结果:</p>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium">Feature A → Feature B:</span>
                          {betweenFeaturesResult.rankBInA !== null ? (
                            <span className="ml-2">
                              在 Feature A 的 <strong>features_out</strong> 中排名 <strong className="text-purple-700">#{betweenFeaturesResult.rankBInA}</strong>
                              {betweenFeaturesResult.weightBInA !== null && (
                                <span className="ml-2 text-gray-600">
                                  (权重: {betweenFeaturesResult.weightBInA.toFixed(6)})
                                </span>
                              )}
                            </span>
                          ) : (
                            <span className="ml-2 text-gray-500">未在 Top 100 中找到</span>
                          )}
                        </div>
                        <div>
                          <span className="font-medium">Feature B ← Feature A:</span>
                          {betweenFeaturesResult.rankAInB !== null ? (
                            <span className="ml-2">
                              在 Feature B 的 <strong>features_in</strong> 中排名 <strong className="text-purple-700">#{betweenFeaturesResult.rankAInB}</strong>
                              {betweenFeaturesResult.weightAInB !== null && (
                                <span className="ml-2 text-gray-600">
                                  (权重: {betweenFeaturesResult.weightAInB.toFixed(6)})
                                </span>
                              )}
                            </span>
                          ) : (
                            <span className="ml-2 text-gray-500">未在 Top 100 中找到</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Top Activations */}
          <Card>
            <CardHeader>
              <CardTitle>Top Activations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button
                  onClick={() => {
                    const nodeData = selectedNode.data;
                    if (nodeData) {
                      fetchTopActivations(
                        nodeData.layer,
                        nodeData.featureIndex,
                        nodeData.featureType
                      );
                    }
                  }}
                  disabled={loadingTopActivations}
                  size="sm"
                >
                  {loadingTopActivations ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      Loading...
                    </>
                  ) : (
                    'Load Top Activations'
                  )}
                </Button>
                {topActivations.length > 0 ? (
                  <div className="grid grid-cols-2 gap-4">
                    {topActivations.map((sample, index) => (
                      <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                        <div className="text-center mb-2">
                          <div className="text-sm font-medium text-gray-700">Top #{index + 1}</div>
                          <div className="text-xs text-gray-500">
                            Activation: {sample.activationStrength.toFixed(3)}
                          </div>
                          {sample.fen && (
                            <div className="text-xs text-gray-400 mt-1 truncate" title={sample.fen}>
                              {sample.fen.substring(0, 30)}...
                            </div>
                          )}
                        </div>
                        {sample.fen ? (
                          <ChessBoard
                            fen={sample.fen}
                            size="small"
                            showCoordinates={false}
                            activations={sample.activations}
                            zPatternIndices={sample.zPatternIndices}
                            zPatternValues={sample.zPatternValues}
                            flip_activation={Boolean(sample.fen && sample.fen.split(' ')[1] === 'b')}
                            autoFlipWhenBlack={true}
                          />
                        ) : (
                          <div className="text-center text-gray-400 text-sm py-4">
                            无效的FEN字符串
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-4">
                    {loadingTopActivations ? '加载中...' : '未找到包含棋盘的激活样本'}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Edge Editor Panel */}
      {editingEdge && (
        <Card>
          <CardHeader>
            <CardTitle>Edit Edge Weight</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <Label htmlFor="edge-weight">Weight:</Label>
              <Input
                id="edge-weight"
                type="number"
                step="0.001"
                value={edgeWeightInput}
                onChange={(e) => setEdgeWeightInput(e.target.value)}
                className="w-32"
              />
              <Button onClick={handleSaveEdgeWeight} size="sm">
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
              <Button onClick={() => setEditingEdge(null)} variant="outline" size="sm">
                <X className="w-4 h-4 mr-2" />
                Cancel
              </Button>
              <Button
                onClick={() => editingEdge && handleDeleteEdge(editingEdge.id)}
                variant="destructive"
                size="sm"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Level Editor Panel */}
      {editingLevel && (
        <Card>
          <CardHeader>
            <CardTitle>Edit Feature Level</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-4">
              <Label htmlFor="feature-level">Level:</Label>
              <Input
                id="feature-level"
                type="number"
                value={levelInput}
                onChange={(e) => setLevelInput(e.target.value)}
                className="w-32"
              />
              <Button onClick={handleSaveLevel} size="sm">
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
              <Button onClick={() => setEditingLevel(null)} variant="outline" size="sm">
                <X className="w-4 h-4 mr-2" />
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Node Actions Panel */}
      {selectedNode && (
        <Card>
          <CardHeader>
            <CardTitle>Node Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col space-y-3">
              <div className="flex items-center space-x-4">
                <Button
                  onClick={() => {
                    const nodeData = selectedNode.data;
                    if (nodeData) {
                      setEditingLevel({
                        featureId: nodeData.featureId,
                        level: nodeData.level || nodeData.layer,
                      });
                      setLevelInput((nodeData.level || nodeData.layer).toString());
                    }
                  }}
                  variant="outline"
                  size="sm"
                >
                  <Edit2 className="w-4 h-4 mr-2" />
                  Edit Level
                </Button>
                <Button
                  onClick={() => {
                    const nodeData = selectedNode.data;
                    if (nodeData && selectedCircuit) {
                      // Convert featureType to componentType
                      const componentType: "attn" | "mlp" = nodeData.featureType === 'lorsa' ? 'attn' : 'mlp';
                      handleRemoveFeatureFromCircuit(
                        nodeData.featureIndex,
                        nodeData.layer,
                        componentType
                      );
                    }
                  }}
                  variant="destructive"
                  size="sm"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  从Circuit删除
                </Button>
              </div>

              <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded border">
                💡 <strong>提示：</strong>选择两个节点来进行交互分析：
                <br />1. 点击选择第一个节点（steering节点）
                <br />2. 按住 Ctrl/Cmd 点击选择第二个节点（target节点）
                <br />3. 然后可以使用"节点交互分析"功能
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Feature Selection from FEN */}
      <Card>
        <CardHeader>
          <CardTitle>从 FEN 添加 Feature 到 Circuit</CardTitle>
          <p className="text-sm text-gray-600">
            选择 FEN、层、位置和组件类型，然后选择要添加的 feature
            {selectedCircuit ? `（将添加到当前 circuit: ${selectedCircuit.circuit_interpretation}）` : '（将创建新 circuit）'}
          </p>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* FEN and position configuration */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="fen-for-feature">FEN 字符串</Label>
                <Input
                  id="fen-for-feature"
                  value={fenForFeatureSelection}
                  onChange={(e) => setFenForFeatureSelection(e.target.value)}
                  placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="layer-for-feature">层 (Layer)</Label>
                <Input
                  id="layer-for-feature"
                  type="number"
                  min="0"
                  max="14"
                  value={layerForFeatureSelection}
                  onChange={(e) => setLayerForFeatureSelection(parseInt(e.target.value) || 0)}
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="position-for-feature">位置 (Position)</Label>
                <Input
                  id="position-for-feature"
                  type="number"
                  min="0"
                  max="63"
                  value={positionForFeatureSelection}
                  onChange={(e) => setPositionForFeatureSelection(parseInt(e.target.value) || 0)}
                  className="bg-white"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="component-type-for-feature">组件类型</Label>
                <select
                  id="component-type-for-feature"
                  value={componentTypeForFeatureSelection}
                  onChange={(e) => setComponentTypeForFeatureSelection(e.target.value as "attn" | "mlp")}
                  className="w-full p-2 border rounded bg-white"
                >
                  <option value="attn">Attention (LoRSA)</option>
                  <option value="mlp">MLP (Transcoder)</option>
                </select>
              </div>
            </div>

            {/* Auto-create edges option */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="auto-create-edges"
                checked={autoCreateEdges}
                onChange={(e) => setAutoCreateEdges(e.target.checked)}
                className="w-4 h-4"
              />
              <Label htmlFor="auto-create-edges" className="text-sm">
                自动为新添加的 feature 创建边（连接到现有 features，权重为 0）
              </Label>
            </div>

            {/* GetFeatureFromFen component */}
            <div className="border rounded-lg p-4">
              <GetFeatureFromFen
                fen={fenForFeatureSelection}
                layer={layerForFeatureSelection}
                position={positionForFeatureSelection}
                componentType={componentTypeForFeatureSelection}
                modelName="lc0/BT4-1024x15x32h"
                saeComboId={saeComboId}
                actionTypes={["add_to_steer"]}
                actionButtonLabels={{
                  add_to_steer: selectedCircuit ? "加入Circuit" : "创建Circuit",
                }}
                showTopActivations={false}
                showFenActivations={false}
                onFeatureAction={(action) => {
                  if (action.type === "add_to_steer") {
                    handleAddFeatureToCircuit(action.featureIndex);
                  }
                }}
                className="border-0 shadow-none" // Remove card styling since it's inside another card
              />
              <div className="mt-2 text-xs text-gray-500">
                💡 点击 feature 列表中的"{selectedCircuit ? "加入Circuit" : "创建Circuit"}"按钮将 feature {selectedCircuit ? "添加到当前 circuit" : "创建新 circuit"}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Node Interaction Analysis Panel */}
      {(selectedNodeId && selectedNodeId2) || (selectedSteeringNodeIds.size > 0 && selectedTargetNodeIds.size > 0) ? (
        <Card>
          <CardHeader>
            <CardTitle>Node Interaction Analysis</CardTitle>
            <p className="text-sm text-gray-600">
              Analyze how steering nodes affect target nodes' activations
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Selection Mode Toggle */}
              <div className="flex items-center space-x-4 p-3 bg-gray-50 rounded border">
                <Label className="text-sm font-medium">选择模式:</Label>
                <Button
                  onClick={() => {
                    setNodeSelectionMode("steering");
                    setSelectedSteeringNodeIds(new Set());
                    setSelectedTargetNodeIds(new Set());
                    setSteeringNodePositions(new Map());
                    setTargetNodePositions(new Map());
                  }}
                  variant={nodeSelectionMode === "steering" ? "default" : "outline"}
                  size="sm"
                >
                  选择 Steering Nodes
                </Button>
                <Button
                  onClick={() => {
                    setNodeSelectionMode("target");
                    setSelectedSteeringNodeIds(new Set());
                    setSelectedTargetNodeIds(new Set());
                    setSteeringNodePositions(new Map());
                    setTargetNodePositions(new Map());
                  }}
                  variant={nodeSelectionMode === "target" ? "default" : "outline"}
                  size="sm"
                >
                  选择 Target Nodes
                </Button>
                <Button
                  onClick={() => {
                    setNodeSelectionMode(null);
                    setSelectedSteeringNodeIds(new Set());
                    setSelectedTargetNodeIds(new Set());
                    setSteeringNodePositions(new Map());
                    setTargetNodePositions(new Map());
                  }}
                  variant="outline"
                  size="sm"
                >
                  取消选择模式
                </Button>
              </div>

              {/* FEN Display with ChessBoard */}
              <div className="space-y-2">
                <Label htmlFor="interaction-fen" className="text-sm font-medium">FEN (用于前向传播):</Label>
                <Input
                  id="interaction-fen"
                  type="text"
                  value={interactionFen}
                  onChange={(e) => {
                    setInteractionFen(e.target.value);
                    // Clear activations when FEN changes
                    setNodeActivations(new Map());
                  }}
                  placeholder={selectedCircuit ? `默认: ${(selectedCircuit as any).original_fen || '无'}` : '输入 FEN 字符串'}
                  className="w-full"
                />
                {selectedCircuit && (selectedCircuit as any).original_fen && (
                  <Button
                    onClick={() => {
                      setInteractionFen((selectedCircuit as any).original_fen || '');
                      setNodeActivations(new Map());
                    }}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                  >
                    使用 Circuit 的 FEN
                  </Button>
                )}
                {interactionFen && (
                  <div className="border rounded-lg p-2 bg-white">
                    <ChessBoard
                      fen={interactionFen}
                      size="small"
                      showCoordinates={true}
                    />
                  </div>
                )}
              </div>

              {/* Selected Nodes Display */}
              <div className="space-y-2">
                <div className="p-2 bg-blue-50 border border-blue-200 rounded">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-blue-900">Steering Nodes ({selectedSteeringNodeIds.size > 0 ? selectedSteeringNodeIds.size : (selectedNodeId ? 1 : 0)}):</p>
                    {selectedSteeringNodeIds.size > 0 && (
                      <Button
                        onClick={() => {
                          setSelectedSteeringNodeIds(new Set());
                          setSteeringNodePositions(new Map());
                        }}
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs"
                      >
                        清空
                      </Button>
                    )}
                  </div>
                  {selectedSteeringNodeIds.size > 0 ? (
                    <div className="mt-2 space-y-2">
                      {Array.from(selectedSteeringNodeIds).map(id => {
                        const node = nodes.find(n => n.id === id);
                        const currentPos = steeringNodePositions.get(id);
                        const activation = nodeActivations.get(id);
                        const isLoading = loadingActivations.has(id);
                        return node ? (
                          <div key={id} className="flex flex-col gap-2 text-sm text-blue-700 bg-white p-2 rounded border border-blue-200">
                            <div className="flex items-center justify-between">
                              <span className="font-medium">
                                {node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L{node.data?.layer} #{node.data?.featureIndex}
                              </span>
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedSteeringNodeIds(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(id);
                                    return newSet;
                                  });
                                  setSteeringNodePositions(prev => {
                                    const newMap = new Map(prev);
                                    newMap.delete(id);
                                    return newMap;
                                  });
                                  setNodeActivations(prev => {
                                    const newMap = new Map(prev);
                                    newMap.delete(id);
                                    return newMap;
                                  });
                                }}
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                              >
                                <X className="w-3 h-3" />
                              </Button>
                            </div>
                            <div className="flex items-center gap-2">
                              <Label htmlFor={`steering-pos-${id}`} className="text-xs">Pos:</Label>
                              <Input
                                id={`steering-pos-${id}`}
                                type="number"
                                min="0"
                                max="63"
                                value={currentPos ?? ''}
                                onChange={(e) => {
                                  const newPos = parseInt(e.target.value);
                                  if (!isNaN(newPos)) {
                                    setSteeringNodePositions(prev => {
                                      const newMap = new Map(prev);
                                      newMap.set(id, newPos);
                                      return newMap;
                                    });
                                    // Clear activation when position changes
                                    setNodeActivations(prev => {
                                      const newMap = new Map(prev);
                                      newMap.delete(id);
                                      return newMap;
                                    });
                                  }
                                }}
                                placeholder="0-63"
                                className="w-16 h-7 text-xs"
                              />
                              {currentPos !== undefined && interactionFen && (
                                <Button
                                  onClick={() => fetchNodeActivation(id, currentPos)}
                                  disabled={isLoading}
                                  variant="outline"
                                  size="sm"
                                  className="h-7 text-xs"
                                >
                                  {isLoading ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                  ) : (
                                    '获取激活值'
                                  )}
                                </Button>
                              )}
                              {activation !== undefined && (
                                <span className={`text-xs font-medium ${activation === 0 ? 'text-red-600' : 'text-green-600'}`}>
                                  激活值: {activation.toFixed(4)}
                                </span>
                              )}
                            </div>
                          </div>
                        ) : null;
                      })}
                    </div>
                  ) : selectedNodeId ? (
                    <div className="mt-2 p-2 bg-white rounded border border-blue-200">
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-blue-700">
                    {(() => {
                      const node = nodes.find(n => n.id === selectedNodeId);
                      return node ? `${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}` : 'Unknown';
                    })()}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Label htmlFor="steering-pos-single" className="text-xs">Pos:</Label>
                          <Input
                            id="steering-pos-single"
                            type="number"
                            min="0"
                            max="63"
                            value={steeringNodePositions.get(selectedNodeId) ?? ''}
                            onChange={(e) => {
                              const newPos = parseInt(e.target.value);
                              if (!isNaN(newPos)) {
                                setSteeringNodePositions(prev => {
                                  const newMap = new Map(prev);
                                  newMap.set(selectedNodeId, newPos);
                                  return newMap;
                                });
                                setNodeActivations(prev => {
                                  const newMap = new Map(prev);
                                  newMap.delete(selectedNodeId);
                                  return newMap;
                                });
                              }
                            }}
                            placeholder="0-63"
                            className="w-16 h-7 text-xs"
                          />
                          {steeringNodePositions.get(selectedNodeId) !== undefined && interactionFen && (
                            <Button
                              onClick={() => fetchNodeActivation(selectedNodeId, steeringNodePositions.get(selectedNodeId)!)}
                              disabled={loadingActivations.has(selectedNodeId)}
                              variant="outline"
                              size="sm"
                              className="h-7 text-xs"
                            >
                              {loadingActivations.has(selectedNodeId) ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                '获取激活值'
                              )}
                            </Button>
                          )}
                          {nodeActivations.get(selectedNodeId) !== undefined && (
                            <span className={`text-xs font-medium ${nodeActivations.get(selectedNodeId) === 0 ? 'text-red-600' : 'text-green-600'}`}>
                              激活值: {nodeActivations.get(selectedNodeId)!.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">点击"选择 Steering Nodes"按钮，然后点击图中的节点进行选择（再次点击可取消选择）</p>
                  )}
                </div>
                <div className="p-2 bg-green-50 border border-green-200 rounded">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-xs font-medium text-green-900">Target Nodes ({selectedTargetNodeIds.size > 0 ? selectedTargetNodeIds.size : (selectedNodeId2 ? 1 : 0)}):</p>
                    {selectedTargetNodeIds.size > 0 && (
                      <Button
                        onClick={() => {
                          setSelectedTargetNodeIds(new Set());
                          setTargetNodePositions(new Map());
                        }}
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs"
                      >
                        清空
                      </Button>
                    )}
                  </div>
                  {selectedTargetNodeIds.size > 0 ? (
                    <div className="mt-2 space-y-2">
                      {Array.from(selectedTargetNodeIds).map(id => {
                        const node = nodes.find(n => n.id === id);
                        const currentPos = targetNodePositions.get(id);
                        const activation = nodeActivations.get(id);
                        const isLoading = loadingActivations.has(id);
                        return node ? (
                          <div key={id} className="flex flex-col gap-2 text-sm text-green-700 bg-white p-2 rounded border border-green-200">
                            <div className="flex items-center justify-between">
                              <span className="font-medium">
                                {node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L{node.data?.layer} #{node.data?.featureIndex}
                              </span>
                              <Button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedTargetNodeIds(prev => {
                                    const newSet = new Set(prev);
                                    newSet.delete(id);
                                    return newSet;
                                  });
                                  setTargetNodePositions(prev => {
                                    const newMap = new Map(prev);
                                    newMap.delete(id);
                                    return newMap;
                                  });
                                  setNodeActivations(prev => {
                                    const newMap = new Map(prev);
                                    newMap.delete(id);
                                    return newMap;
                                  });
                                }}
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                              >
                                <X className="w-3 h-3" />
                              </Button>
                            </div>
                            <div className="flex items-center gap-2">
                              <Label htmlFor={`target-pos-${id}`} className="text-xs">Pos:</Label>
                              <Input
                                id={`target-pos-${id}`}
                                type="number"
                                min="0"
                                max="63"
                                value={currentPos ?? ''}
                                onChange={(e) => {
                                  const newPos = parseInt(e.target.value);
                                  if (!isNaN(newPos)) {
                                    setTargetNodePositions(prev => {
                                      const newMap = new Map(prev);
                                      newMap.set(id, newPos);
                                      return newMap;
                                    });
                                    // Clear activation when position changes
                                    setNodeActivations(prev => {
                                      const newMap = new Map(prev);
                                      newMap.delete(id);
                                      return newMap;
                                    });
                                  }
                                }}
                                placeholder="0-63"
                                className="w-16 h-7 text-xs"
                              />
                              {currentPos !== undefined && interactionFen && (
                                <Button
                                  onClick={() => fetchNodeActivation(id, currentPos)}
                                  disabled={isLoading}
                                  variant="outline"
                                  size="sm"
                                  className="h-7 text-xs"
                                >
                                  {isLoading ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                  ) : (
                                    '获取激活值'
                                  )}
                                </Button>
                              )}
                              {activation !== undefined && (
                                <span className={`text-xs font-medium ${activation === 0 ? 'text-red-600' : 'text-green-600'}`}>
                                  激活值: {activation.toFixed(4)}
                                </span>
                              )}
                            </div>
                          </div>
                        ) : null;
                      })}
                    </div>
                  ) : selectedNodeId2 ? (
                    <div className="mt-2 p-2 bg-white rounded border border-green-200">
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-green-700">
                    {(() => {
                      const node = nodes.find(n => n.id === selectedNodeId2);
                      return node ? `${node.data?.featureType === 'lorsa' ? 'LoRSA' : 'TC'} L${node.data?.layer} #${node.data?.featureIndex}` : 'Unknown';
                    })()}
                          </span>
                </div>
                        <div className="flex items-center gap-2">
                          <Label htmlFor="target-pos-single" className="text-xs">Pos:</Label>
                          <Input
                            id="target-pos-single"
                            type="number"
                            min="0"
                            max="63"
                            value={targetNodePositions.get(selectedNodeId2) ?? ''}
                            onChange={(e) => {
                              const newPos = parseInt(e.target.value);
                              if (!isNaN(newPos)) {
                                setTargetNodePositions(prev => {
                                  const newMap = new Map(prev);
                                  newMap.set(selectedNodeId2, newPos);
                                  return newMap;
                                });
                                setNodeActivations(prev => {
                                  const newMap = new Map(prev);
                                  newMap.delete(selectedNodeId2);
                                  return newMap;
                                });
                              }
                            }}
                            placeholder="0-63"
                            className="w-16 h-7 text-xs"
                          />
                          {targetNodePositions.get(selectedNodeId2) !== undefined && interactionFen && (
                            <Button
                              onClick={() => fetchNodeActivation(selectedNodeId2, targetNodePositions.get(selectedNodeId2)!)}
                              disabled={loadingActivations.has(selectedNodeId2)}
                              variant="outline"
                              size="sm"
                              className="h-7 text-xs"
                            >
                              {loadingActivations.has(selectedNodeId2) ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : (
                                '获取激活值'
                              )}
                            </Button>
                          )}
                          {nodeActivations.get(selectedNodeId2) !== undefined && (
                            <span className={`text-xs font-medium ${nodeActivations.get(selectedNodeId2) === 0 ? 'text-red-600' : 'text-green-600'}`}>
                              激活值: {nodeActivations.get(selectedNodeId2)!.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">点击"选择 Target Nodes"按钮，然后点击图中的节点进行选择（再次点击可取消选择）</p>
                  )}
                </div>
              </div>

              {/* FEN Input */}
              <div className="space-y-2">
                <Label htmlFor="interaction-fen" className="text-sm font-medium">FEN (用于前向传播，留空则使用 Circuit 的 FEN):</Label>
                <Input
                  id="interaction-fen"
                  type="text"
                  value={interactionFen}
                  onChange={(e) => setInteractionFen(e.target.value)}
                  placeholder={selectedCircuit ? `默认: ${(selectedCircuit as any).original_fen || '无'}` : '输入 FEN 字符串'}
                  className="w-full"
                />
                {selectedCircuit && (selectedCircuit as any).original_fen && (
                  <Button
                    onClick={() => setInteractionFen((selectedCircuit as any).original_fen || '')}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                  >
                    使用 Circuit 的 FEN
                  </Button>
                )}
              </div>

              {/* Steering Scale Input */}
              <div className="flex items-center space-x-2">
                <Label htmlFor="steering-scale" className="text-sm">Steering Scale:</Label>
                <Input
                  id="steering-scale"
                  type="number"
                  step="0.1"
                  value={steeringScale}
                  onChange={(e) => setSteeringScale(e.target.value)}
                  className="w-20"
                />
              </div>

              {/* Analyze Button */}
              <Button
                onClick={analyzeNodeInteraction}
                disabled={analyzingInteraction || (selectedSteeringNodeIds.size === 0 && !selectedNodeId) || (selectedTargetNodeIds.size === 0 && !selectedNodeId2)}
                className="w-full"
              >
                {analyzingInteraction ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Analyzing Interaction...
                  </>
                ) : (
                  'Analyze Node Interaction'
                )}
              </Button>

              {/* Results Display */}
              {interactionResult && (
                <div className="space-y-3 mt-4">
                  <div className="p-3 bg-purple-50 border border-purple-200 rounded">
                    <p className="text-sm font-medium text-purple-900 mb-2">Interaction Results:</p>
                    <div className="space-y-2 text-sm mb-3">
                      <div>
                        <span className="font-medium">Steering Scale:</span>
                        <span className="ml-2">{interactionResult.steering_scale}</span>
                      </div>
                      <div>
                        <span className="font-medium">Steering Nodes Count:</span>
                        <span className="ml-2">{interactionResult.steering_nodes_count}</span>
                      </div>
                    </div>
                    
                    {/* Target Nodes Results */}
                    {interactionResult.target_nodes && Array.isArray(interactionResult.target_nodes) ? (
                      <div className="space-y-3 mt-3">
                        <p className="text-xs font-medium text-purple-900 mb-2">Target Nodes Results:</p>
                        {interactionResult.target_nodes.map((targetResult: any, idx: number) => (
                          <div key={idx} className="p-2 bg-white border border-purple-200 rounded">
                            <p className="text-xs font-medium text-purple-800 mb-1">{targetResult.target_node}</p>
                            <div className="space-y-1 text-xs">
                              <div>
                                <span className="font-medium">Original Activation:</span>
                                <span className="ml-2">{targetResult.original_activation?.toFixed(6)}</span>
                              </div>
                              <div>
                                <span className="font-medium">Modified Activation:</span>
                                <span className="ml-2">{targetResult.modified_activation?.toFixed(6)}</span>
                              </div>
                              <div>
                                <span className="font-medium">Activation Change:</span>
                                <span className={`ml-2 ${targetResult.activation_change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {targetResult.activation_change?.toFixed(6)}
                                </span>
                              </div>
                              <div>
                                <span className="font-medium">Activation Ratio:</span>
                                <span className={`ml-2 ${targetResult.activation_ratio > 1 ? 'text-green-600' : targetResult.activation_ratio < 1 ? 'text-red-600' : 'text-gray-600'}`}>
                                  {targetResult.activation_ratio === null ? 'N/A' :
                                   targetResult.activation_ratio === Number.POSITIVE_INFINITY ? '∞' :
                                   targetResult.activation_ratio?.toFixed(4) + 'x'}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      // Backward compatibility: single target node result
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="font-medium">Target Node:</span>
                        <span className="ml-2">{interactionResult.target_node}</span>
                      </div>
                      <div>
                        <span className="font-medium">Original Activation:</span>
                        <span className="ml-2">{interactionResult.original_activation?.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="font-medium">Modified Activation:</span>
                        <span className="ml-2">{interactionResult.modified_activation?.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="font-medium">Activation Change:</span>
                        <span className={`ml-2 ${interactionResult.activation_change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {interactionResult.activation_change?.toFixed(6)}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Activation Ratio:</span>
                        <span className={`ml-2 ${interactionResult.activation_ratio > 1 ? 'text-green-600' : interactionResult.activation_ratio < 1 ? 'text-red-600' : 'text-gray-600'}`}>
                          {interactionResult.activation_ratio === null ? 'N/A' :
                           interactionResult.activation_ratio === Number.POSITIVE_INFINITY ? '∞' :
                           interactionResult.activation_ratio?.toFixed(4) + 'x'}
                        </span>
                      </div>
                      </div>
                    )}
                  </div>

                  {/* Steering Details */}
                  {interactionResult.steering_details && interactionResult.steering_details.length > 0 && (
                    <div className="p-3 bg-gray-50 border border-gray-200 rounded">
                      <p className="text-sm font-medium text-gray-900 mb-2">Steering Details:</p>
                      <div className="space-y-1 text-xs">
                        {interactionResult.steering_details.map((detail: any, idx: number) => (
                          <div key={idx} className="flex justify-between">
                            <span>{detail.node}</span>
                            <span className={detail.found ? 'text-green-600' : 'text-red-600'}>
                              {detail.found ? `✓ ${detail.activation_value?.toFixed(4)}` : '✗ Not found'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};
