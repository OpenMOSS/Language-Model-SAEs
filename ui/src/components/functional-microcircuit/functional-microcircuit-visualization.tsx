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
import {
  listCircuitAnnotations,
  getCircuitAnnotation,
  CircuitAnnotation,
  addEdgeToCircuit,
  removeEdgeFromCircuit,
  updateEdgeWeight,
  setFeatureLevel,
} from '@/utils/api';

interface GlobalWeightData {
  feature_type: string;
  layer_idx: number;
  feature_idx: number;
  features_in: Array<{ name: string; clerp?: string; rank?: number }>;
  features_out: Array<{ name: string; clerp?: string; rank?: number }>;
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
const FeatureNode = ({ data }: { data: FeatureNodeData }) => {
  return (
    <div className="px-4 py-3 bg-white border-2 border-gray-300 rounded-lg shadow-md min-w-[200px]">
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
  const [draggingFrom, setDraggingFrom] = useState<string | null>(null);

  // Global Weight state
  const [loadingGlobalWeights, setLoadingGlobalWeights] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);

  // Top Activation state
  const [topActivations, setTopActivations] = useState<TopActivationData[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);

  // Edge editing state
  const [editingEdge, setEditingEdge] = useState<{ id: string; weight: number } | null>(null);
  const [edgeWeightInput, setEdgeWeightInput] = useState<string>('0');

  // Level editing state
  const [editingLevel, setEditingLevel] = useState<{ featureId: string; level: number } | null>(null);
  const [levelInput, setLevelInput] = useState<string>('0');

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
    setSelectedNodeId(node.id === selectedNodeId ? null : node.id);
  }, [selectedNodeId]);

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

  // Fetch Global Weight for a feature
  const fetchGlobalWeight = useCallback(async (layer: number, featureIndex: number, featureType: string) => {
    if (!saeComboId) {
      alert('请先选择 SAE 组合');
      return;
    }

    setLoadingGlobalWeights(true);
    setLoadingMessage(null);
    try {
      // Preload models first
      await preloadModels(saeComboId);

      const params = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        feature_type: featureType === 'lorsa' ? 'lorsa' : 'tc',
        layer_idx: layer.toString(),
        feature_idx: featureIndex.toString(),
        k: '100',
        sae_combo_id: saeComboId,
      });

      let response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
      
      // If 503 (not loaded), try preloading again
      if (response.status === 503) {
        setLoadingMessage('模型未加载，正在自动加载...');
        await preloadModels(saeComboId);
        setLoadingMessage('正在计算全局权重...');
        response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/global_weight?${params.toString()}`);
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: GlobalWeightData = await response.json();
      
      // Store global weight for the feature
      // For now, we'll use a simple metric - could be improved
      const weight = data.features_in.length + data.features_out.length;

      // Update node data with global weight
      setNodes((nds) => nds.map(node => {
        const nodeData = node.data;
        if (nodeData && nodeData.layer === layer && 
            nodeData.featureIndex === featureIndex &&
            nodeData.featureType === featureType) {
          return {
            ...node,
            data: {
              ...nodeData,
              globalWeight: weight,
            },
          };
        }
        return node;
      }));
    } catch (err) {
      console.error('Failed to fetch global weight:', err);
      alert('获取全局权重失败: ' + (err instanceof Error ? err.message : '未知错误'));
    } finally {
      setLoadingGlobalWeights(false);
      setLoadingMessage(null);
    }
  }, [saeComboId, setNodes, preloadModels]);

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

  // Get selected node
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodes.find(n => n.id === selectedNodeId);
  }, [selectedNodeId, nodes]);

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
            </p>
          </CardHeader>
          <CardContent>
            <div className="h-[600px] border rounded-lg">
              <ReactFlow
                nodes={nodes}
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
          {/* Global Weight */}
          <Card>
            <CardHeader>
              <CardTitle>Global Weight Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                <Button
                  onClick={() => {
                    const nodeData = selectedNode.data;
                    if (nodeData) {
                      fetchGlobalWeight(
                        nodeData.layer,
                        nodeData.featureIndex,
                        nodeData.featureType
                      );
                    }
                  }}
                  disabled={loadingGlobalWeights}
                  size="sm"
                >
                  {loadingGlobalWeights ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      {loadingMessage || 'Loading...'}
                    </>
                  ) : (
                    'Load Global Weight'
                  )}
                </Button>
              </div>
              {loadingMessage && (
                <div className="p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
                  {loadingMessage}
                </div>
              )}
              {selectedNode.data?.globalWeight !== undefined && (
                <div className="p-3 bg-purple-50 border border-purple-200 rounded mt-2">
                  <p className="text-sm font-medium text-purple-900">Global Weight:</p>
                  <p className="text-lg font-bold text-purple-700">
                    {selectedNode.data.globalWeight.toFixed(4)}
                  </p>
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
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
