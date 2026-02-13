import { useState, useCallback, useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { useCircuitState } from "@/contexts/AppStateContext";
import { LinkGraphContainer } from "./link-graph-container";
import { NodeConnections } from "./node-connections";
import { CircuitJsonData } from "./link-graph/utils";
import { Node } from "./link-graph/types";
import { Feature } from "@/types/feature";
import { FeatureCard } from "@/components/feature/feature-card";
import { ChessBoard } from "@/components/chess/chess-board";
import React from "react";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
import { CustomFenInput } from "@/components/feature/custom-fen-input";
import { PosFeatureCard } from "@/components/feature/pos-feature-card";
import { CircuitInterpretationCard } from "./circuit-interpretation-card";
import { useCircuitFileUpload } from "@/hooks/useCircuitFileUpload";
import { FileUploadZone } from "./FileUploadZone";
import { useActivationData } from "@/hooks/useActivationData";
import { useFenExtraction } from "@/hooks/useFenExtraction";
import { useDictionaryName } from "@/hooks/useDictionaryName";
import { useCircuitStateReducer } from "@/hooks/useCircuitStateReducer";
import { mergeCircuitGraphs, UNIQUE_GRAPH_COLORS, POSITION_MAPPING_HIGHLIGHT_COLOR } from "@/utils/graphMergeUtils";
import { parseNodeId, normalizeZPattern } from "@/utils/activationUtils";
import { NodeActivationData } from "@/utils/activationUtils";

/**
 * Format probability for display
 * Intelligently formats probability values based on magnitude
 */
const formatProbability = (prob: number): string => {
  const percentage = prob * 100;
  if (Math.abs(percentage) >= 0.01) {
    return `${percentage.toFixed(2)}%`;
  } else if (Math.abs(percentage) >= 0.0001) {
    return `${percentage.toFixed(4)}%`;
  } else if (prob !== 0) {
    return `${percentage.toExponential(2)}%`;
  } else {
    return `${percentage.toFixed(2)}%`; // For true 0, display 0.00%
  }
};

export const CircuitVisualization = () => {
  const {
    circuitData: linkGraphData,
    isLoading,
    error,
    clickedId,
    hoveredId,
    pinnedIds,
    hiddenIds,
    setCircuitData: setLinkGraphData,
    setLoading,
    setError,
    setClickedId,
    setHoveredId,
    setPinnedIds,
    setHiddenIds,
  } = useCircuitState();

  // Use reducer for state management instead of multiple useState calls
  const circuitState = useCircuitStateReducer();
  const {
    state: {
      file,
      activation,
      display,
      featureDiffing,
      positionMapping,
      dense,
      sync,
      clerp,
      steering,
      posFeature,
    },
    actions,
  } = circuitState;

  // Local UI state (not managed by reducer)
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);

  // Extract FEN and moves using custom hook
  const fenExtraction = useFenExtraction({
    linkGraphData,
    originalCircuitJson: file.originalCircuitJson,
    multiOriginalJsons: file.multiOriginalJsons,
  });

  // Get dictionary names using custom hook
  const dictionaryName = useDictionaryName({ linkGraphData });

  // Get activation data using custom hook
  const activationDataHook = useActivationData({
    originalCircuitJson: file.originalCircuitJson,
    updateCounter: clerp.updateCounter,
  });

  // Create aliases for easier access to reducer state (for backward compatibility during refactoring)
  const originalCircuitJson = file.originalCircuitJson;
  const originalFileName = file.originalFileName;
  const updateCounter = clerp.updateCounter;
  const hasUnsavedChanges = file.hasUnsavedChanges;
  const saveHistory = file.saveHistory;
  const multiOriginalJsons = file.multiOriginalJsons;
  const topActivations = activation.topActivations;
  const loadingTopActivations = activation.loadingTopActivations;
  const tokenPredictions = activation.tokenPredictions;
  const loadingTokenPredictions = activation.loadingTokenPredictions;
  const allPositionsActivationData = activation.allPositionsActivationData;
  const loadingAllPositions = activation.loadingAllPositions;
  const multiGraphActivationData = activation.multiGraphActivationData;
  const loadingBackendZPattern = activation.loadingBackendZPattern;
  const backendZPatternByNode = activation.backendZPatternByNode;
  const showAllPositions = display.showAllPositions;
  const showSubgraph = display.showSubgraph;
  const subgraphData = display.subgraphData;
  const subgraphRootNodeId = display.subgraphRootNodeId;
  const showDiffingLogs = display.showDiffingLogs;
  const perturbedFen = featureDiffing.perturbedFen;
  const isComparingFens = featureDiffing.isComparingFens;
  const inactiveNodes = featureDiffing.inactiveNodes;
  const diffingLogs = featureDiffing.diffingLogs;
  const enablePositionMapping = positionMapping.enablePositionMapping;
  const positionMappingSelections = positionMapping.positionMappingSelections;
  const draftPositionMappingSelections = positionMapping.draftPositionMappingSelections;
  const positionMappingApplyNonce = positionMapping.positionMappingApplyNonce;
  const denseNodes = dense.denseNodes;
  const denseThreshold = dense.denseThreshold;
  const checkingDenseFeatures = dense.checkingDenseFeatures;
  const syncingToBackend = sync.syncingToBackend;
  const syncingFromBackend = sync.syncingFromBackend;
  const editingClerp = clerp.editingClerp;
  const isSaving = clerp.isSaving;
  const steeringScale = steering.steeringScale;
  const steeringScaleInput = steering.steeringScaleInput;
  const posFeatureLayer = posFeature.posFeatureLayer;
  const posFeaturePositions = posFeature.posFeaturePositions;
  const posFeatureComponentType = posFeature.posFeatureComponentType;

  // Create aliases for setter functions (for backward compatibility during refactoring)
  const setOriginalCircuitJson = actions.file.setOriginalJson;
  // setUpdateCounter needs to support functional updates: (prev) => prev + 1
  const setUpdateCounter = useCallback((updater: number | ((prev: number) => number)) => {
    if (typeof updater === 'function') {
      const newValue = updater(updateCounter);
      actions.clerp.setUpdateCounter(newValue);
    } else {
      actions.clerp.setUpdateCounter(updater);
    }
  }, [updateCounter, actions.clerp]);
  const setHasUnsavedChanges = actions.file.setHasUnsavedChanges;
  // setSaveHistory needs to support functional updates: (prev) => [...prev, newEntry]
  const setSaveHistory = useCallback((updater: string[] | ((prev: string[]) => string[])) => {
    if (typeof updater === 'function') {
      const newHistory = updater(saveHistory);
      // Clear and rebuild history
      actions.file.clearSaveHistory();
      newHistory.forEach(entry => actions.file.addSaveHistory(entry));
    } else {
      actions.file.clearSaveHistory();
      updater.forEach(entry => actions.file.addSaveHistory(entry));
    }
  }, [saveHistory, actions.file]);
  const setTopActivations = actions.activation.setTopActivations;
  const setLoadingTopActivations = actions.activation.setLoadingTopActivations;
  const setTokenPredictions = actions.activation.setTokenPredictions;
  const setLoadingTokenPredictions = actions.activation.setLoadingTokenPredictions;
  const setAllPositionsActivationData = actions.activation.setAllPositionsData;
  const setLoadingAllPositions = actions.activation.setLoadingAllPositions;
  const setMultiGraphActivationData = actions.activation.setMultiGraphData;
  const setLoadingBackendZPattern = actions.activation.setLoadingBackendZPattern;
  const setBackendZPatternByNode = actions.activation.setBackendZPatternByNode;
  const setShowAllPositions = actions.display.setShowAllPositions;
  const setShowSubgraph = actions.display.setShowSubgraph;
  const setSubgraphData = actions.display.setSubgraphData;
  const setSubgraphRootNodeId = actions.display.setSubgraphRootNodeId;
  const setShowDiffingLogs = actions.display.setShowDiffingLogs;
  const setPerturbedFen = actions.featureDiffing.setPerturbedFen;
  const setIsComparingFens = actions.featureDiffing.setIsComparingFens;
  const setInactiveNodes = actions.featureDiffing.setInactiveNodes;
  // setDiffingLogs needs to support functional updates: (prev) => [...prev, newEntry] or []
  // Note: reducer's addDiffingLog automatically adds timestamp and formats message, so we extract just the message part
  const setDiffingLogs = useCallback((updater: Array<{timestamp: number; message: string}> | ((prev: Array<{timestamp: number; message: string}>) => Array<{timestamp: number; message: string}>)) => {
    if (typeof updater === 'function') {
      const newLogs = updater(diffingLogs);
      // Clear and rebuild logs
      actions.featureDiffing.clearDiffingLogs();
      // Extract message from logEntry (reducer will add timestamp and format)
      newLogs.forEach(log => {
        // Remove the timestamp prefix if present: "[HH:MM:SS] message" -> "message"
        const message = log.message.replace(/^\[\d{1,2}:\d{2}:\d{2}\]\s*/, '');
        actions.featureDiffing.addDiffingLog(message);
      });
    } else {
      actions.featureDiffing.clearDiffingLogs();
      updater.forEach(log => {
        const message = log.message.replace(/^\[\d{1,2}:\d{2}:\d{2}\]\s*/, '');
        actions.featureDiffing.addDiffingLog(message);
      });
    }
  }, [diffingLogs, actions.featureDiffing]);
  const setEnablePositionMapping = actions.positionMapping.setEnablePositionMapping;
  // setPositionMappingSelections needs to support functional updates: (prev) => Record<number, number>
  const setPositionMappingSelections = useCallback((updater: Record<number, number> | ((prev: Record<number, number>) => Record<number, number>)) => {
    if (typeof updater === 'function') {
      const newSelections = updater(positionMappingSelections);
      actions.positionMapping.setPositionMappingSelections(newSelections);
    } else {
      actions.positionMapping.setPositionMappingSelections(updater);
    }
  }, [positionMappingSelections, actions.positionMapping]);
  // setDraftPositionMappingSelections needs to support functional updates: (prev) => Record<number, number>
  const setDraftPositionMappingSelections = useCallback((updater: Record<number, number> | ((prev: Record<number, number>) => Record<number, number>)) => {
    if (typeof updater === 'function') {
      const newSelections = updater(draftPositionMappingSelections);
      actions.positionMapping.setDraftPositionMappingSelections(newSelections);
    } else {
      actions.positionMapping.setDraftPositionMappingSelections(updater);
    }
  }, [draftPositionMappingSelections, actions.positionMapping]);
  // setPositionMappingApplyNonce needs to support functional updates: (x) => x + 1
  const setPositionMappingApplyNonce = useCallback((updater: number | ((prev: number) => number)) => {
    if (typeof updater === 'function') {
      const newValue = updater(positionMappingApplyNonce);
      actions.positionMapping.setPositionMappingApplyNonce(newValue);
    } else {
      actions.positionMapping.setPositionMappingApplyNonce(updater);
    }
  }, [positionMappingApplyNonce, actions.positionMapping]);
  const setDenseNodes = actions.dense.setDenseNodes;
  const setDenseThreshold = actions.dense.setDenseThreshold;
  const setCheckingDenseFeatures = actions.dense.setCheckingDenseFeatures;
  const setSyncingToBackend = actions.sync.setSyncingToBackend;
  const setSyncingFromBackend = actions.sync.setSyncingFromBackend;
  const setEditingClerp = actions.clerp.setEditingClerp;
  const setIsSaving = actions.clerp.setIsSaving;
  const setSteeringScale = actions.steering.setSteeringScale;
  const setSteeringScaleInput = actions.steering.setSteeringScaleInput;
  const setPosFeatureLayer = actions.posFeature.setPosFeatureLayer;
  const setPosFeaturePositions = actions.posFeature.setPosFeaturePositions;
  const setPosFeatureComponentType = actions.posFeature.setPosFeatureComponentType;

  const parseNodeIdParts = useCallback((nodeId: string) => {
    const parsed = parseNodeId(nodeId);
    return { rawLayer: parsed.rawLayer, featureOrHead: parsed.featureOrHead, ctxIdx: parsed.ctxIdx };
  }, []);

  // When multi-graph file list changes, initialize position selections (default 0)
  useEffect(() => {
    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!names || names.length <= 1) return;
    setPositionMappingSelections((prev) => {
      const next: Record<number, number> = { ...prev };
      for (let i = 0; i < names.length; i++) {
        if (typeof next[i] !== "number" || !Number.isFinite(next[i])) {
          next[i] = 0;
        }
      }
      return next;
    });
  }, [linkGraphData, setPositionMappingSelections]);

  // Sync draft state: when multi-graph changes or first entry, make draft default to applied selections
  useEffect(() => {
    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!names || names.length <= 1) return;
    setDraftPositionMappingSelections((prev) => {
      const next: Record<number, number> = { ...prev };
      for (let i = 0; i < names.length; i++) {
        const applied = positionMappingSelections[i];
        const v = (typeof applied === "number" && Number.isFinite(applied)) ? applied : 0;
        if (typeof next[i] !== "number" || !Number.isFinite(next[i])) {
          next[i] = v;
        }
      }
      return next;
    });
  }, [linkGraphData, positionMappingSelections, setDraftPositionMappingSelections]);

  const handleFeatureClick = useCallback((node: Node, isMetaKey: boolean) => {
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedIds.includes(node.nodeId)
        ? pinnedIds.filter(id => id !== node.nodeId)
        : [...pinnedIds, node.nodeId];
      setPinnedIds(newPinnedIds);
    } else {
      // Set clicked node
      const newClickedId = node.nodeId === clickedId ? null : node.nodeId;
      setClickedId(newClickedId);
      
      // 注释掉自动退出子图模式的逻辑，让用户在子图中可以自由点击其他节点
      // 用户可以通过"退出子图"按钮或"显示子图"按钮手动控制子图模式
      // if (newClickedId !== clickedId && showSubgraph) {
      //   setShowSubgraph(false);
      //   setSubgraphData(null);
      //   setSubgraphRootNodeId(null);
      //   console.log('🔄 切换节点，自动退出子图模式');
      // }
    }
  }, [clickedId, pinnedIds, setClickedId, setPinnedIds]);

  const handleFeatureHover = useCallback((nodeId: string | null) => {
    // Only update if the hovered ID has actually changed
    if (nodeId !== hoveredId) {
      setHoveredId(nodeId);
    }
  }, [hoveredId, setHoveredId]);

  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
  }, []);

  const handleConnectedFeaturesLoading = useCallback((_loading: boolean) => {
    // 保留回调函数以保持接口兼容性
  }, []);

  // Use file upload hook
  const { handleSingleFileUpload: handleSingleFileUploadHook, handleMultiFilesUpload: handleMultiFilesUploadHook } = useCircuitFileUpload({
    onSuccess: useCallback((data: any, originalJson: any, fileName: string, multiJsons?: { json: CircuitJsonData; fileName: string }[]) => {
      setLinkGraphData(data);
      // Reset circuit state when loading new data
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      
      // Update reducer state
      actions.file.setOriginalJson(originalJson);
      actions.file.setOriginalFileName(fileName);
      actions.file.setMultiOriginalJsons(multiJsons || []);
      actions.clerp.setEditingClerp('');
      actions.file.setHasUnsavedChanges(false);
      actions.file.clearSaveHistory();
      
      // Reset feature diffing state
      actions.featureDiffing.setPerturbedFen('');
      actions.featureDiffing.setInactiveNodes(new Set());
      actions.featureDiffing.clearDiffingLogs();
      actions.display.setShowDiffingLogs(false);
      
      // Reset subgraph state
      actions.display.setShowSubgraph(false);
      actions.display.setSubgraphData(null);
      actions.display.setSubgraphRootNodeId(null);
      
      // Reset activation display mode
      actions.display.setShowAllPositions(false);
      actions.activation.setAllPositionsData(null);
      actions.activation.setMultiGraphData({});
    }, [setLinkGraphData, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures, actions]),
    onError: setError,
    setLoading,
  });

  // Wrapper functions for compatibility
  const handleSingleFileUpload = useCallback(async (file: File) => {
    await handleSingleFileUploadHook(file);
  }, [handleSingleFileUploadHook]);

  const handleMultiFilesUpload = useCallback(async (files: FileList | File[]) => {
    await handleMultiFilesUploadHook(files);
  }, [handleMultiFilesUploadHook]);


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
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  // Extract FEN from circuit data (using hook)
  const extractFenFromPrompt = fenExtraction.extractFenFromPrompt;

  // Extract FEN and moves from circuit JSON (using hooks)
  const extractFenFromCircuitJson = fenExtraction.extractFenFromCircuitJson;
  const extractOutputMove = fenExtraction.extractOutputMove;
  const extractOutputMoveFromCircuitJson = fenExtraction.extractOutputMoveFromCircuitJson;
 
  // Get node activation data (using hook)
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    return activationDataHook.getNodeActivationData(nodeId);
  }, [activationDataHook]);

  // Get node activation data from JSON (using hook)
  const getNodeActivationDataFromJson = activationDataHook.getNodeActivationDataFromJson;

  // Get dictionary name (using hook)
  const getDictionaryName = dictionaryName.getDictionaryName;

  // Get SAE name for Circuit (using hook)
  const getSaeNameForCircuit = dictionaryName.getSaeNameForCircuit;
 

  // Synchronous version: get activation data for all positions from file (for multi-file mode)
  const getAllPositionsActivationDataSync = useCallback((nodeId: string | null, jsonData?: any): NodeActivationData | null => {
    const dataToSearch = jsonData || file.originalCircuitJson;
    if (!nodeId || !dataToSearch) {
      return null;
    }

    // 解析 node_id -> rawLayer, featureOrHead
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead };
    };
    const parsed = parseFromNodeId(nodeId);

    // 获取节点类型
    let nodesToSearch: any[] = [];
    if (dataToSearch.nodes && Array.isArray(dataToSearch.nodes)) {
      nodesToSearch = dataToSearch.nodes;
    } else if (Array.isArray(dataToSearch)) {
      nodesToSearch = dataToSearch;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray((dataToSearch as any)[key])) {
          nodesToSearch = (dataToSearch as any)[key];
          break;
        }
      }
    }

    let featureTypeForNode: string | undefined = undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find(n => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    // 构建可扫描的记录集合
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) {
              candidateRecords.push(item);
            }
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(dataToSearch);

    // 匹配函数：匹配相同的 layer 和 feature_idx/head_idx，但忽略 position
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && indexOk;
    };

    // 找到所有匹配的记录（所有位置）
    const matchedRecords = candidateRecords.filter(rec => tryMatchRecord(rec, featureTypeForNode));

    if (matchedRecords.length === 0) {
      return null;
    }

    // 合并所有位置的激活值
    const mergedActivations = new Array(64).fill(0);
    // ⚠️ 注意：z_pattern 语义上是“某个 query position 的 attention pattern”
    // 在“所有位置模式”下如果跨 position 聚合 z_pattern 会产生误导，因此这里明确不聚合/不展示 z_pattern

    for (const rec of matchedRecords) {
      if (rec.activations && Array.isArray(rec.activations) && rec.activations.length === 64) {
        for (let i = 0; i < 64; i++) {
          const currentValue = mergedActivations[i];
          const newValue = rec.activations[i];
          if (Math.abs(newValue) > Math.abs(currentValue)) {
            mergedActivations[i] = newValue;
          }
        }
      }
    }

    return {
      activations: mergedActivations,
      zPatternIndices: undefined,
      zPatternValues: undefined,
      nodeType: featureTypeForNode,
      clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
    };
  }, [normalizeZPattern, originalCircuitJson]);

  // 从后端获取所有位置的激活数据
  const fetchAllPositionsFromBackend = useCallback(async (
    nodeId: string,
    fen: string,
    dictionary: string,
    featureIndex: number
  ): Promise<NodeActivationData | null> => {
    try {
      setLoadingAllPositions(true);
      console.log('🔍 从后端获取所有位置数据:', { nodeId, fen, dictionary, featureIndex });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ fen: fen.trim() }),
        }
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('✅ 从后端获取到所有位置数据:', data);
      
      // 在所有位置模式下，聚合所有位置的激活数据
      // 显示该feature在每个棋盘位置上的激活强度（取最大值）
      const mergedActivations = new Array(64).fill(0);

      if (data.positions && Array.isArray(data.positions)) {
        for (const posData of data.positions) {
          if (posData.activations && Array.isArray(posData.activations) && posData.activations.length === 64) {
            // 对每个棋盘位置，保留最大激活值
            for (let i = 0; i < 64; i++) {
              const newValue = posData.activations[i];
              if (Math.abs(newValue) > Math.abs(mergedActivations[i])) {
                mergedActivations[i] = newValue;
              }
            }
          }
        }
      }

      console.log('✅ 聚合完成（所有位置模式）:', {
        hasActivations: mergedActivations.some(v => v !== 0),
        nonZeroCount: mergedActivations.filter(v => v !== 0).length
      });

      // 获取节点类型
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const nodeType = currentNode?.feature_type;

      const result = {
        activations: mergedActivations,
        zPatternIndices: undefined, // 所有位置模式下不显示z_pattern
        zPatternValues: undefined,
        nodeType: nodeType,
        clerp: (currentNode as any)?.clerp,
      };
      
      console.log('📤 返回的NodeActivationData:', {
        hasActivations: !!result.activations,
        activationsLength: result.activations?.length,
        hasZPatternIndices: !!result.zPatternIndices,
        hasZPatternValues: !!result.zPatternValues
      });
      
      return result;
    } catch (error) {
      console.error('❌ 从后端获取所有位置数据失败:', error);
      return null;
    } finally {
      actions.activation.setLoadingAllPositions(false);
    }
  }, [linkGraphData, normalizeZPattern, actions]);

  // 与 CustomFenInput 保持一致：调用 analyze_fen 获取“全局(所有位置)激活 + z_pattern(若有)”
  // - activations: feature_acts_indices/values -> 64维稠密数组
  // - z_pattern: z_pattern_indices/values（LoRSA 才有）
  const fetchAnalyzeFenFromBackend = useCallback(async (
    dictionary: string,
    featureIndex: number,
    fen: string,
    signal?: AbortSignal
  ): Promise<NodeActivationData | null> => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ fen: fen.trim() }),
          signal,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // 解析 activations（稀疏 -> 64稠密）
      let activations: number[] | undefined = undefined;
      if (data.feature_acts_indices && data.feature_acts_values) {
        activations = new Array(64).fill(0);
        const indices = data.feature_acts_indices as number[];
        const values = data.feature_acts_values as number[];
        for (let i = 0; i < Math.min(indices.length, values.length); i++) {
          const idx = indices[i];
          const v = values[i];
          if (typeof idx === "number" && idx >= 0 && idx < 64) activations[idx] = v;
        }
      }

      // 解析 z_pattern（后端 snake_case）
      const { zPatternIndices, zPatternValues } = normalizeZPattern(
        (data as any).z_pattern_indices,
        (data as any).z_pattern_values
      );

      return { activations, zPatternIndices, zPatternValues };
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return null;
      console.error("❌ analyze_fen 失败:", error);
      return null;
    }
  }, [normalizeZPattern]);

  // 从后端获取某个 LoRSA feature 在某个 query position 的 z_pattern（单位置展示用）
  const fetchZPatternForPosFromBackend = useCallback(async (
    dictionary: string,
    featureIndex: number,
    fen: string,
    queryPos: number,
    signal?: AbortSignal
  ): Promise<{ zPatternIndices?: number[][]; zPatternValues?: number[] } | null> => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ fen: fen.trim() }),
          signal,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const data = await response.json();
      const positions = data?.positions;
      if (!Array.isArray(positions)) return null;

      const posData = positions.find((p: any) => Number(p?.position) === queryPos);
      if (!posData) return null;

      // 后端返回 snake_case：z_pattern_indices / z_pattern_values
      const { zPatternIndices, zPatternValues } = normalizeZPattern(
        (posData as any).z_pattern_indices,
        (posData as any).z_pattern_values
      );

      return { zPatternIndices, zPatternValues };
    } catch (error) {
      // fetch 被 AbortController 取消时不用报错
      if (error instanceof DOMException && error.name === "AbortError") return null;
      console.error("❌ 从后端获取 z_pattern 失败:", error);
      return null;
    }
  }, [normalizeZPattern]);

  // 获取该 feature 在所有位置的激活数据
  // 注意：在所有位置模式下，逻辑与 CustomFenInput 一致：调用 analyze_fen（不再信任 JSON）
  const getAllPositionsActivationData = useCallback(async (nodeId: string | null, _jsonData?: any): Promise<NodeActivationData | null> => {
    if (!nodeId) {
      return null;
    }

    // 解析 node_id -> rawLayer, featureOrHead
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead };
    };
    const parsed = parseFromNodeId(nodeId);

    // 获取节点类型（用于确定是LoRSA还是Transcoder）
    const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
    const featureTypeForNode = currentNode?.feature_type;

    const isLorsa = featureTypeForNode?.toLowerCase() === 'lorsa';
    const dictionary = getDictionaryName(parsed.layerForActivation, isLorsa);

    // 检查是否为多图模式
    const isMultiFile = !!(linkGraphData?.metadata?.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1);

    if (isMultiFile) {
      // 多图模式：对每个graph都进行前向推理
      console.log('🔍 多图所有位置模式：对每个graph进行前向推理');
      setLoadingAllPositions(true);

      const multiResults: Record<number, NodeActivationData> = {};

      for (let i = 0; i < multiOriginalJsons.length; i++) {
        const jsonData = multiOriginalJsons[i].json;
        const fileFen = extractFenFromCircuitJson(jsonData);

        if (fileFen) {
          console.log(`📊 处理第${i+1}个graph，FEN: ${fileFen}`);
          const result = await fetchAllPositionsFromBackend(nodeId, fileFen, dictionary, parsed.featureOrHead);
          if (result) {
            multiResults[i] = {
              ...result,
              nodeType: featureTypeForNode,
              clerp: (currentNode as any)?.clerp,
            };
            console.log(`✅ 第${i+1}个graph处理完成`);
          } else {
            console.log(`❌ 第${i+1}个graph处理失败`);
          }
        } else {
          console.log(`⚠️ 第${i+1}个graph无法提取FEN`);
        }
      }

      // 更新多图激活数据状态
      setMultiGraphActivationData(multiResults);
      setLoadingAllPositions(false);
      console.log('✅ 所有graph的激活数据已获取完成');

      return null; // 多图模式不返回单个结果
    } else {
      // 单图模式：在所有位置模式下调用后端获取所有位置数据
      console.log('🔍 单图所有位置模式：调用 analyze_fen_all_positions');
      const fen = extractFenFromPrompt();
      if (!fen) {
        console.log('❌ 无法提取FEN，无法从后端获取数据');
        return null;
      }

      const result = await fetchAllPositionsFromBackend(nodeId, fen, dictionary, parsed.featureOrHead);
      if (!result) return null;

      return {
        ...result,
        nodeType: featureTypeForNode,
        clerp: (currentNode as any)?.clerp,
      };
    }
  }, [linkGraphData, extractFenFromPrompt, getDictionaryName, fetchAnalyzeFenFromBackend, multiOriginalJsons, setMultiGraphActivationData, setLoadingAllPositions, extractFenFromCircuitJson]);

  // 提取相关数据
  const fen = extractFenFromPrompt();
  const outputMove = extractOutputMove();
  const nodeActivationData = getNodeActivationData(clickedId);
  
  // 当切换节点时，重置显示模式为单个位置
  useEffect(() => {
    setShowAllPositions(false);
    setAllPositionsActivationData(null);
    setMultiGraphActivationData({});
  }, [clickedId]);

  // 点击节点时：实时从后端拉取该节点的 z_pattern（LoRSA + 单位置模式 + 单文件场景）
  useEffect(() => {
    // 清空旧的 z_pattern（避免切换节点时短暂显示上一个节点的 pattern）
    setBackendZPatternByNode(null);

    if (!clickedId) return;
    if (showAllPositions) return; // 所有位置模式不展示/不聚合 z_pattern

    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    const isMultiFile = !!(names && names.length > 1);
    if (isMultiFile) return; // 多文件模式下先保持原逻辑（避免每文件/每点击多次重算导致很慢）

    const currentNode = (linkGraphData?.nodes || []).find((n: any) => n?.nodeId === clickedId);
    const featureType = typeof currentNode?.feature_type === "string" ? currentNode.feature_type.toLowerCase() : "";
    const isLorsa = featureType === "lorsa";
    if (!isLorsa) return; // Transcoder 没有 z_pattern

    const fenLocal = extractFenFromPrompt();
    if (!fenLocal) return;

    const parts = clickedId.split("_");
    const rawLayer = Number(parts[0]) || 0;
    const featureIndex = Number(parts[1]) || 0;
    const pos = Number(parts[2]) || 0;
    const layerIdx = Math.floor(rawLayer / 2);
    const dictionary = getDictionaryName(layerIdx, true);

    const controller = new AbortController();
    setLoadingBackendZPattern(true);

    fetchZPatternForPosFromBackend(dictionary, featureIndex, fenLocal, pos, controller.signal)
      .then((zp) => {
        if (!zp) return;
        setBackendZPatternByNode({
          nodeId: clickedId,
          zPatternIndices: zp.zPatternIndices,
          zPatternValues: zp.zPatternValues,
        });
      })
      .finally(() => {
        setLoadingBackendZPattern(false);
      });

    return () => {
      controller.abort();
    };
  }, [
    clickedId,
    showAllPositions,
    linkGraphData,
    extractFenFromPrompt,
    getDictionaryName,
    fetchZPatternForPosFromBackend,
  ]);

  // 当点击节点或切换模式时，更新所有位置的激活数据
  useEffect(() => {
    if (clickedId && showAllPositions) {
      const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
      const isMultiFile = !!(names && names.length > 1);

      if (!isMultiFile) {
        // 单文件模式：调用 getAllPositionsActivationData 获取数据
        setLoadingAllPositions(true);
        getAllPositionsActivationData(clickedId).then((allPosData) => {
          setAllPositionsActivationData(allPosData);
          setLoadingAllPositions(false);
        }).catch((error) => {
          console.error('获取所有位置数据失败:', error);
          setAllPositionsActivationData(null);
          setLoadingAllPositions(false);
        });
      } else {
        // 多文件模式：直接调用 getAllPositionsActivationData，它会处理多文件逻辑
        getAllPositionsActivationData(clickedId);
      }
    } else {
      setAllPositionsActivationData(null);
      setLoadingAllPositions(false);
    }
  }, [clickedId, showAllPositions, linkGraphData, getAllPositionsActivationData]);

  // 确定要显示的激活数据
  // - activations：沿用现有逻辑（JSON/后端 all-positions 聚合）
  // - z_pattern：在“单位置模式”下用后端实时计算结果覆盖（避免 JSON 内 z_pattern 偶发错误）
  const displayActivationData = useMemo(() => {
    const base = (showAllPositions && allPositionsActivationData)
      ? allPositionsActivationData
      : nodeActivationData;

    if (!showAllPositions && clickedId && backendZPatternByNode?.nodeId === clickedId) {
      return {
        ...base,
        zPatternIndices: backendZPatternByNode.zPatternIndices,
        zPatternValues: backendZPatternByNode.zPatternValues,
      };
    }
    return base;
  }, [showAllPositions, allPositionsActivationData, nodeActivationData, clickedId, backendZPatternByNode]);

  // 修复Hook使用 - 移到组件顶层，避免条件调用
  useEffect(() => {
    if (clickedId && nodeActivationData) {
      // 无论clerp是undefined、空字符串还是有内容，都设置到编辑器中
      const clerpValue = nodeActivationData.clerp || '';
      console.log('🔄 更新编辑器状态:', {
        nodeId: clickedId,
        clerpValue,
        clerpType: typeof nodeActivationData.clerp,
        clerpLength: clerpValue.length,
        updateCounter
      });
      setEditingClerp(clerpValue);
    } else {
      // 没有选中节点时，清空编辑器
      console.log('🔄 清空编辑器状态');
      setEditingClerp('');
      // 重置所有位置显示模式
      setShowAllPositions(false);
      setAllPositionsActivationData(null);
      // 同时重置多图激活数据
      setMultiGraphActivationData({});
    }
  }, [clickedId, nodeActivationData?.clerp, updateCounter]);

  const handleSaveClerp = useCallback(async () => {
    console.log('🚀 开始保存clerp:', {
      clickedId,
      hasOriginalCircuitJson: !!originalCircuitJson,
      editingClerp,
      editingClerpLength: editingClerp.length,
      trimmedLength: editingClerp.trim().length
    });
    
    if (!clickedId || !originalCircuitJson) {
      console.log('❌ 保存失败：缺少必要数据');
      return;
    }

    // 允许保存空内容，但至少要有一些变化
    const trimmedClerp = editingClerp.trim();
    
    setIsSaving(true);
    
    try {
      // 先创建深拷贝，避免直接修改原始数据
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // 查找并更新对应的节点
      let updated = false;
      let nodesToSearch: any[] = [];
      
      if (updatedCircuitJson.nodes && Array.isArray(updatedCircuitJson.nodes)) {
        nodesToSearch = updatedCircuitJson.nodes;
      } else if (Array.isArray(updatedCircuitJson)) {
        nodesToSearch = updatedCircuitJson;
      } else {
        const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
        for (const key of possibleArrayKeys) {
          if (Array.isArray(updatedCircuitJson[key])) {
            nodesToSearch = updatedCircuitJson[key];
            break;
          }
        }
        
        if (nodesToSearch.length === 0) {
          const values = Object.values(updatedCircuitJson);
          const arrayValue = values.find(v => Array.isArray(v));
          if (arrayValue) {
            nodesToSearch = arrayValue as any[];
          }
        }
      }

      // 直接通过node_id匹配并更新节点的clerp
      for (const node of nodesToSearch) {
        if (node && typeof node === 'object' && node.node_id === clickedId) {
          // 设置clerp字段，无论之前是否存在
          const previousClerp = node.clerp;
          node.clerp = trimmedClerp;
          updated = true;
          console.log('✅ 已更新节点clerp:', {
            node_id: clickedId,
            feature: node.feature,
            layer: node.layer,
            feature_type: node.feature_type,
            previousClerp: previousClerp || '(空)',
            newClerp: trimmedClerp || '(空)',
            newClerpLength: trimmedClerp.length
          });
          break;
        }
      }

      if (updated) {
        // 更新状态为修改后的深拷贝
        setOriginalCircuitJson(updatedCircuitJson);
        
        // 强制触发重新获取节点数据
        setUpdateCounter(prev => prev + 1);
        
        // 标记为有未保存的更改
        setHasUnsavedChanges(true);
        
        console.log('✅ 本地数据已更新，触发重新渲染');
        console.log('🔍 验证更新:', {
          nodeId: clickedId,
          updatedClerp: updatedCircuitJson.nodes?.find((n: any) => n.node_id === clickedId)?.clerp,
          updateCounter: updateCounter + 1
        });
        
        // 自动下载更新后的文件（使用原文件名）
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const fileName = originalFileName || 'circuit_data.json';
        const baseName = fileName.replace('.json', '');
        const updatedFileName = `${baseName}_updated_${timestamp}.json`;
        
        const updatedJsonString = JSON.stringify(updatedCircuitJson, null, 2);
        const blob = new Blob([updatedJsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = updatedFileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        // 添加到保存历史
        setSaveHistory(prev => [...prev, `${new Date().toLocaleTimeString()}: 节点 ${clickedId} - ${trimmedClerp.length === 0 ? '清空clerp' : `更新为: ${trimmedClerp.substring(0, 30)}...`}`]);
        
        console.log('📥 文件已自动下载:', updatedFileName);
        
        // 显示成功消息和使用指引
        alert(`✅ Clerp已成功保存并下载！${trimmedClerp.length === 0 ? '(保存为空内容)' : ''}\n\n📁 文件已保存到Downloads文件夹:\n${updatedFileName}\n\n💡 使用提示:\n1. 可以直接用新文件替换原文件\n2. 或者重新上传新文件到此页面\n3. 文件名包含时间戳避免覆盖`);
        
      } else {
        throw new Error(`未找到对应的节点数据 (node_id: ${clickedId})`);
      }
    } catch (err) {
      console.error('保存失败:', err);
      alert('保存失败: ' + (err instanceof Error ? err.message : '未知错误'));
    } finally {
      setIsSaving(false);
    }
  }, [clickedId, originalCircuitJson, editingClerp, originalFileName, setOriginalCircuitJson, updateCounter]);

  // 快速导出当前状态的函数
  const handleQuickExport = useCallback(() => {
    if (!originalCircuitJson) return;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const fileName = originalFileName || 'circuit_data.json';
    const baseName = fileName.replace('.json', '');
    const exportFileName = `${baseName}_export_${timestamp}.json`;
    
    const jsonString = JSON.stringify(originalCircuitJson, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    setHasUnsavedChanges(false);
    console.log('📤 快速导出完成:', exportFileName);
    alert(`📤 文件已导出到Downloads文件夹:\n${exportFileName}\n\n💡 要使用更新后的文件:\n1. 用新文件替换原文件\n2. 或者拖拽新文件到此页面重新加载`);
  }, [originalCircuitJson, originalFileName]);

  // 获取 Top Activation 数据的函数
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId) return;
    
    setLoadingTopActivations(true);
    try {
      // 从 nodeId 解析出 feature 信息
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // 确定节点类型和对应的字典名
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
      
      // 使用辅助函数获取字典名
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      
      console.log('🔍 获取 Top Activation 数据:', {
        nodeId,
        layerIdx,
        featureIndex,
        dictionary,
        isLorsa
      });
      
      // 调用后端 API 获取 feature 数据
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: "GET",
          headers: {
            Accept: "application/x-msgpack",
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const arrayBuffer = await response.arrayBuffer();
      const decoded = await import("@msgpack/msgpack").then(module => module.decode(new Uint8Array(arrayBuffer)));
      const camelcaseKeys = await import("camelcase-keys").then(module => module.default);
      
      // 解析数据
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
      // 提取样本数据
      const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
      const allSamples: any[] = [];
      
      for (const group of sampleGroups) {
        if (group.samples && Array.isArray(group.samples)) {
          allSamples.push(...group.samples);
        }
      }
      
      // 查找包含 FEN 的样本并提取激活值
      const chessSamples: any[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          
          for (const line of lines) {
            const trimmed = line.trim();
            
            // 检查是否包含 FEN 格式
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
                  // 验证 FEN 格式
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
                    // 处理稀疏激活数据 - 正确映射到64格棋盘
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0; // 使用最大激活值而不是总和
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      // 创建64格的激活数组
                      activationsArray = new Array(64).fill(0);
                      
                      // 将稀疏激活值映射到正确的棋盘位置，并找到最大激活值
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        // 确保索引在有效范围内
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          // 使用最大激活值（与feature页面逻辑一致）
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                      
                      console.log('🔍 处理激活数据:', {
                        indicesLength: sample.featureActsIndices.length,
                        valuesLength: sample.featureActsValues.length,
                        nonZeroCount: activationsArray.filter(v => v !== 0).length,
                        maxActivation
                      });
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation, // 使用最大激活值作为排序依据
                      activations: activationsArray,
                      ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break; // 找到一个有效 FEN 就跳出
                  }
                }
              }
            }
          }
        }
      }
      
      // 按最大激活值排序并取前8个（与feature页面逻辑一致）
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      console.log('✅ 获取到 Top Activation 数据:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length
      });
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error('❌ 获取 Top Activation 数据失败:', error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [linkGraphData, getDictionaryName]);

  // 检查 SAE 是否已加载（直接检查后端状态）
  const checkSaeLoaded = useCallback(async (): Promise<boolean> => {
    try {
      const saeComboId = typeof window !== 'undefined' 
        ? window.localStorage.getItem("bt4_sae_combo_id") 
        : null;
      
      if (!saeComboId) {
        console.warn('未找到 sae_combo_id，请先加载 SAE 组合');
        return false;
      }
      
      const params = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        sae_combo_id: saeComboId,
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/loading_logs?${params.toString()}`
      );
      
      if (response.ok) {
        const data = await response.json();
        const logs = data.logs || [];
        
        // 如果正在加载，返回 false
        if (data.is_loading === true) {
          console.log('SAE 正在加载中...');
          return false;
        }
        
        // 检查是否有成功加载的日志（包含完成消息）
        const hasSuccessLog = logs.some((log: { message: string }) => 
          log.message.includes('✅ 预加载完成') || 
          log.message.includes('预加载完成') ||
          log.message.includes('already_loaded') ||
          log.message.includes('已就绪')
        );
        
        // 如果有成功日志，说明已加载
        if (hasSuccessLog) {
          console.log('✅ SAE 已加载（从日志确认）');
          return true;
        }
        
        // 如果没有日志，说明可能还没有加载过
        if (logs.length === 0) {
          console.warn('未找到加载日志，SAE 可能未加载');
          return false;
        }
        
        // 如果有日志但不在加载中，且没有成功消息，可能是加载失败或正在加载
        console.warn('SAE 状态不明确，日志存在但无成功消息');
        return false;
      }
      
      return false;
    } catch (error) {
      console.warn('检查 SAE 加载状态失败:', error);
      return false;
    }
  }, []);

  // 获取 Token Predictions 数据的函数
  const fetchTokenPredictions = useCallback(async (nodeId: string, currentSteeringScale?: number) => {
    if (!nodeId || !fen) return;

    // 直接检查后端状态，而不是依赖全局状态
    const saeLoaded = await checkSaeLoaded();
    if (!saeLoaded) {
      console.warn("TC/LoRSA 未加载，跳过 steering_analysis 调用");
      alert("请先在上方加载 TC/LoRSA 组合（SaeComboLoader），再使用 steering 功能。");
      setTokenPredictions(null);
      return;
    }
    
    setLoadingTokenPredictions(true);
    try {
      // 从 nodeId 解析出特征信息
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const pos = Number(parts[2]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // 确定节点类型
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const featureType = currentNode?.feature_type?.toLowerCase() === 'lorsa' ? 'lorsa' : 'transcoder';
      
      // 使用传入的 steeringScale 或当前状态中的值
      const scaleToUse = currentSteeringScale !== undefined ? currentSteeringScale : steeringScale;
      
      console.log('🔍 获取 Token Predictions 数据:', {
        nodeId,
        layerIdx,
        featureIndex,
        pos,
        featureType,
        fen,
        steering_scale: scaleToUse
      });
      
      // 调用后端 API 进行 steering 分析（支持 steering_scale 参数）
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/steering_analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            fen: fen,
            feature_type: featureType,
            layer: layerIdx,
            pos: pos,
            feature: featureIndex,
            steering_scale: scaleToUse,
            metadata: linkGraphData?.metadata
          })
        }
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        // 如果是503错误，说明模型未加载
        if (response.status === 503) {
          alert("请先在上方加载 TC/LoRSA 组合（SaeComboLoader），再使用 steering 功能。");
        }
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 获取到 Token Predictions 数据:', result);
      
      setTokenPredictions(result);
      
    } catch (error) {
      console.error('❌ 获取 Token Predictions 数据失败:', error);
      setTokenPredictions(null);
    } finally {
      setLoadingTokenPredictions(false);
    }
  }, [fen, linkGraphData, steeringScale, checkSaeLoaded]);

  // 当点击节点时获取 Top Activation 数据（Token Predictions 改为手动触发）
  useEffect(() => {
    if (clickedId) {
      fetchTopActivations(clickedId);
    } else {
      setTopActivations([]);
      setTokenPredictions(null);
    }
  }, [clickedId, fetchTopActivations]);

  // 同步clerps到后端interpretations
  const syncClerpsToBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert('⚠️ 没有可用的节点数据');
      return;
    }
    
    setSyncingToBackend(true);
    try {
      // 从metadata中提取analysis_name
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      // 准备节点数据
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          clerp: node.clerp || '',
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('📤 开始同步clerps到后端:', {
        totalNodes: nodes.length,
        nodesWithClerp: nodes.filter((n: any) => n.clerp).length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_clerps_to_interpretations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 同步完成:', result);
      
      alert(
        `✅ Clerp同步到后端完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 成功同步: ${result.synced}\n` +
        `- 跳过(无clerp): ${result.skipped}\n` +
        `- 失败: ${result.errors}`
      );
      
    } catch (error) {
      console.error('❌ 同步失败:', error);
      alert(`❌ 同步失败: ${error}`);
    } finally {
      setSyncingToBackend(false);
    }
  }, [originalCircuitJson, linkGraphData]);

  // 从后端interpretations同步到clerps
  const syncClerpsFromBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert('⚠️ 没有可用的节点数据');
      return;
    }
    
    setSyncingFromBackend(true);
    try {
      // 从metadata中提取analysis_name
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      // 准备节点数据
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('📥 开始从后端同步interpretations到clerps:', {
        totalNodes: nodes.length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ 同步完成:', result);
      
      // 更新原始JSON数据
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // 根据返回的updated_nodes更新clerp
      const updatedNodesMap = new Map(
        result.updated_nodes.map((n: any) => [n.node_id, n.clerp])
      );
      
      let updatedCount = 0;
      updatedCircuitJson.nodes.forEach((node: any) => {
        if (updatedNodesMap.has(node.node_id)) {
          const newClerp = updatedNodesMap.get(node.node_id);
          if (newClerp) {
            node.clerp = newClerp;
            updatedCount++;
          }
        }
      });
      
      // 更新状态
      setOriginalCircuitJson(updatedCircuitJson);
      setUpdateCounter(prev => prev + 1);
      setHasUnsavedChanges(true);
      
      alert(
        `✅ 从后端同步Interpretation完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 找到interpretation: ${result.found}\n` +
        `- 未找到: ${result.not_found}\n` +
        `- 实际更新: ${updatedCount}\n\n` +
        `💡 建议: 点击"导出"按钮保存更新后的文件`
      );
      
    } catch (error) {
      console.error('❌ 同步失败:', error);
      alert(`❌ 同步失败: ${error}`);
    } finally {
      setSyncingFromBackend(false);
    }
  }, [originalCircuitJson, linkGraphData, setOriginalCircuitJson, setUpdateCounter, setHasUnsavedChanges]);

  // 检查dense features的函数
  const checkDenseFeatures = useCallback(async () => {
    if (!linkGraphData || !linkGraphData.nodes) {
      console.warn('⚠️ 没有可用的节点数据');
      return;
    }
    
    setCheckingDenseFeatures(true);
    try {
      const threshold = denseThreshold === '' ? null : parseInt(denseThreshold);
      
      // 从linkGraphData中提取所有节点的信息
      const nodes = linkGraphData.nodes.map(node => {
        // 从nodeId解析layer和feature
        const parts = node.nodeId.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.nodeId,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      // 从metadata中提取模型名称并转换为analysis_name
      const metadata = (linkGraphData.metadata || {}) as any;
      const lorsaAnalysisNameRaw = metadata.lorsa_analysis_name;
      const tcAnalysisNameRaw = metadata.tc_analysis_name || metadata.clt_analysis_name;
      // 从metadata中读取sae_series，如果没有则使用默认值
      const saeSeries = (metadata as any).sae_series || 'BT4-exp128';
      
      // 根据analysis_name构建模板
      // 格式：如果analysis_name是 "BT4_lorsa_k30_e16"，模板应该是 "BT4_lorsa_L{}A_k30_e16"
      // 如果analysis_name是 "BT4_lorsa"（默认），模板应该是 "BT4_lorsa_L{}A"
      let lorsaAnalysisName = undefined;
      let tcAnalysisName = undefined;
      
      if (lorsaAnalysisNameRaw) {
        if (lorsaAnalysisNameRaw.includes('BT4_lorsa')) {
          // 提取后缀（如果有）："BT4_lorsa_k30_e16" -> "k30_e16"
          const suffix = lorsaAnalysisNameRaw.replace('BT4_lorsa', '').replace(/^_/, '');
          if (suffix) {
            lorsaAnalysisName = `BT4_lorsa_L{}A_${suffix}`;
          } else {
            lorsaAnalysisName = 'BT4_lorsa_L{}A';
          }
        } else if (lorsaAnalysisNameRaw.includes('T82')) {
          lorsaAnalysisName = 'lc0-lorsa-L{}';
        }
      }
      
      if (tcAnalysisNameRaw) {
        if (tcAnalysisNameRaw.includes('BT4_tc')) {
          // 提取后缀（如果有）："BT4_tc_k30_e16" -> "k30_e16"
          const suffix = tcAnalysisNameRaw.replace('BT4_tc', '').replace(/^_/, '');
          if (suffix) {
            tcAnalysisName = `BT4_tc_L{}M_${suffix}`;
          } else {
            tcAnalysisName = 'BT4_tc_L{}M';
          }
        } else if (tcAnalysisNameRaw.includes('T82')) {
          tcAnalysisName = 'lc0_L{}M_16x_k30_lr2e-03_auxk_sparseadam';
        }
      }
      
      console.log('🔍 开始检查dense features:', {
        totalNodes: nodes.length,
        threshold: threshold,
        saeSeries: saeSeries,
        lorsaAnalysisNameRaw: lorsaAnalysisNameRaw,
        tcAnalysisNameRaw: tcAnalysisNameRaw,
        lorsaAnalysisName: lorsaAnalysisName,
        tcAnalysisName: tcAnalysisName,
        sampleNodes: nodes.slice(0, 3)
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/check_dense_features`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            threshold: threshold,
            sae_series: saeSeries,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ Dense features检查完成:', {
        denseNodeCount: result.dense_nodes.length,
        totalNodes: result.total_nodes,
        threshold: result.threshold
      });
      
      setDenseNodes(new Set(result.dense_nodes));
      
    } catch (error) {
      console.error('❌ 检查dense features失败:', error);
      alert(`检查dense features失败: ${error}`);
    } finally {
      setCheckingDenseFeatures(false);
    }
  }, [linkGraphData, denseThreshold]);

  // 应用dense节点颜色覆盖
  const applyDenseNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || denseNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (denseNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#000000',  // 黑色
            isDense: true  // 标记为dense节点
          };
        }
        return node;
      })
    };
  }, [denseNodes]);

  // 应用inactive节点颜色覆盖（金色）
  const applyInactiveNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || inactiveNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (inactiveNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#FFD700',  // 金色
            isInactive: true  // 标记为inactive节点
          };
        }
        return node;
      })
    };
  }, [inactiveNodes]);

  // 应用 position 映射高亮覆盖
  const applyPositionMappingHighlights = useCallback((data: any) => {
    if (!enablePositionMapping) return data;
    if (!data || !data.nodes || !data.metadata?.sourceFileNames || data.metadata.sourceFileNames.length <= 1) {
      return data;
    }

    // 仅对“每个文件所选 position”上的 feature 节点做跨文件对齐：
    // key = (rawLayer, featureOrHead, feature_type_norm)
    // 若同 key 在 >=2 个 source 中出现，则这些节点都高亮
    type Hit = { nodeId: string; sourceIdx: number };
    const buckets = new Map<string, Hit[]>();

    for (const node of data.nodes as any[]) {
      const nodeId = String(node.nodeId);
      const { rawLayer, featureOrHead, ctxIdx } = parseNodeIdParts(nodeId);

      // 仅针对 feature 节点：排除 embedding/logit/error
      const ftRaw = typeof node.feature_type === "string" ? node.feature_type : "";
      const ft = ftRaw.toLowerCase();
      if (ft.includes("embedding") || ft.includes("logit") || ft.includes("error")) continue;

      const srcs: number[] = Array.isArray(node.sourceIndices)
        ? node.sourceIndices
        : (typeof node.sourceIndex === "number" ? [node.sourceIndex] : []);

      for (const s of srcs) {
        const sel = positionMappingSelections[s];
        if (typeof sel !== "number") continue;
        if (ctxIdx !== sel) continue;
        const typeNorm = ft.includes("lorsa") ? "lorsa" : "tc";
        const key = `${rawLayer}_${featureOrHead}_${typeNorm}`;
        const arr = buckets.get(key) || [];
        arr.push({ nodeId, sourceIdx: s });
        buckets.set(key, arr);
      }
    }

    const highlightNodeIds = new Set<string>();
    buckets.forEach((hits) => {
      const uniqSources = new Set(hits.map((h) => h.sourceIdx));
      if (uniqSources.size >= 2) {
        for (const h of hits) highlightNodeIds.add(h.nodeId);
      }
    });

    if (highlightNodeIds.size === 0) return data;

    return {
      ...data,
      nodes: (data.nodes as any[]).map((node) => {
        // Dense 节点优先级最高：保持黑色，不允许被 position 映射高亮覆盖
        if ((node as any)?.isDense === true) {
          return node;
        }
        if (highlightNodeIds.has(String(node.nodeId))) {
          return {
            ...node,
            nodeColor: POSITION_MAPPING_HIGHLIGHT_COLOR,
            isPositionMapped: true,
          };
        }
        return node;
      }),
    };
  }, [enablePositionMapping, parseNodeIdParts, positionMappingSelections, POSITION_MAPPING_HIGHLIGHT_COLOR]);

  // 获取应用了dense和inactive颜色的图数据
  const displayLinkGraphData = useMemo(() => {
    let data = linkGraphData;
    data = applyDenseNodeColors(data);
    data = applyInactiveNodeColors(data);
    data = applyPositionMappingHighlights(data);
    return data;
  }, [linkGraphData, applyDenseNodeColors, applyInactiveNodeColors, applyPositionMappingHighlights]);

  // 统计当前高亮命中的节点数（用于给用户反馈“有没有生效”）
  const positionMappedCount = useMemo(() => {
    const names = (displayLinkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!enablePositionMapping || !names || names.length <= 1) return 0;
    return (displayLinkGraphData?.nodes || []).filter((n: any) => (n as any)?.isPositionMapped === true).length;
  }, [displayLinkGraphData, enablePositionMapping]);

  // 比较FEN激活差异
  const compareFenActivations = useCallback(async () => {
    if (!originalCircuitJson || !perturbedFen.trim()) {
      alert('请先上传graph文件并输入perturbed FEN');
      return;
    }

    // 获取原始FEN
    const originalFen = extractFenFromPrompt();
    if (!originalFen) {
      alert('无法从graph文件中提取原始FEN');
      return;
    }

    setIsComparingFens(true);
    setDiffingLogs([]);
    setShowDiffingLogs(true);

    const addLog = (message: string) => {
      const logEntry = {
        timestamp: Date.now(),
        message: `[${new Date().toLocaleTimeString()}] ${message}`
      };
      setDiffingLogs(prev => [...prev, logEntry]);
      console.log(logEntry.message);
    };

    try {
      addLog('开始比较FEN激活差异...');
      addLog(`原始FEN: ${originalFen}`);
      addLog(`扰动FEN: ${perturbedFen}`);

      // 从metadata中提取模型名称
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const modelName = metadata?.model_name || 'lc0/BT4-1024x15x32h';

      addLog(`使用模型: ${modelName}`);

      // 调用后端API
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/compare_fen_activations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            graph_json: originalCircuitJson,
            original_fen: originalFen,
            perturbed_fen: perturbedFen.trim(),
            model_name: modelName,
            activation_threshold: 0.0
          })
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `HTTP ${response.status}: ${errorText}`;
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch {
          // 如果无法解析JSON，使用原始错误文本
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      
      addLog(`✅ 比较完成:`);
      addLog(`   - 总节点数: ${result.total_nodes}`);
      addLog(`   - 未激活节点数: ${result.inactive_nodes_count}`);

      // 更新inactive nodes集合
      const inactiveNodeIds = new Set<string>(
        result.inactive_nodes.map((node: any) => String(node.node_id))
      );
      setInactiveNodes(inactiveNodeIds);

      // 显示统计信息
      if (result.statistics) {
        addLog(`按层统计:`);
        Object.entries(result.statistics.by_layer).forEach(([layer, count]) => {
          addLog(`   Layer ${layer}: ${count} 个节点`);
        });
        addLog(`按类型统计:`);
        Object.entries(result.statistics.by_type).forEach(([type, count]) => {
          addLog(`   ${type}: ${count} 个节点`);
        });
      }

      alert(
        `✅ FEN激活差异比较完成！\n\n` +
        `📊 统计:\n` +
        `- 总节点数: ${result.total_nodes}\n` +
        `- 未激活节点数: ${result.inactive_nodes_count}\n\n` +
        `💡 未激活的节点已在图中标记为金色`
      );

    } catch (error) {
      console.error('❌ 比较失败:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      addLog(`❌ 比较失败: ${errorMessage}`);
      alert(`比较失败: ${errorMessage}`);
    } finally {
      setIsComparingFens(false);
    }
  }, [originalCircuitJson, perturbedFen, linkGraphData, extractFenFromPrompt]);

  // 递归查找某个节点的所有上游节点（包括它们的上游）
  const findUpstreamNodes = useCallback((nodeId: string, graphData: any): Set<string> => {
    const upstreamNodes = new Set<string>();
    const visited = new Set<string>();
    
    const traverse = (currentNodeId: string) => {
      if (visited.has(currentNodeId)) return;
      visited.add(currentNodeId);
      upstreamNodes.add(currentNodeId);
      
      // 查找指向当前节点的所有边（入边）
      const incomingLinks = graphData.links.filter((link: any) => link.target === currentNodeId);
      
      // 递归查找每个源节点的上游
      for (const link of incomingLinks) {
        traverse(link.source);
      }
    };
    
    traverse(nodeId);
    return upstreamNodes;
  }, []);

  // 创建子图数据
  const createSubgraph = useCallback((rootNodeId: string, graphData: any) => {
    const upstreamNodeIds = findUpstreamNodes(rootNodeId, graphData);
    
    // 过滤节点：只保留上游节点
    const subgraphNodes = graphData.nodes.filter((node: any) => 
      upstreamNodeIds.has(node.nodeId)
    );
    
    // 过滤边：只保留两端都在子图中的边
    const subgraphLinks = graphData.links.filter((link: any) => 
      upstreamNodeIds.has(link.source) && upstreamNodeIds.has(link.target)
    );
    
    // 创建子图数据结构
    const subgraph = {
      nodes: subgraphNodes,
      links: subgraphLinks,
      metadata: {
        ...graphData.metadata,
        subgraphRoot: rootNodeId,
        originalNodeCount: graphData.nodes.length,
        subgraphNodeCount: subgraphNodes.length,
        originalLinkCount: graphData.links.length,
        subgraphLinkCount: subgraphLinks.length,
        createdAt: new Date().toISOString(),
        isSubgraph: true
      }
    };
    
    console.log('🔍 创建子图:', {
      rootNodeId,
      totalUpstreamNodes: upstreamNodeIds.size,
      subgraphNodes: subgraphNodes.length,
      subgraphLinks: subgraphLinks.length,
      originalNodes: graphData.nodes.length,
      originalLinks: graphData.links.length
    });
    
    return subgraph;
  }, [findUpstreamNodes]);

  // 显示子图
  const handleShowSubgraph = useCallback(() => {
    if (!clickedId || !displayLinkGraphData) return;
    
    const subgraph = createSubgraph(clickedId, displayLinkGraphData);
    setSubgraphData(subgraph);
    setSubgraphRootNodeId(clickedId);
    setShowSubgraph(true);
    
    console.log('🎯 显示子图模式:', {
      rootNodeId: clickedId,
      nodeCount: subgraph.nodes.length,
      linkCount: subgraph.links.length
    });
  }, [clickedId, displayLinkGraphData, createSubgraph]);

  // 退出子图模式
  const handleExitSubgraph = useCallback(() => {
    setShowSubgraph(false);
    setSubgraphData(null);
    setSubgraphRootNodeId(null);
    console.log('🔙 退出子图模式');
  }, []);

  // 保存子图为JSON文件
  const handleSaveSubgraph = useCallback(() => {
    if (!subgraphData || !subgraphRootNodeId) return;
    
    // 从原始数据中获取完整的节点信息（包括激活数据、z_pattern等）
    const enrichSubgraphWithOriginalData = (subgraph: any) => {
      if (!originalCircuitJson) return subgraph;
      
      const enrichedNodes = subgraph.nodes.map((node: any) => {
        // 从原始JSON中查找对应的完整节点数据
        const originalNodeData = getNodeActivationDataFromJson(originalCircuitJson, node.nodeId);
        
        return {
          ...node,
          // 添加原始数据中的完整信息
          activations: originalNodeData.activations,
          zPatternIndices: originalNodeData.zPatternIndices,
          zPatternValues: originalNodeData.zPatternValues,
          clerp: originalNodeData.clerp,
          // 保留所有原始字段
          ...(originalCircuitJson.nodes?.find((n: any) => n.node_id === node.nodeId) || {})
        };
      });
      
      return {
        ...subgraph,
        nodes: enrichedNodes
      };
    };
    
    const enrichedSubgraph = enrichSubgraphWithOriginalData(subgraphData);
    
    // 生成文件名
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const rootNodeForFilename = subgraphRootNodeId.replace(/[^a-zA-Z0-9]/g, '_');
    const fileName = `subgraph_${rootNodeForFilename}_${timestamp}.json`;
    
    // 创建下载
    const jsonString = JSON.stringify(enrichedSubgraph, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    console.log('💾 子图已保存:', {
      fileName,
      rootNodeId: subgraphRootNodeId,
      nodeCount: enrichedSubgraph.nodes.length,
      linkCount: enrichedSubgraph.links.length
    });
    
    alert(
      `✅ 子图已保存！\n\n` +
      `📁 文件名: ${fileName}\n` +
      `🎯 根节点: ${subgraphRootNodeId}\n` +
      `📊 统计:\n` +
      `  - 节点数: ${enrichedSubgraph.nodes.length}\n` +
      `  - 边数: ${enrichedSubgraph.links.length}\n` +
      `  - 包含完整激活数据和z_pattern信息\n\n` +
      `💡 文件已保存到Downloads文件夹`
    );
    
  }, [subgraphData, subgraphRootNodeId, originalCircuitJson, getNodeActivationDataFromJson]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-red-600 mb-2">Failed to load circuit visualization</h3>
          <p className="text-gray-600">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading circuit visualization...</p>
        </div>
      </div>
    );
  }

  if (!linkGraphData) {
    return (
      <div className="space-y-6">
        <SaeComboLoader />

        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Circuit Visualization</h2>
        </div>

        <FileUploadZone
          isDragOver={isDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onFileInput={handleFileInput}
        />
      </div>
    );
  }

  // 调试传递给ChessBoard的数据
  if (clickedId && nodeActivationData) {
    console.log('🎲 传递给ChessBoard的数据:', {
      nodeId: clickedId,
      hasActivations: !!nodeActivationData.activations,
      activationsLength: nodeActivationData.activations?.length || 0,
      hasZPatternIndices: !!nodeActivationData.zPatternIndices,
      hasZPatternValues: !!nodeActivationData.zPatternValues,
      nodeType: nodeActivationData.nodeType,
      hasClerp: !!nodeActivationData.clerp,
      clerpLength: nodeActivationData.clerp?.length || 0
    });
  }

  return (
    <div className="space-y-6 w-full max-w-full overflow-hidden">
      {/* 全局 BT4 SAE 组合选择（LoRSA / Transcoder） */}
      <SaeComboLoader />

      {/* Header */}
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex items-center space-x-2 min-w-0">
          <h2 className="text-l font-bold whitespace-nowrap">Prompt:</h2>
          <h2 className="text-l truncate">{displayLinkGraphData?.metadata?.prompt_tokens?.join(' ') || ''}</h2>
        </div>
        <div className="flex flex-wrap items-start justify-end gap-3 ml-auto">
          <button
            onClick={() => setLinkGraphData(null)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Upload New File
          </button>
          {/* Graph Feature Diffing 控件 - 只在单图时显示 */}
          {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-yellow-50 rounded-md border border-yellow-200">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-700">Perturb FEN:</label>
                <input
                  type="text"
                  value={perturbedFen}
                  onChange={(e) => setPerturbedFen(e.target.value)}
                  placeholder="Enter perturbed FEN..."
                  className="w-64 px-2 py-1 text-sm border border-gray-300 rounded"
                  disabled={isComparingFens}
                  title="输入扰动后的FEN字符串，用于比较激活差异"
                />
                <button
                  onClick={compareFenActivations}
                  disabled={isComparingFens || !perturbedFen.trim()}
                  className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="比较原始FEN和扰动FEN的激活差异"
                >
                  {isComparingFens ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                      Comparing...
                    </>
                  ) : (
                    'Compare Activation Differences'
                  )}
                </button>
                {inactiveNodes.size > 0 && (
                  <span className="text-sm text-yellow-700 font-medium">
                    {inactiveNodes.size} inactive nodes
                  </span>
                )}
                <button
                  onClick={() => setShowDiffingLogs(!showDiffingLogs)}
                  className="px-2 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  title="显示/隐藏比较日志"
                >
                  {showDiffingLogs ? 'Hide Logs' : 'Show Logs'}
                </button>
              </div>
            </div>
          )}
          {/* Clerp同步控件 */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-blue-50 rounded-md border border-blue-200">
            <button
              onClick={syncClerpsToBackend}
              disabled={syncingToBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="将JSON中所有节点的clerp同步到后端MongoDB的interpretation"
            >
              {syncingToBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Uploading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Upload Clerp
                </>
              )}
            </button>
            <button
              onClick={syncClerpsFromBackend}
              disabled={syncingFromBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="从后端MongoDB读取interpretation并同步到JSON节点的clerp"
            >
              {syncingFromBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Downloading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                  </svg>
                  Download Clerp
                </>
              )}
            </button>
          </div>
          
          {/* Dense Feature检查控件 */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-md">
            <label className="text-sm text-gray-700">Dense threshold:</label>
            <input
              type="number"
              value={denseThreshold}
              onChange={(e) => setDenseThreshold(e.target.value)}
              placeholder="No limit"
              className="w-24 px-2 py-1 text-sm border border-gray-300 rounded"
              title="激活次数阈值，空表示无限大（所有节点保留）"
            />
            <button
              onClick={checkDenseFeatures}
              disabled={checkingDenseFeatures}
              className="px-3 py-1 text-sm bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="检查哪些节点是dense feature"
            >
              {checkingDenseFeatures ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Checking...
                </>
              ) : (
                'Check Dense'
              )}
            </button>
            {denseNodes.size > 0 && (
              <span className="text-sm text-purple-700 font-medium">
                {denseNodes.size} dense nodes
              </span>
            )}
          </div>
          
          {/* 颜色-文件名图例（多文件时显示） */}
          {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="hidden md:flex items-center space-x-3 mr-4">
              {displayLinkGraphData.metadata.sourceFileNames.map((name, idx) => (
                <div key={idx} className="flex items-center space-x-1 text-xs">
                  <span
                    className="inline-block rounded-full"
                    style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                    title={name}
                  />
                  <span className="text-gray-600 truncate max-w-[140px]" title={name}>{name}</span>
                </div>
              ))}
            </div>
          )}

          {/* Position 映射高亮开关（多文件时显示） */}
          {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="flex flex-wrap items-center gap-2 px-3 py-1 bg-purple-50 rounded-md border border-purple-200">
              <label className="text-sm text-purple-800 font-medium flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={enablePositionMapping}
                  onChange={(e) => setEnablePositionMapping(e.target.checked)}
                />
                Position mapping highlight
              </label>
              <span className="text-xs text-purple-700">
                为每个文件选一个 pos（0-63），高亮“不同文件的不同 pos 上但同一 (layer, feature) 的节点”
              </span>
              {enablePositionMapping && (
                <span className="text-xs text-purple-700">
                  Current matches: <span className="font-semibold">{positionMappedCount}</span> nodes
                </span>
              )}
            </div>
          )}
          {hasUnsavedChanges && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-orange-100 text-orange-800 rounded-md text-sm">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span>Unsaved changes</span>
              <button
                onClick={handleQuickExport}
                className="ml-2 px-2 py-1 bg-orange-200 hover:bg-orange-300 text-orange-900 rounded text-xs transition-colors"
                title="立即导出所有更改"
              >
                Export
              </button>
            </div>
          )}
          {saveHistory.length > 0 && (
            <div className="relative group">
              <button className="px-3 py-1 text-sm bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors">
                Save History ({saveHistory.length})
              </button>
              <div className="absolute right-0 top-full mt-1 w-80 bg-white border rounded-lg shadow-lg z-10 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-3">
                  <h4 className="font-medium text-gray-900 mb-2">Recent changes:</h4>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {saveHistory.slice(-5).reverse().map((entry, index) => (
                      <div key={index} className="text-xs text-gray-600 p-2 bg-gray-50 rounded">
                        {entry}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chess Board Display - 单文件 */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="flex justify-center gap-6 mb-6">
          {/* Circuit棋盘状态 - 左侧 */}
          <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-center flex-1">
                Circuit Board State
                {clickedId && displayActivationData && (
                  <span className="text-sm font-normal text-blue-600 ml-2">
                    (节点: {clickedId}{displayActivationData.nodeType ? ` - ${displayActivationData.nodeType.toUpperCase()}` : ''})
                  </span>
                )}
              </h3>
              {clickedId && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setShowAllPositions(!showAllPositions)}
                    className={`px-3 py-1 text-sm rounded transition-colors ${
                      showAllPositions
                        ? 'bg-blue-500 text-white hover:bg-blue-600'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    title={showAllPositions ? '显示单个位置的激活' : '显示所有位置的激活（合并）'}
                  >
                    {showAllPositions ? 'Single Position Mode' : 'All Positions Mode'}
                  </button>
                </div>
              )}
            </div>
            {outputMove && (
              <div className="text-center mb-2 text-sm text-green-600 font-medium">
                Output move: {outputMove} 🎯
              </div>
            )}
            {clickedId && showAllPositions && loadingAllPositions && (
              <div className="text-center mb-2 text-sm text-blue-600">
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <span>Loading activation data from backend...</span>
                </div>
              </div>
            )}
            {clickedId && !showAllPositions && loadingBackendZPattern && (
              <div className="text-center mb-2 text-sm text-blue-600">
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <span>Computing z_pattern from backend...</span>
                </div>
              </div>
            )}
            {clickedId && displayActivationData && displayActivationData.activations && (
              <div className="text-center mb-2 text-sm text-purple-600">
                {showAllPositions ? (
                  <>
                    All positions merged: {displayActivationData.activations.filter((v: number) => v !== 0).length} non-zero activations
                  </>
                ) : (
                  <>
                    Activations: {displayActivationData.activations.filter((v: number) => v !== 0).length} non-zero
                    {displayActivationData.zPatternIndices && displayActivationData.zPatternValues && 
                      `, ${displayActivationData.zPatternValues.length} Z-pattern connections`
                    }
                  </>
                )}
              </div>
            )}
            <ChessBoard
              fen={fen}
              size="medium"
              showCoordinates={true}
              move={outputMove || undefined}
              activations={displayActivationData?.activations}
              zPatternIndices={displayActivationData?.zPatternIndices}
              zPatternValues={displayActivationData?.zPatternValues}
              flip_activation={Boolean(fen && fen.split(' ')[1] === 'b')}
              autoFlipWhenBlack={true}
              sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
              analysisName={displayActivationData?.nodeType || 'Circuit Node'}
              moveColor={(clickedId ? (displayLinkGraphData.nodes.find(n => n.nodeId === clickedId)?.nodeColor) : undefined) as any}
              showSelfPlay={true}
            />
          </div>

          {/* 自定义FEN分析 - 右侧 */}
          {clickedId && (() => {
            // 从 nodeId 解析出 layer 和 featureIndex
            const parts = clickedId.split('_');
            const rawLayer = parseInt(parts[0]) || 0;
            const featureIdx = parseInt(parts[1]) || 0;
            const layerIdx = Math.floor(rawLayer / 2);
            
            // 确定节点类型和对应的字典名
            const currentNode = displayLinkGraphData?.nodes.find(n => n.nodeId === clickedId);
            const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
            const dictionary = getDictionaryName(layerIdx, isLorsa);
            
            return (
              <CustomFenInput
                dictionary={dictionary}
                featureIndex={featureIdx}
                disabled={false}
              />
            );
          })()}
        </div>
      )}

      {/* PosFeatureCard - 位置 Feature 分析 */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="w-full max-w-6xl mx-auto mb-6">
          <div className="bg-white rounded-lg border shadow-sm p-4">
            <h3 className="text-lg font-semibold mb-4">Position Feature Analysis</h3>
            <div className="grid grid-cols-[auto_1fr_auto_auto_auto_auto] gap-4 items-center mb-4">
              <label className="font-bold">FEN:</label>
              <input
                type="text"
                value={fen}
                readOnly
                className="px-3 py-2 border rounded bg-gray-50"
              />
              <label className="font-bold">Layer:</label>
              <input
                type="number"
                min="0"
                max="14"
                value={posFeatureLayer}
                onChange={(e) => setPosFeatureLayer(parseInt(e.target.value) || 0)}
                className="w-20 px-3 py-2 border rounded"
              />
              <label className="font-bold">Positions:</label>
              <input
                type="text"
                placeholder="e.g. 36 or 16,20,34"
                value={posFeaturePositions}
                onChange={(e) => setPosFeaturePositions(e.target.value)}
                className="w-48 px-3 py-2 border rounded"
              />
              <label className="font-bold">Component:</label>
              <select
                value={posFeatureComponentType}
                onChange={(e) => setPosFeatureComponentType(e.target.value as "attn" | "mlp")}
                className="px-3 py-2 border rounded"
              >
                <option value="attn">Attention</option>
                <option value="mlp">MLP</option>
              </select>
            </div>
            {fen && posFeaturePositions.trim() && (() => {
              const parsedPositions = posFeaturePositions
                .split(",")
                .map((p) => parseInt(p.trim()))
                .filter((p) => !isNaN(p) && p >= 0 && p < 64);
              
              if (parsedPositions.length === 0) return null;
              
              return (
                <PosFeatureCard
                  fen={fen}
                  layer={posFeatureLayer}
                  positions={parsedPositions}
                  componentType={posFeatureComponentType}
                  saeComboId={
                    typeof window !== "undefined"
                      ? window.localStorage.getItem("bt4_sae_combo_id") || undefined
                      : undefined
                  }
                />
              );
            })()}
          </div>
        </div>
      )}

      {/* Graph Feature Diffing 日志显示 - 只在单图时显示 */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && showDiffingLogs && (
        <div className="w-full border rounded-lg overflow-hidden mb-6">
          <div className="bg-yellow-800 text-white px-4 py-2 flex items-center justify-between">
            <h3 className="font-semibold">FEN Activation Diff Log</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setDiffingLogs([])}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                Clear Log
              </button>
              <button
                onClick={() => setShowDiffingLogs(false)}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                Hide
              </button>
            </div>
          </div>
          <div 
            id="diffing-logs-container"
            className="bg-gray-900 text-green-400 p-4 font-mono text-sm max-h-64 overflow-y-auto"
          >
            <div className="space-y-1">
              {diffingLogs.length === 0 ? (
                <div className="text-gray-500">No logs yet...</div>
              ) : (
                diffingLogs.map((log, index) => (
                  <div key={index} className="whitespace-pre-wrap">
                    {log.message}
                  </div>
                ))
              )}
              {isComparingFens && (
                <div className="text-yellow-400 animate-pulse">Comparing...</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chess Board Display - 多文件：为每个源文件渲染一个棋盘，并按来源显示激活 */}
      {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
        <div className="space-y-4 mb-6">
          {/* 多文件：position 映射选择器 */}
          {enablePositionMapping && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="text-sm font-medium text-purple-900">Position 映射选择（每文件一个）</div>
                <div className="text-xs text-purple-700">
                  说明：先在下方输入 pos（草稿），再点击“应用映射”才会生效并刷新图（不会改变节点合并规则）
                </div>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => {
                    // 应用：将草稿写入真正的选择，并强制刷新一次图
                    setPositionMappingSelections(() => {
                      const next: Record<number, number> = {};
                      const names = displayLinkGraphData?.metadata?.sourceFileNames || [];
                      for (let i = 0; i < names.length; i++) {
                        const v = draftPositionMappingSelections[i];
                        next[i] = (typeof v === "number" && Number.isFinite(v)) ? Math.max(0, Math.min(63, v)) : 0;
                      }
                      return next;
                    });
                    setPositionMappingApplyNonce((x) => x + 1);
                  }}
                  className="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors"
                  title="将当前输入的 pos 应用到高亮逻辑，并刷新图"
                >
                  Apply Mapping
                </button>
                <button
                  onClick={() => {
                    // 重置草稿为已应用值
                    setDraftPositionMappingSelections((prev) => {
                      const names = displayLinkGraphData?.metadata?.sourceFileNames || [];
                      const next: Record<number, number> = { ...prev };
                      for (let i = 0; i < names.length; i++) {
                        const applied = positionMappingSelections[i];
                        next[i] = (typeof applied === "number" && Number.isFinite(applied)) ? applied : 0;
                      }
                      return next;
                    });
                  }}
                  className="px-3 py-1 text-sm bg-white text-purple-800 border border-purple-300 rounded hover:bg-purple-100 transition-colors"
                  title="撤销未应用的修改"
                >
                  Revert
                </button>
                <span className="text-xs text-purple-700">
                  Applied matches: <span className="font-semibold">{positionMappedCount}</span>
                </span>
              </div>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                {displayLinkGraphData.metadata.sourceFileNames.map((name: string, idx: number) => (
                  <div key={`pos-map-${idx}`} className="flex items-center gap-3 bg-white border rounded p-2">
                    <span
                      className="inline-block rounded-full"
                      style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                      title={name}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="text-xs text-gray-600 truncate" title={name}>{name}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-gray-700">pos</label>
                      <input
                        type="number"
                        min={0}
                        max={63}
                        className="w-20 px-2 py-1 text-sm border rounded"
                        value={draftPositionMappingSelections[idx] ?? positionMappingSelections[idx] ?? 0}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setDraftPositionMappingSelections((prev) => ({
                            ...prev,
                            [idx]: Number.isFinite(v) ? Math.max(0, Math.min(63, v)) : 0,
                          }));
                        }}
                        title="选择该文件用于对齐/高亮的 position（0-63）"
                      />
                      <span className="text-xs text-purple-700">↦ 高亮</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-xs text-purple-700">
                Highlight color: <span className="font-mono">{POSITION_MAPPING_HIGHLIGHT_COLOR}</span>
              </div>
            </div>
          )}

          {/* 所有位置模式切换按钮（多文件时） */}
          {clickedId && (
            <div className="flex justify-center">
              <button
                onClick={() => setShowAllPositions(!showAllPositions)}
                className={`px-4 py-2 text-sm rounded transition-colors ${
                  showAllPositions
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                title={showAllPositions ? '显示单个位置的激活' : '显示所有位置的激活（合并）'}
              >
                {showAllPositions ? '单位置模式' : '所有位置模式'}
              </button>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {multiOriginalJsons.map((entry, idx) => {
              const fileFen = extractFenFromCircuitJson(entry.json);
              if (!fileFen) return null;
              const fileMove = extractOutputMoveFromCircuitJson(entry.json);
              // 判断当前选中节点是否属于该文件
              const currentNode = clickedId ? displayLinkGraphData.nodes.find(n => n.nodeId === clickedId) : null;
              const belongs = currentNode && (currentNode.sourceIndices?.includes(idx) || currentNode.sourceIndex === idx);
              
              // 获取该文件的激活数据
              let perFileActivation: NodeActivationData = { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
              if (clickedId) {
                if (showAllPositions) {
                  // 在所有位置模式下，为每个文件都获取该feature的激活数据
                  const multiGraphData = multiGraphActivationData[idx];
                  if (multiGraphData) {
                    perFileActivation = multiGraphData;
                  } else {
                    // 回退到从JSON文件中同步获取数据
                    const allPosData = getAllPositionsActivationDataSync(clickedId, entry.json);
                    perFileActivation = allPosData || perFileActivation;
                  }
                } else {
                  // 获取单个位置的激活数据
                  perFileActivation = getNodeActivationDataFromJson(entry.json, clickedId);
                }
              }
              
              return (
                <div key={idx} className="bg-white rounded-lg border shadow-sm p-4 pb-8">
                  <h3 className="text-md font-semibold mb-3 flex items-center justify-center">
                    <span
                      className="inline-block rounded-full mr-2"
                      style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                      title={entry.fileName}
                    />
                    <span className="truncate" title={entry.fileName}>{entry.fileName}</span>
                    {clickedId && belongs && (
                      <span className="text-xs font-normal text-blue-600 ml-2">(contains node)</span>
                    )}
                  </h3>
                  {fileMove && (
                    <div className="text-center mb-2 text-sm text-green-600 font-medium">
                      输出移动: {fileMove} 🎯
                    </div>
                  )}
                  {clickedId && belongs && showAllPositions && loadingAllPositions && (
                    <div className="text-center mb-2 text-sm text-blue-600">
                      <div className="flex items-center justify-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                        <span>Loading activation data from backend...</span>
                      </div>
                    </div>
                  )}
                  {clickedId && belongs && perFileActivation.activations && (
                    <div className="text-center mb-2 text-sm text-purple-600">
                      {showAllPositions ? (
                        <>
                          All positions merged: {perFileActivation.activations.filter((v: number) => v !== 0).length} non-zero
                        </>
                      ) : (
                        <>
                          Activations: {perFileActivation.activations.filter((v: number) => v !== 0).length} non-zero
                          {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                            `, ${perFileActivation.zPatternValues.length} Z-pattern connections`}
                        </>
                      )}
                    </div>
                  )}
                  <ChessBoard
                    fen={fileFen}
                    size="medium"
                    showCoordinates={true}
                    move={fileMove || undefined}
                    activations={perFileActivation.activations}
                    zPatternIndices={perFileActivation.zPatternIndices}
                    zPatternValues={perFileActivation.zPatternValues}
                    flip_activation={Boolean(fileFen && fileFen.split(' ')[1] === 'b')}
                    autoFlipWhenBlack={true}
                    sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
                    analysisName={(perFileActivation?.nodeType || 'Circuit Node') + ` @${idx+1}`}
                    moveColor={UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length]}
                    showSelfPlay={true}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Circuit Visualization Layout */}
      <div className="space-y-6 w-full max-w-full overflow-hidden">
        {/* 子图模式控制栏 */}
        {clickedId && (
          <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-blue-900">选中节点:</span>
                <code className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-mono">
                  {clickedId}
                </code>
              </div>
              
              {!showSubgraph ? (
                <button
                  onClick={handleShowSubgraph}
                  className="px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors flex items-center"
                  title="显示以该节点为根的子图（包含所有上游节点）"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                  显示子图
                </button>
              ) : (
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2 text-sm text-green-700">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">子图模式</span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                      {subgraphData?.nodes.length || 0} 个节点
                    </span>
                  </div>
                  
                  <button
                    onClick={handleSaveSubgraph}
                    className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded hover:bg-green-600 transition-colors flex items-center"
                    title="保存子图为JSON文件（包含完整的激活数据和z_pattern）"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    保存子图
                  </button>
                  
                  <button
                    onClick={handleExitSubgraph}
                    className="px-3 py-1 bg-gray-500 text-white text-sm font-medium rounded hover:bg-gray-600 transition-colors flex items-center"
                    title="退出子图模式，显示完整图形"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    退出子图
                  </button>
                </div>
              )}
            </div>
            
            {showSubgraph && subgraphData && (
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex items-center space-x-2">
                  <span>根节点:</span>
                  <code className="px-1 bg-gray-100 rounded">{subgraphRootNodeId}</code>
                </div>
                <div className="flex items-center space-x-4">
                  <span>
                    节点: {subgraphData.nodes.length}/{subgraphData.metadata?.originalNodeCount || 0}
                    <span className="text-green-600 ml-1">
                      ({((subgraphData.nodes.length / (subgraphData.metadata?.originalNodeCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                  <span>
                    边: {subgraphData.links.length}/{subgraphData.metadata?.originalLinkCount || 0}
                    <span className="text-blue-600 ml-1">
                      ({((subgraphData.links.length / (subgraphData.metadata?.originalLinkCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Top Row: Link Graph and Node Connections side by side */}
        <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
          {/* Link Graph Component - Left Side */}
          <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <div className="w-full h-full overflow-hidden relative">
              {(showSubgraph ? subgraphData : displayLinkGraphData) && (
                <LinkGraphContainer 
                  key={`${showSubgraph ? "sub" : "full"}-${positionMappingApplyNonce}`}
                  data={showSubgraph ? subgraphData : displayLinkGraphData} 
                  onNodeClick={handleFeatureClick}
                  onNodeHover={handleFeatureHover}
                  onFeatureSelect={handleFeatureSelect}
                  onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                  onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  pinnedIds={pinnedIds}
                />
              )}
            </div>
          </div>

          {/* Node Connections Component - Right Side */}
          <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            {(showSubgraph ? subgraphData : displayLinkGraphData) && (
              <NodeConnections
                data={showSubgraph ? subgraphData : displayLinkGraphData}
                clickedId={clickedId}
                hoveredId={hoveredId}
                pinnedIds={pinnedIds}
                hiddenIds={hiddenIds}
                onFeatureClick={handleFeatureClick}
                onFeatureSelect={handleFeatureSelect}
                onFeatureHover={handleFeatureHover}
              />
            )}
          </div>
        </div>

        {/* Top Activation Section */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Top Activation 棋盘</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">节点: {clickedId}</span>
                {loadingTopActivations && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">加载中...</span>
                  </div>
                )}
              </div>
            </div>
            
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">正在获取 Top Activation 数据...</p>
                </div>
              </div>
            ) : topActivations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {topActivations.map((sample, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                    <div className="text-center mb-2">
                      <div className="text-sm font-medium text-gray-700">
                        Top #{index + 1}
                      </div>
                      <div className="text-xs text-gray-500">
                        最大激活值: {sample.activationStrength.toFixed(3)}
                      </div>
                    </div>
                    <ChessBoard
                      fen={sample.fen}
                      size="small"
                      showCoordinates={false}
                      activations={sample.activations}
                      zPatternIndices={sample.zPatternIndices}
                      zPatternValues={sample.zPatternValues}
                      sampleIndex={sample.sampleIndex}
                      analysisName={`Context ${sample.contextId}`}
                      flip_activation={Boolean(sample.fen && sample.fen.split(' ')[1] === 'b')}
                      autoFlipWhenBlack={true}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>未找到包含棋盘的激活样本</p>
              </div>
            )}
          </div>
        )}

        {/* Circuit Interpretation Section */}
        {clickedId && displayLinkGraphData && (() => {
          const currentNode = displayLinkGraphData.nodes.find(n => n.nodeId === clickedId);
          if (!currentNode) return null;
          
          const parts = clickedId.split('_');
          const rawLayer = parseInt(parts[0]) || 0;
          const featureIndex = parseInt(parts[1]) || 0;
          const layerIdx = Math.floor(rawLayer / 2);
          
          return (
            <CircuitInterpretationCard
              node={{
                nodeId: clickedId,
                layer: layerIdx,
                feature: featureIndex,
                feature_type: currentNode.feature_type || '',
              }}
              saeComboId={
                (typeof window !== 'undefined' 
                  ? window.localStorage.getItem("bt4_sae_combo_id") 
                  : null) || 'k_30_e_16'
              }
              saeSeries="BT4-exp128"
              getSaeName={getSaeNameForCircuit}
            />
          );
        })()}

        {/* Token Predictions Section (简化版) */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Token Predictions</h3>
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-600" htmlFor="steering-scale-input">steering_scale:</label>
                  <input
                    id="steering-scale-input"
                    type="number"
                    step={0.1}
                    className="w-24 px-2 py-1 border rounded text-sm"
                    value={steeringScaleInput}
                    onChange={(e) => {
                      const inputValue = e.target.value;
                      setSteeringScaleInput(inputValue);
                      const v = parseFloat(inputValue);
                      if (Number.isFinite(v)) {
                        setSteeringScale(v);
                      }
                    }}
                    onBlur={() => {
                      if (steeringScaleInput === '' || steeringScaleInput === '-') {
                        setSteeringScale(0);
                        setSteeringScaleInput('0');
                      }
                    }}
                    title="调节steering放大系数，支持负数输入"
                  />
                </div>
                <button
                  onClick={() => clickedId && fetchTokenPredictions(clickedId)}
                  disabled={loadingTokenPredictions || !clickedId || !fen}
                  className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="运行特征干预分析"
                >
                  {loadingTokenPredictions ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      分析中...
                    </>
                  ) : (
                    '开始分析'
                  )}
                </button>
                {loadingTokenPredictions && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">分析中...</span>
                  </div>
                )}
              </div>
            </div>

            {loadingTokenPredictions ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">正在运行特征干预分析...</p>
                </div>
              </div>
            ) : tokenPredictions ? (
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-3 border">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600">steering_scale:</span>
                      <span className="ml-1 font-medium">{Number(steeringScale).toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">合法移动数:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.total_legal_moves}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">平均概率差:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_prob_diff?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">平均Logit差:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_logit_diff?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">原始Value:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.original_value?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className={`text-gray-600 ${tokenPredictions.statistics?.value_diff > 0 ? 'text-green-600' : tokenPredictions.statistics?.value_diff < 0 ? 'text-red-600' : ''}`}>
                        Value变化:
                        <span className="ml-1 font-medium">
                          {tokenPredictions.statistics?.value_diff > 0 ? '+' : ''}{tokenPredictions.statistics?.value_diff?.toFixed(4)}
                        </span>
                      </span>
                    </div>
                  </div>
                </div>

                {/* 概率差异最大前5（增加最多） */}
                {tokenPredictions.promoting_moves && tokenPredictions.promoting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">概率差异最大（增加最多）Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.promoting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>排名: #{index + 1}</div>
                              <div>概率差: <span className="font-medium">{formatProbability(move.prob_diff)}</span></div>
                              <div>原始概率: {formatProbability(move.original_prob)}</div>
                              <div>修改后概率: {formatProbability(move.modified_prob)}</div>
                              <div>Logit差: {move.diff?.toFixed(4)}</div>
                              <div>原始Logit: {move.original_logit?.toFixed(4)}</div>
                              <div>修改后Logit: {move.modified_logit?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 概率差异最小前5（减少最多，负数最小） */}
                {tokenPredictions.inhibiting_moves && tokenPredictions.inhibiting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">概率差异最小（减少最多）Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.inhibiting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>排名: #{index + 1}</div>
                              <div>概率差: <span className="font-medium">{formatProbability(move.prob_diff)}</span></div>
                              <div>原始概率: {formatProbability(move.original_prob)}</div>
                              <div>修改后概率: {formatProbability(move.modified_prob)}</div>
                              <div>Logit差: {move.diff?.toFixed(4)}</div>
                              <div>原始Logit: {move.original_logit?.toFixed(4)}</div>
                              <div>修改后Logit: {move.modified_logit?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>点击"开始分析"按钮以运行Token Predictions分析</p>
                <p className="text-sm mt-2">请先在上方加载 TC/LoRSA 组合（SaeComboLoader）</p>
              </div>
            )}
          </div>
        )}

        {/* Feature Interpretation Editor - New Section */}
        {clickedId && nodeActivationData && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Feature Interpretation Editor</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">节点: {clickedId}</span>
                {nodeActivationData.nodeType && (
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
                    {nodeActivationData.nodeType.toUpperCase()}
                  </span>
                )}
              </div>
            </div>
            
            {/* 始终显示编辑器，无论clerp是否存在或为空 */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium text-gray-700">
                  Feature Interpretation (可编辑)
                  {nodeActivationData.clerp === undefined && (
                    <span className="text-xs text-gray-500 ml-2">(节点暂无interpretation字段，可新建)</span>
                  )}
                  {nodeActivationData.clerp === '' && (
                    <span className="text-xs text-gray-500 ml-2">(当前为空，可编辑)</span>
                  )}
                </label>
                <div className="text-xs text-gray-500">
                  字符数: {editingClerp.length}
                </div>
              </div>
              <textarea
                value={editingClerp}
                onChange={(e) => setEditingClerp(e.target.value)}
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder={
                  nodeActivationData.clerp === undefined 
                    ? "该节点暂无interpretation字段，您可以在此输入新的interpretation内容..." 
                    : "输入或编辑节点的interpretation内容..."
                }
              />
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setEditingClerp(nodeActivationData.clerp || '')}
                  className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  disabled={isSaving}
                >
                  重置
                </button>
                {(() => {
                  const isDisabled = isSaving || editingClerp.trim() === (nodeActivationData.clerp || '');
                  console.log('🔍 按钮状态调试:', {
                    isSaving,
                    editingClerpTrimmed: editingClerp.trim(),
                    nodeActivationDataClerp: nodeActivationData.clerp,
                    nodeActivationDataClerpOrEmpty: nodeActivationData.clerp || '',
                    isEqual: editingClerp.trim() === (nodeActivationData.clerp || ''),
                    isDisabled
                  });
                  
                  return (
                    <button
                      onClick={handleSaveClerp}
                      disabled={isDisabled}
                      className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                      title="保存更改并自动下载更新后的文件到Downloads文件夹"
                    >
                      {isSaving && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      )}
                      {isSaving ? '保存中...' : '保存并下载'}
                    </button>
                  );
                })()}
              </div>
              {editingClerp.trim() !== (nodeActivationData.clerp || '') && (
                <div className="text-xs text-orange-600 bg-orange-50 p-2 rounded">
                  ⚠️ 内容已修改，请点击"保存并下载"以保存更改
                </div>
              )}
              
              {/* 显示当前状态信息 */}
              <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                <div className="flex justify-between">
                  <span>
                    原始状态: {
                      nodeActivationData.clerp === undefined 
                        ? '无interpretation字段' 
                        : nodeActivationData.clerp === '' 
                          ? '空字符串' 
                          : `有内容 (${nodeActivationData.clerp.length} 字符)`
                    }
                  </span>
                  <span>
                    当前编辑: {editingClerp === '' ? '空' : `${editingClerp.length} 字符`}
                  </span>
                </div>
              </div>
              
              {/* 使用说明 */}
              <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                <div className="font-medium mb-1">💡 文件更新工作流程:</div>
                <ol className="list-decimal list-inside space-y-1 text-blue-700">
                  <li>编辑interpretation内容后点击"保存并下载"</li>
                  <li>更新后的文件会自动下载到Downloads文件夹</li>
                  <li>用新文件替换原文件，或重新拖拽到此页面</li>
                  <li>文件名包含时间戳，避免意外覆盖</li>
                </ol>
                <div className="mt-2 text-xs">
                  <strong>提示:</strong> 由于浏览器安全限制，无法直接修改原文件，但下载的文件包含所有更改。
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Row: Feature Card below Link Graph Container */}
        {clickedId && displayLinkGraphData && (() => {
          // 获取当前选中节点的信息
          const currentNode = displayLinkGraphData.nodes.find(node => node.nodeId === clickedId);
          
          if (!currentNode) {
            return null;
          }
          
          // 从node_id解析真正的feature ID (格式: layer_featureId_ctxIdx)
          // 注意：layer需要除以2得到实际的模型层数，因为M和A分别占一层
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
          
          // 调试节点连接信息
          console.log('🔍 节点连接调试:', {
            nodeId: currentNode.nodeId,
            hasSourceLinks: !!currentNode.sourceLinks,
            sourceLinksCount: currentNode.sourceLinks?.length || 0,
            hasTargetLinks: !!currentNode.targetLinks,
            targetLinksCount: currentNode.targetLinks?.length || 0,
            totalLinksInData: displayLinkGraphData.links.length
          });
          
          // 使用辅助函数获取字典名
          const dictionary = getDictionaryName(layerIdx, isLorsa);
          
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
              {selectedFeature ? (
                <FeatureCard feature={selectedFeature} />
              ) : (
                <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                  <div className="text-center">
                    <p className="text-gray-600">No feature is available for this node</p>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
      </div>

    </div>
  );
};

export default CircuitVisualization;
