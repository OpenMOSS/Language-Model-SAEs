import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { useCircuitState } from "@/contexts/AppStateContext";
import { LinkGraphContainer } from "./link-graph-container";
import { NodeConnections } from "./node-connections";
import { transformCircuitData, CircuitJsonData } from "./link-graph/utils";
import { Node } from "./link-graph/types";
import { Feature } from "@/types/feature";
import { FeatureCard } from "@/components/feature/feature-card";
import { ChessBoard } from "@/components/chess/chess-board";
import React from "react"; // Added missing import for React

// Node activation data type
interface NodeActivationData {
  activations?: number[];
  zPatternIndices?: any;
  zPatternValues?: number[];
  nodeType?: string;
  clerp?: string;
}

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

  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [isLoadingConnectedFeatures, setIsLoadingConnectedFeatures] = useState(false);
  const [originalCircuitJson, setOriginalCircuitJson] = useState<any>(null); // Store original JSON data (single or merged)
  const [editingClerp, setEditingClerp] = useState<string>(''); // Current clerp being edited
  const [isSaving, setIsSaving] = useState(false); // Save-in-progress state
  const [originalFileName, setOriginalFileName] = useState<string>(''); // Original filename (for single-file case)
  const [updateCounter, setUpdateCounter] = useState(0); // Counter to force re-computation
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // Whether there are unsaved changes
  const [saveHistory, setSaveHistory] = useState<string[]>([]); // Save history entries

  // Multi-graph support: store multiple original JSONs and filenames
  const [multiOriginalJsons, setMultiOriginalJsons] = useState<{ json: CircuitJsonData; fileName: string }[]>([]);

  // Color palette for nodes/edges that are unique to each graph (up to 4 graphs)
  const UNIQUE_GRAPH_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B"]; // blue, orange, green, red
  const DUPLICATE_FEATURE_ALT_COLOR = "#FF9800"; // reserved alt color for duplicate features (amber)

  // Merge multiple JSON graphs into a single LinkGraphData (nodes merged by node_id, edges by (source,target))
  const mergeGraphs = useCallback((jsons: CircuitJsonData[], fileNames?: string[]) => {
    // First convert each JSON into LinkGraphData
    const graphs = jsons.map(j => transformCircuitData(j));

    // Merge metadata (simple strategy: concatenate prompt_tokens and annotate number of sources)
    const mergedMetadata: any = {
      ...(graphs[0]?.metadata || {}),
      prompt_tokens: graphs.map((g, i) => `[#${i + 1}] ` + (g?.metadata?.prompt_tokens?.join(' ') || '')).filter(Boolean),
      sourceFileNames: fileNames && fileNames.length ? fileNames : undefined,
    };

    // Merge nodes
    type NodeAccum = {
      base: any; // Use one source node as base (keep feature_type etc.)
      presentIn: number[]; // Indices of graphs this node appears in
    };

    const nodeMap = new Map<string, NodeAccum>();

    graphs.forEach((g, gi) => {
      g.nodes.forEach((n: any) => {
        const key = n.nodeId;
        if (!nodeMap.has(key)) {
          nodeMap.set(key, { base: { ...n }, presentIn: [gi] });
        } else {
          const acc = nodeMap.get(key)!;
          // Merge optional fields (prefer non-null values)
          acc.base.localClerp = acc.base.localClerp ?? n.localClerp;
          acc.base.remoteClerp = acc.base.remoteClerp ?? n.remoteClerp;
          // Track all graph indices this node appears in
          if (!acc.presentIn.includes(gi)) acc.presentIn.push(gi);
        }
      });
    });

    // Assign node colors:
    // - presentIn.length > 1 (shared across multiple graphs): keep transformCircuitData feature_type color (acc.base.nodeColor)
    // - only in a single graph: override with UNIQUE_GRAPH_COLORS[graphIndex]
    let mergedNodes: any[] = [];
    nodeMap.forEach(({ base, presentIn }) => {
      const isShared = presentIn.length > 1;
      const isError = typeof base.feature_type === 'string' && base.feature_type.toLowerCase().includes('error');
      const nodeColor = isError
        ? '#95a5a6'
        : (isShared ? base.nodeColor : UNIQUE_GRAPH_COLORS[presentIn[0] % UNIQUE_GRAPH_COLORS.length]);
      const sourceIndices = presentIn.slice();
      const sourceFiles = (fileNames && fileNames.length)
        ? sourceIndices.map(i => fileNames[i]).filter(Boolean)
        : undefined;
      mergedNodes.push({
        ...base,
        nodeColor,
        sourceIndices,
        sourceIndex: sourceIndices.length === 1 ? sourceIndices[0] : undefined,
        sourceFiles,
      });
    });

    // Remove alternate color logic: in multi-file setting we keep only
    // - shared nodes: use type colors from transformCircuitData
    // - unique nodes: use per-file UNIQUE_GRAPH_COLORS[index]

    // Merge edges under (source,target) key:
    // - if shared across graphs: keep transform colors (sign controls red/green) and take max strokeWidth, sum/avg weights
    // - if unique to one graph: keep edge and still use sign-based color; we don't override with per-graph color here
    type LinkAccum = {
      sources: number[]; // Graph indices this edge appears in
      weightSum: number;
      maxStroke: number;
      color: string; // First sign-based color is sufficient
      pctInputSum: number;
      weightsBySource: Record<number, number>;
      pctBySource: Record<number, number>;
    };

    const linkKey = (s: string, t: string) => `${s}__${t}`;
    const linkMap = new Map<string, LinkAccum>();

    graphs.forEach((g, gi) => {
      (g.links || []).forEach((e: any) => {
        const k = linkKey(e.source, e.target);
        if (!linkMap.has(k)) {
          linkMap.set(k, {
            sources: [gi],
            weightSum: e.weight ?? 0,
            maxStroke: e.strokeWidth ?? 1,
            color: e.color,
            pctInputSum: e.pctInput ?? 0,
            weightsBySource: { [gi]: e.weight ?? 0 },
            pctBySource: { [gi]: e.pctInput ?? (Math.abs(e.weight ?? 0) * 100) },
          });
        } else {
          const acc = linkMap.get(k)!;
          if (!acc.sources.includes(gi)) acc.sources.push(gi);
          acc.weightSum += (e.weight ?? 0);
          acc.maxStroke = Math.max(acc.maxStroke, e.strokeWidth ?? 1);
          // Keep first sign-based color; no override needed
          acc.pctInputSum += (e.pctInput ?? 0);
          acc.weightsBySource[gi] = (acc.weightsBySource[gi] || 0) + (e.weight ?? 0);
          acc.pctBySource[gi] = (acc.pctBySource[gi] || 0) + (e.pctInput ?? (Math.abs(e.weight ?? 0) * 100));
        }
      });
    });

    const mergedLinks: any[] = [];
    linkMap.forEach((acc, k) => {
      const [source, target] = k.split("__");
      const isShared = acc.sources.length > 1;
      const avgWeight = acc.weightSum / acc.sources.length;
      const avgPct = acc.pctInputSum / acc.sources.length;
      // Follow transform's sign-based coloring: positive = green, negative = red
      const color = avgWeight > 0 ? "#4CAF50" : "#F44336";
      mergedLinks.push({
        source,
        target,
        pathStr: "",
        color: isShared ? color : color,
        strokeWidth: acc.maxStroke,
        weight: avgWeight,
        pctInput: avgPct,
        sources: acc.sources,
        weightsBySource: acc.weightsBySource,
        pctBySource: acc.pctBySource,
      });
    });

    // Rebuild sourceLinks/targetLinks for nodes
    const nodeById: Record<string, any> = {};
    mergedNodes.forEach(n => { nodeById[n.nodeId] = { ...n, sourceLinks: [], targetLinks: [] }; });
    mergedLinks.forEach(l => {
      if (nodeById[l.source]) nodeById[l.source].sourceLinks.push(l);
      if (nodeById[l.target]) nodeById[l.target].targetLinks.push(l);
    });

    const finalNodes = Object.values(nodeById);

    const mergedData: any = {
      nodes: finalNodes,
      links: mergedLinks,
      metadata: mergedMetadata,
    };

    return mergedData;
  }, []);

  const handleFeatureClick = useCallback((node: Node, isMetaKey: boolean) => {
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedIds.includes(node.nodeId)
        ? pinnedIds.filter(id => id !== node.nodeId)
        : [...pinnedIds, node.nodeId];
      setPinnedIds(newPinnedIds);
    } else {
      // Set clicked node
      setClickedId(node.nodeId === clickedId ? null : node.nodeId);
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
    setIsLoadingConnectedFeatures(false);
  }, []);

  const handleConnectedFeaturesLoading = useCallback((loading: boolean) => {
    setIsLoadingConnectedFeatures(loading);
  }, []);

  // Single-file upload (kept for backward compatibility)
  const handleSingleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('Please upload a JSON file');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const text = await file.text();
      const jsonData: CircuitJsonData = JSON.parse(text);
      // Basic transform
      const data = transformCircuitData(jsonData);
      // Inject source info (single file has index 0)
      let annotated = {
        ...data,
        nodes: data.nodes.map(n => ({
          ...n,
          sourceIndex: 0,
          sourceIndices: [0],
          sourceFiles: [file.name],
        })),
        metadata: {
          ...data.metadata,
          sourceFileNames: [file.name],
        }
      } as any;

      setLinkGraphData(annotated);
      // Reset circuit state when loading new data
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      setOriginalCircuitJson(jsonData); // Store original JSON data
      setOriginalFileName(file.name); // Store original filename
      setEditingClerp(''); // Reset editor state
      setHasUnsavedChanges(false); // Clear unsaved-changes flag
      setSaveHistory([]); // Clear save history
      setMultiOriginalJsons([{ json: jsonData, fileName: file.name }]);
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures]);

  // Multi-file upload (1–4 files)
  const handleMultiFilesUpload = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files).filter(f => f.name.endsWith('.json')).slice(0, 4);
    if (list.length === 0) {
      setError('Please upload 1-4 JSON files');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const texts = await Promise.all(list.map(f => f.text()));
      const jsons: CircuitJsonData[] = texts.map(t => JSON.parse(t));
      const fileNames = list.map(f => f.name);

      // Merge graphs
      const merged = jsons.length === 1 
        ? (() => {
            const data = transformCircuitData(jsons[0]);
            return {
              ...data,
              nodes: data.nodes.map(n => ({
                ...n,
                sourceIndex: 0,
                sourceIndices: [0],
                sourceFiles: [fileNames[0]],
              })),
              metadata: {
                ...data.metadata,
                sourceFileNames: [fileNames[0]],
              }
            };
          })()
        : mergeGraphs(jsons, fileNames);

      setLinkGraphData(merged);
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      setOriginalCircuitJson(jsons.length === 1 ? jsons[0] : merged); // Single graph keeps original JSON, multi-graph keeps merged JSON
      setOriginalFileName(list.length === 1 ? list[0].name : `merged_${list.length}_graphs.json`);
      setEditingClerp('');
      setHasUnsavedChanges(false);
      setSaveHistory([]);
      setMultiOriginalJsons(list.map((f, i) => ({ json: jsons[i], fileName: f.name })));
    } catch (err) {
      console.error('Failed to load circuit data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load circuit data');
    } finally {
      setLoading(false);
    }
  }, [mergeGraphs, setLinkGraphData, setLoading, setError, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures]);

  const handleFileUpload = useCallback(async (file: File) => {
    // Backward compatible: keep single-file path
    return handleSingleFileUpload(file);
  }, [handleSingleFileUpload]);

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

  // Extract FEN string from circuit data
  const extractFenFromPrompt = useCallback(() => {
    if (!linkGraphData?.metadata?.prompt_tokens) return null;
    
    const promptText = linkGraphData.metadata.prompt_tokens.join(' ');
    console.log('🔍 Searching for FEN string:', promptText);
    
    // More permissive FEN detection
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      // Check for FEN-like pattern – must contain slashes and enough content
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
          const boardRows = boardPart.split('/');
          
          if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
            console.log('✅ Found FEN string:', trimmed);
            return trimmed;
          }
        }
      }
    }
    
    // If no full FEN found, try a simpler match
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    if (simpleMatch) {
      console.log('✅ Found simple FEN match:', simpleMatch[0]);
      return simpleMatch[0];
    }
    
    console.log('❌ No FEN string found');
    return null;
  }, [linkGraphData]);

  // Extract FEN from a given circuit JSON (per-file)
  const extractFenFromCircuitJson = useCallback((json: any): string | null => {
    const tokens = json?.metadata?.prompt_tokens;
    if (!tokens) return null;
    const promptText = Array.isArray(tokens) ? tokens.join(' ') : String(tokens);
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const boardRows = parts[0].split('/');
          if (boardRows.length === 8 && /^[wb]$/.test(parts[1])) return trimmed;
        }
      }
    }
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    return simpleMatch ? simpleMatch[0] : null;
  }, []);
 
  // Extract output move from prompt
  const extractOutputMove = useCallback(() => {
    if (!linkGraphData) return null;

    // 1) Prefer metadata target_move or logit_moves[0]
    const tm = (linkGraphData as any)?.metadata?.target_move;
    if (typeof tm === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(tm)) {
      return tm.toLowerCase();
    }
    const lm0 = (linkGraphData as any)?.metadata?.logit_moves?.[0];
    if (typeof lm0 === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(lm0)) {
      return lm0.toLowerCase();
    }

    // 2) Fallback: parse from prompt_tokens
    if (!linkGraphData?.metadata?.prompt_tokens) return null;
    const promptText = linkGraphData.metadata.prompt_tokens.join(' ');
    console.log('🔍 Searching for output move:', promptText);

    const movePatterns = [
      /(?:Output|Move)[:：]\s*([a-h][1-8][a-h][1-8])/i,
      /\b([a-h][1-8][a-h][1-8])\b/g
    ];

    for (const pattern of movePatterns) {
      const matches = promptText.match(pattern);
      if (matches) {
        const lastMatch = Array.isArray(matches) ? matches[matches.length - 1] : matches;
        const moveMatch = lastMatch.match(/[a-h][1-8][a-h][1-8]/);
        if (moveMatch) {
          console.log('✅ Found move:', moveMatch[0]);
          return moveMatch[0].toLowerCase();
        }
      }
    }

    console.log('❌ No output move found');
    return null;
  }, [linkGraphData]);

  // Extract output move from a specific circuit JSON
  const extractOutputMoveFromCircuitJson = useCallback((json: any): string | null => {
    if (!json) return null;

    // 1) Prefer metadata target_move or logit_moves[0]
    const tm = json?.metadata?.target_move;
    if (typeof tm === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(tm)) {
      return tm.toLowerCase();
    }
    const lm0 = json?.metadata?.logit_moves?.[0];
    if (typeof lm0 === 'string' && /^[a-h][1-8][a-h][1-8]([qrbn])?$/i.test(lm0)) {
      return lm0.toLowerCase();
    }

    // 2) Fallback: parse from prompt_tokens
    const tokens = json?.metadata?.prompt_tokens;
    if (!tokens) return null;
    const promptText = Array.isArray(tokens) ? tokens.join(' ') : String(tokens);
    const patterns = [
      /(?:Output|Move)[:：]\s*([a-h][1-8][a-h][1-8])/i,
      /\b([a-h][1-8][a-h][1-8])\b/g
    ];
    for (const pattern of patterns) {
      const matches = promptText.match(pattern);
      if (matches) {
        const lastMatch = Array.isArray(matches) ? matches[matches.length - 1] : matches;
        const moveMatch = lastMatch.match(/[a-h][1-8][a-h][1-8]/);
        if (moveMatch) return moveMatch[0].toLowerCase();
      }
    }
    return null;
  }, []);
 
  // Improved getNodeActivationData helper
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    if (!nodeId || !originalCircuitJson) {
      console.log('❌ Missing required parameters:', { nodeId, hasOriginalCircuitJson: !!originalCircuitJson });
      return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    }
    
    console.log(`🔍 Looking up activation data for node ${nodeId}...`);
    console.log('📋 Original JSON structure:', {
      type: typeof originalCircuitJson,
      isArray: Array.isArray(originalCircuitJson),
      hasNodes: !!originalCircuitJson.nodes,
      nodesLength: originalCircuitJson.nodes?.length || 0,
      keys: Object.keys(originalCircuitJson)
    });
    
    // Parse node_id -> rawLayer, featureOrHead, ctx(position)
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const ctxIdx = Number(parts[2]) || 0;
      // Divide raw layer index by 2 to get actual model layer
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead, ctxIdx };
    };
    const parsed = parseFromNodeId(nodeId);

    // 1) First try matching directly in nodes array (if node objects inline activations/zPattern* fields)
    let nodesToSearch: any[] = [];
    if (originalCircuitJson.nodes && Array.isArray(originalCircuitJson.nodes)) {
      nodesToSearch = originalCircuitJson.nodes;
    } else if (Array.isArray(originalCircuitJson)) {
      nodesToSearch = originalCircuitJson;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray((originalCircuitJson as any)[key])) {
          nodesToSearch = (originalCircuitJson as any)[key];
          break;
        }
      }
      if (nodesToSearch.length === 0) {
        const values = Object.values(originalCircuitJson);
        const arrayValue = values.find(v => Array.isArray(v)) as any[] | undefined;
        if (arrayValue) nodesToSearch = arrayValue;
      }
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find(node => node?.node_id === nodeId);
      if (exactMatch) {
        const inlineActs = exactMatch.activations;
        const inlineZIdx = exactMatch.zPatternIndices;
        const inlineZVal = exactMatch.zPatternValues;
        console.log('✅ Inline node fields check:', {
          hasInlineActivations: !!inlineActs,
          hasInlineZIdx: !!inlineZIdx,
          hasInlineZVal: !!inlineZVal,
        });
        if (inlineActs || (inlineZIdx && inlineZVal)) {
          return {
            activations: inlineActs,
            zPatternIndices: inlineZIdx,
            zPatternValues: inlineZVal,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // 2) Based on node_id, match in activation records (layer/position/head_idx/feature_idx)
    // Build candidate records by scanning all arrays in originalCircuitJson and picking items with key fields
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const keys = Object.keys(item);
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
    pushCandidateArrays(originalCircuitJson);

    console.log('🧭 Candidate record count:', candidateRecords.length);

    // Matching function: choose head_idx vs feature_idx based on feature_type
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === parsed.ctxIdx;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && posOk && indexOk;
    };

    // Try to determine feature_type for this node from nodes array
    let featureTypeForNode: string | undefined = undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find(n => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find(rec => tryMatchRecord(rec, featureTypeForNode));
    if (matched) {
      console.log('✅ Found activation record via parsed match:', {
        nodeId,
        layerForActivation: parsed.layerForActivation,
        ctxIdx: parsed.ctxIdx,
        featureOrHead: parsed.featureOrHead,
        featureTypeForNode,
      });
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
      };
    }

    // 3) Fallback: fuzzy match using node_id prefix
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter(node => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        console.log('🔍 Using fuzzy node match:', {
          node_id: firstMatch.node_id,
          hasActivations: !!firstMatch.activations,
        });
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    console.log('❌ No matching node/record found');
    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, [originalCircuitJson, updateCounter]);

  const getNodeActivationDataFromJson = useCallback((jsonData: any, nodeId: string | null): NodeActivationData => {
    if (!nodeId || !jsonData) return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    const parts = nodeId.split('_');
    const rawLayer = Number(parts[0]) || 0;
    const featureOrHead = Number(parts[1]) || 0;
    const ctxIdx = Number(parts[2]) || 0;
    const layerForActivation = Math.floor(rawLayer / 2);

    // 1) First try a direct match in the nodes array
    let nodesToSearch: any[] = [];
    if (jsonData.nodes && Array.isArray(jsonData.nodes)) {
      nodesToSearch = jsonData.nodes;
    } else if (Array.isArray(jsonData)) {
      nodesToSearch = jsonData;
    } else {
      const possibleArrayKeys = ['data', 'features', 'items', 'activations'];
      for (const key of possibleArrayKeys) {
        if (Array.isArray(jsonData[key])) {
          nodesToSearch = jsonData[key];
          break;
        }
      }
      if (nodesToSearch.length === 0) {
        const values = Object.values(jsonData);
        const arrayValue = values.find(v => Array.isArray(v)) as any[] | undefined;
        if (arrayValue) nodesToSearch = arrayValue;
      }
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find((node: any) => node?.node_id === nodeId);
      if (exactMatch) {
        if (exactMatch.activations || (exactMatch.zPatternIndices && exactMatch.zPatternValues)) {
          return {
            activations: exactMatch.activations,
            zPatternIndices: exactMatch.zPatternIndices,
            zPatternValues: exactMatch.zPatternValues,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        }
      }
    }

    // 2) Deep-scan activation records and match by (layer/position/index)
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) candidateRecords.push(item);
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(jsonData);

    // Try to infer feature_type from nodes
    let featureTypeForNode: string | undefined;
    if (nodesToSearch.length > 0) {
      const nodeMeta = nodesToSearch.find((n: any) => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    const matched = candidateRecords.find((rec: any) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;
      const layerOk = !Number.isNaN(recLayer) && recLayer === layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === ctxIdx;
      let indexOk = false;
      if (featureTypeForNode) {
        const t = featureTypeForNode.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === featureOrHead;
        else if (t === 'cross layer transcoder') indexOk = recFeatIdx === featureOrHead;
        else indexOk = (recHead === featureOrHead) || (recFeatIdx === featureOrHead);
      } else {
        indexOk = (recHead === featureOrHead) || (recFeatIdx === featureOrHead);
      }
      return layerOk && posOk && indexOk;
    });

    if (matched) {
      const clerp = (nodesToSearch.find((n: any) => n?.node_id === nodeId) || {}).clerp;
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp,
      };
    }

    // 3) Fuzzy match
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter((node: any) => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, []);
 
  // Extract derived data for UI
  const fen = extractFenFromPrompt();
  const outputMove = extractOutputMove();
  const nodeActivationData = getNodeActivationData(clickedId);

  // Keep clerp editor in sync with selected node (top-level hook, no conditional usage)
  React.useEffect(() => {
    if (clickedId && nodeActivationData) {
      const clerpValue = nodeActivationData.clerp || '';
      console.log('🔄 Updating editor state:', {
        nodeId: clickedId,
        clerpValue,
        clerpType: typeof nodeActivationData.clerp,
        clerpLength: clerpValue.length,
        updateCounter
      });
      setEditingClerp(clerpValue);
    } else {
      console.log('🔄 Clearing editor state');
      setEditingClerp('');
    }
  }, [clickedId, nodeActivationData?.clerp, updateCounter]);

  const handleSaveClerp = useCallback(async () => {
    console.log('🚀 Starting clerp save:', {
      clickedId,
      hasOriginalCircuitJson: !!originalCircuitJson,
      editingClerp,
      editingClerpLength: editingClerp.length,
      trimmedLength: editingClerp.trim().length
    });
    
    if (!clickedId || !originalCircuitJson) {
      console.log('❌ Save failed: missing required data');
      return;
    }

    // Allow saving empty content as long as there is a change
    const trimmedClerp = editingClerp.trim();
    
    setIsSaving(true);
    
    try {
      // Deep-copy to avoid mutating original data
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // Find and update the matching node
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

      // Match by node_id and update node.clerp
      for (const node of nodesToSearch) {
        if (node && typeof node === 'object' && node.node_id === clickedId) {
          const previousClerp = node.clerp;
          node.clerp = trimmedClerp;
          updated = true;
          console.log('✅ Updated node clerp:', {
            node_id: clickedId,
            feature: node.feature,
            layer: node.layer,
            feature_type: node.feature_type,
            previousClerp: previousClerp || '(empty)',
            newClerp: trimmedClerp || '(empty)',
            newClerpLength: trimmedClerp.length
          });
          break;
        }
      }

      if (updated) {
        // Commit updated JSON back into state
        setOriginalCircuitJson(updatedCircuitJson);
        
        // Force refresh of node data
        setUpdateCounter(prev => prev + 1);
        
        // Mark unsaved changes
        setHasUnsavedChanges(true);
        
        console.log('✅ Local data updated, triggering re-render');
        console.log('🔍 Verification:', {
          nodeId: clickedId,
          updatedClerp: updatedCircuitJson.nodes?.find((n: any) => n.node_id === clickedId)?.clerp,
          updateCounter: updateCounter + 1
        });
        
        // Automatically download updated JSON file (derive from original filename)
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
        
        // Append entry to save history
        setSaveHistory(prev => [
          ...prev,
          `${new Date().toLocaleTimeString()}: node ${clickedId} - ${
            trimmedClerp.length === 0 ? 'cleared clerp' : `updated to: ${trimmedClerp.substring(0, 30)}...`
          }`
        ]);
        
        console.log('📥 Updated file downloaded:', updatedFileName);
        
        alert(
          `✅ Clerp saved and downloaded successfully! ${trimmedClerp.length === 0 ? '(saved as empty)' : ''}\n\n` +
          `📁 File saved to your Downloads folder:\n${updatedFileName}\n\n` +
          `💡 How to use the updated file:\n` +
          `1. Replace the original file with this updated one\n` +
          `2. Or upload the new file to this page again\n` +
          `3. The filename includes a timestamp to avoid accidental overwrites`
        );
        
      } else {
        throw new Error(`Could not find node data for node_id: ${clickedId}`);
      }
    } catch (err) {
      console.error('Save failed:', err);
      alert('Save failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setIsSaving(false);
    }
  }, [clickedId, originalCircuitJson, editingClerp, originalFileName, setOriginalCircuitJson, updateCounter]);

  // Quick export of current JSON state
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
    console.log('📤 Quick export completed:', exportFileName);
    alert(
      `📤 File exported to your Downloads folder:\n${exportFileName}\n\n` +
      `💡 To use the updated file:\n` +
      `1. Replace the original file with this one\n` +
      `2. Or drag-and-drop the new file onto this page to reload`
    );
  }, [originalCircuitJson, originalFileName]);

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
        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Circuit Visualization</h2>
        </div>

        {/* Upload Interface */}
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
                Upload Circuit Data
              </h3>
              <p className="text-gray-600 mb-4">
                Drag and drop 1-4 JSON files here, or click to browse
              </p>
              <input
                type="file"
                accept=".json"
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
                multiple
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 cursor-pointer transition-colors"
              >
                Choose Files
              </label>
            </div>
            <p className="text-sm text-gray-500">
              Supports uploading multiple JSON files (1-4) to merge graphs
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Debug data passed to ChessBoard
  if (clickedId && nodeActivationData) {
    console.log('🎲 Data passed to ChessBoard:', {
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
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <h2 className="text-l font-bold">Prompt:</h2>
          <h2 className="text-l">{linkGraphData.metadata.prompt_tokens.join(' ')}</h2>
        </div>
        <div className="flex items-center space-x-2">
          {/* Color legend for each source file (multi-file only) */}
          {linkGraphData.metadata.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="hidden md:flex items-center space-x-3 mr-4">
              {linkGraphData.metadata.sourceFileNames.map((name, idx) => (
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
          {hasUnsavedChanges && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-orange-100 text-orange-800 rounded-md text-sm">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span>There are unsaved changes</span>
              <button
                onClick={handleQuickExport}
                className="ml-2 px-2 py-1 bg-orange-200 hover:bg-orange-300 text-orange-900 rounded text-xs transition-colors"
                title="Export all changes now"
              >
                Export
              </button>
            </div>
          )}
          {saveHistory.length > 0 && (
            <div className="relative group">
              <button className="px-3 py-1 text-sm bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors">
                Save history ({saveHistory.length})
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
          <button
            onClick={() => setLinkGraphData(null)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Upload New File
          </button>
        </div>
      </div>

      {/* Chess Board Display - single-file */}
      {(!linkGraphData.metadata.sourceFileNames || linkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="flex justify-center mb-6">
          <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
            <h3 className="text-lg font-semibold mb-4 text-center">
              Circuit Board State
              {clickedId && nodeActivationData && (
                <span className="text-sm font-normal text-blue-600 ml-2">
                  (Node: {clickedId}{nodeActivationData.nodeType ? ` - ${nodeActivationData.nodeType.toUpperCase()}` : ''})
                </span>
              )}
            </h3>
            {outputMove && (
              <div className="text-center mb-2 text-sm text-green-600 font-medium">
                Output move: {outputMove} 🎯
              </div>
            )}
            {clickedId && nodeActivationData && nodeActivationData.activations && (
              <div className="text-center mb-2 text-sm text-purple-600">
                Activations: {nodeActivationData.activations.filter((v: number) => v !== 0).length} non-zero activations
                {nodeActivationData.zPatternIndices && nodeActivationData.zPatternValues && 
                  `, ${nodeActivationData.zPatternValues.length} Z-pattern connections`
                }
              </div>
            )}
            <ChessBoard
              fen={fen}
              size="medium"
              showCoordinates={true}
              move={outputMove || undefined}
              activations={nodeActivationData?.activations}
              zPatternIndices={nodeActivationData?.zPatternIndices}
              zPatternValues={nodeActivationData?.zPatternValues}
              flip_activation={false}
              sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
              analysisName={nodeActivationData?.nodeType || 'Circuit Node'}
              moveColor={(clickedId ? (linkGraphData.nodes.find(n => n.nodeId === clickedId)?.nodeColor) : undefined) as any}
            />
          </div>
        </div>
      )}

      {/* Chess Board Display - multi-file: one board per source file with per-source activations */}
      {linkGraphData.metadata.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {multiOriginalJsons.map((entry, idx) => {
            const fileFen = extractFenFromCircuitJson(entry.json);
            if (!fileFen) return null;
            const fileMove = extractOutputMoveFromCircuitJson(entry.json);
            // Check whether the selected node belongs to this file
            const currentNode = clickedId ? linkGraphData.nodes.find(n => n.nodeId === clickedId) : null;
            const belongs = currentNode && (currentNode.sourceIndices?.includes(idx) || currentNode.sourceIndex === idx);
            const perFileActivation = (clickedId && belongs)
              ? getNodeActivationDataFromJson(entry.json, clickedId)
              : { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
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
                    <span className="text-xs font-normal text-blue-600 ml-2">(contains this node)</span>
                  )}
                </h3>
                {fileMove && (
                  <div className="text-center mb-2 text-sm text-green-600 font-medium">
                    Output move: {fileMove} 🎯
                  </div>
                )}
                {clickedId && belongs && perFileActivation.activations && (
                  <div className="text-center mb-2 text-sm text-purple-600">
                    Activations: {perFileActivation.activations.filter((v: number) => v !== 0).length} non-zero activations
                    {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                      `, ${perFileActivation.zPatternValues.length} Z-pattern connections`}
                  </div>
                )}
                <ChessBoard
                  fen={fileFen}
                  size="medium"
                  showCoordinates={true}
                  move={fileMove || undefined}
                  activations={belongs ? perFileActivation.activations : undefined}
                  zPatternIndices={belongs ? perFileActivation.zPatternIndices : undefined}
                  zPatternValues={belongs ? perFileActivation.zPatternValues : undefined}
                  flip_activation={false}
                  sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
                  analysisName={(perFileActivation?.nodeType || 'Circuit Node') + ` @${idx+1}`}
                  moveColor={UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length]}
                />
              </div>
            );
          })}
        </div>
      )}

      {/* Circuit Visualization Layout */}
      <div className="space-y-6 w-full max-w-full overflow-hidden">
        {/* Top Row: Link Graph and Node Connections side by side */}
        <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
          {/* Link Graph Component - Left Side */}
          <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <div className="w-full h-full overflow-hidden relative">
              <LinkGraphContainer 
                data={linkGraphData} 
                onNodeClick={handleFeatureClick}
                onNodeHover={handleFeatureHover}
                onFeatureSelect={handleFeatureSelect}
                onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                clickedId={clickedId}
                hoveredId={hoveredId}
                pinnedIds={pinnedIds}
              />
            </div>
          </div>

          {/* Node Connections Component - Right Side */}
          <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <NodeConnections
              data={linkGraphData}
              clickedId={clickedId}
              hoveredId={hoveredId}
              pinnedIds={pinnedIds}
              hiddenIds={hiddenIds}
              onFeatureClick={handleFeatureClick}
              onFeatureSelect={handleFeatureSelect}
              onFeatureHover={handleFeatureHover}
            />
          </div>
        </div>

        {/* Clerp Editor - New Section */}
        {clickedId && nodeActivationData && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Node Clerp Editor</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Node: {clickedId}</span>
                {nodeActivationData.nodeType && (
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
                    {nodeActivationData.nodeType.toUpperCase()}
                  </span>
                )}
              </div>
            </div>
            
            {/* Always show the editor, regardless of whether clerp exists or is empty */}
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium text-gray-700">
                  Clerp content (editable)
                  {nodeActivationData.clerp === undefined && (
                    <span className="text-xs text-gray-500 ml-2">(This node has no clerp field yet — you can create one.)</span>
                  )}
                  {nodeActivationData.clerp === '' && (
                    <span className="text-xs text-gray-500 ml-2">(Currently empty — you can edit it.)</span>
                  )}
                </label>
                <div className="text-xs text-gray-500">
                  Characters: {editingClerp.length}
                </div>
              </div>
              <textarea
                value={editingClerp}
                onChange={(e) => setEditingClerp(e.target.value)}
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder={
                  nodeActivationData.clerp === undefined 
                    ? "This node has no clerp field yet; you can enter new clerp content here..." 
                    : "Enter or edit the clerp content for this node..."
                }
              />
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setEditingClerp(nodeActivationData.clerp || '')}
                  className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  disabled={isSaving}
                >
                  Reset
                </button>
                {(() => {
                  const isDisabled = isSaving || editingClerp.trim() === (nodeActivationData.clerp || '');
                  console.log('🔍 Button state debug:', {
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
                      title="Save changes and automatically download the updated file to your Downloads folder"
                    >
                      {isSaving && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      )}
                      {isSaving ? 'Saving...' : 'Save & Download'}
                    </button>
                  );
                })()}
              </div>
              {editingClerp.trim() !== (nodeActivationData.clerp || '') && (
                <div className="text-xs text-orange-600 bg-orange-50 p-2 rounded">
                  ⚠️ Content has changed. Click "Save & Download" to persist your edits.
                </div>
              )}
              
              {/* Current state summary */}
              <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                <div className="flex justify-between">
                  <span>
                    Original state:{' '}
                    {
                      nodeActivationData.clerp === undefined 
                        ? 'no clerp field' 
                        : nodeActivationData.clerp === '' 
                          ? 'empty string' 
                          : `has content (${nodeActivationData.clerp.length} chars)`
                    }
                  </span>
                  <span>
                    Current edit: {editingClerp === '' ? 'empty' : `${editingClerp.length} chars`}
                  </span>
                </div>
              </div>
              
              {/* Usage instructions */}
              <div className="text-xs text-blue-600 bg-blue-50 p-3 rounded border-l-4 border-blue-200">
                <div className="font-medium mb-1">💡 File update workflow:</div>
                <ol className="list-decimal list-inside space-y-1 text-blue-700">
                  <li>Edit the clerp content and click "Save & Download".</li>
                  <li>The updated file will be downloaded to your Downloads folder.</li>
                  <li>Replace the original file with the new one, or drag-and-drop it back onto this page.</li>
                  <li>The filename includes a timestamp to avoid accidental overwrites.</li>
                </ol>
                <div className="mt-2 text-xs">
                  <strong>Note:</strong> Because of browser security restrictions, the original file cannot be modified in place, but the downloaded file contains all changes.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Row: Feature Card below Link Graph Container */}
        {clickedId && (() => {
          // Get info for currently selected node
          const currentNode = linkGraphData.nodes.find(node => node.nodeId === clickedId);
          
          if (!currentNode) {
            return null;
          }
          
          // Parse feature ID from nodeId (format: layer_featureId_ctxIdx)
          // Note: divide layer index by 2 to get actual model layer (M and A occupy separate layers)
          const parseNodeId = (nodeId: string) => {
            const parts = nodeId.split('_');
            if (parts.length >= 2) {
              const rawLayer = parseInt(parts[0]) || 0;
              return {
                layerIdx: Math.floor(rawLayer / 2),
                featureIndex: parseInt(parts[1]) || 0
              };
            }
            return { layerIdx: 0, featureIndex: 0 };
          };
          
          const { layerIdx, featureIndex } = parseNodeId(currentNode.nodeId);
          const isLorsa = currentNode.feature_type?.toLowerCase() === 'lorsa';
          
          // Debug node connectivity
          console.log('🔍 Node connectivity debug:', {
            nodeId: currentNode.nodeId,
            hasSourceLinks: !!currentNode.sourceLinks,
            sourceLinksCount: currentNode.sourceLinks?.length || 0,
            hasTargetLinks: !!currentNode.targetLinks,
            targetLinksCount: currentNode.targetLinks?.length || 0,
            totalLinksInData: linkGraphData.links.length
          });
          
          // Build dictionary name according to node type
          const dictionary = isLorsa 
            ? (linkGraphData?.metadata?.lorsa_analysis_name && linkGraphData.metadata.lorsa_analysis_name.includes('BT4')
                ? `BT4_lorsa_L${layerIdx}A` 
                : `lc0-lorsa-L${layerIdx}`)
            : (linkGraphData?.metadata?.tc_analysis_name && linkGraphData.metadata.tc_analysis_name.includes('BT4')
                ? `BT4_tc_L${layerIdx}M` 
                : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`);
          
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
                  {/* Link to Feature page */}
                  {currentNode && featureIndex !== undefined && (
                    <Link
                      to={`/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${featureIndex}`}
                      className="inline-flex items-center px-3 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors"
                      title={`Go to L${layerIdx} ${nodeTypeDisplay} feature #${featureIndex}`}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                      View L{layerIdx} {nodeTypeDisplay} #{featureIndex}
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
