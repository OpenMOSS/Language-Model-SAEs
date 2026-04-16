import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { Node, LinkGraphData } from './link-graph/types';
import { fetchFeature } from '@/utils/api';
import { Feature } from '@/types/feature';

// This file is for Interaction Circuit Page, which is used to display the feature connections of the interaction circuit.

interface FeatureConnectionsProps {
  data: LinkGraphData;
  clickedId: string | null;
  hoveredId: string | null;
  onFeatureClick: (node: Node) => void;
  onFeatureHover: (nodeId: string | null) => void;
  getDictionaryNameForNode: (layer: number, isLorsa: boolean) => string;
}

interface FeatureWithInterpretation {
  node: Node;
  interpretation: string | null;
  feature: Feature | null;
  loading: boolean;
  editing: boolean;
  editText: string;
}

export const FeatureConnections: React.FC<FeatureConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  onFeatureClick,
  onFeatureHover,
  getDictionaryNameForNode,
}) => {
  const [featureInterpretations, setFeatureInterpretations] = useState<Map<string, FeatureWithInterpretation>>(new Map());
  const [savingNodeId, setSavingNodeId] = useState<string | null>(null);
  const [syncingAllInterpretations, setSyncingAllInterpretations] = useState(false);
  const loadedNodesRef = useRef<Set<string>>(new Set());

  // Find clicked node
  const clickedNode = useMemo(() => 
    data.nodes.find(node => node.nodeId === clickedId),
    [data.nodes, clickedId]
  );

  // Calculate Input Features and Output Features
  const { inputNodes, outputNodes } = useMemo(() => {
    if (!clickedNode) {
      return { inputNodes: [], outputNodes: [] };
    }

    // Input Features: all nodes pointing to current node (i.e., source of targetLinks)
    // First find source node IDs of all links pointing to current node
    const inputNodeIds = clickedNode.targetLinks?.map(link => link.source) || [];
    const inputNodes = data.nodes.filter(node => 
      node.nodeId !== clickedNode.nodeId &&
      inputNodeIds.includes(node.nodeId)
    );

    // Output Features: all nodes that current node points to (i.e., target of sourceLinks)
    // First find target node IDs of all links that current node points to
    const outputNodeIds = clickedNode.sourceLinks?.map(link => link.target) || [];
    const outputNodes = data.nodes.filter(node =>
      node.nodeId !== clickedNode.nodeId &&
      outputNodeIds.includes(node.nodeId)
    );

    return { inputNodes, outputNodes };
  }, [data.nodes, clickedNode]);

  // Format node ID display
  const formatNodeId = useCallback((node: Node): string => {
    const parts = node.nodeId.split('_');
    if (parts.length < 3) return node.nodeId;
    
    const layer = parseInt(parts[0]) || 0;
    const feature = parseInt(parts[2]) || 0;
    const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
    
    return `${isLorsa ? 'A' : 'M'}${layer}#${feature}@${parts[1]}`;
  }, []);

  // Get node interpretation
  const fetchNodeInterpretation = useCallback(async (node: Node) => {
    const nodeId = node.nodeId;
    
    // Skip if already loaded
    if (loadedNodesRef.current.has(nodeId)) {
      return;
    }

    // Mark as loading
    loadedNodesRef.current.add(nodeId);

    // Set loading state
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      newMap.set(nodeId, {
        node,
        interpretation: null,
        feature: null,
        loading: true,
        editing: false,
        editText: '',
      });
      return newMap;
    });

    try {
      // Extract layer and feature from nodeId
      const parts = nodeId.split('_');
      const layer = parseInt(parts[0]) || 0;
      const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(parts[2]) || 0);
      const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
      
      const dictionary = getDictionaryNameForNode(layer, isLorsa);
      
      // fetchFeature requires analysisName (dictionary name), layer, and featureId
      // Note: fetchFeature internally handles {} replacement, but the dictionary we pass is already a complete name
      const feature = await fetchFeature(dictionary, layer, featureIndex);
      
      const interpretation = feature?.interpretation?.text || null;
      
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        newMap.set(nodeId, {
          node,
          interpretation,
          feature,
          loading: false,
          editing: false,
          editText: interpretation || '',
        });
        return newMap;
      });
    } catch (error) {
      console.error('Failed to fetch feature:', error);
      loadedNodesRef.current.delete(nodeId); // Loading failed, allow retry
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        newMap.set(nodeId, {
          node,
          interpretation: null,
          feature: null,
          loading: false,
          editing: false,
          editText: '',
        });
        return newMap;
      });
    }
  }, [getDictionaryNameForNode]);

  // Get interpretation when clickedId, inputNodes, or outputNodes change
  useEffect(() => {
    if (!clickedId) {
      loadedNodesRef.current.clear();
      setFeatureInterpretations(new Map());
      return;
    }
    
    // Get interpretation for clicked node itself
    if (clickedNode) {
      fetchNodeInterpretation(clickedNode);
    }
    
    // Get interpretations for input and output nodes
    const allNodes = [...inputNodes, ...outputNodes];
    allNodes.forEach(node => {
      fetchNodeInterpretation(node);
    });
  }, [inputNodes, outputNodes, clickedId, clickedNode, fetchNodeInterpretation]);

  // Save interpretation
  const saveInterpretation = useCallback(async (nodeId: string) => {
    const item = featureInterpretations.get(nodeId);
    if (!item || !item.feature) return;

    setSavingNodeId(nodeId);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${item.feature.dictionaryName}/features/${item.feature.featureIndex}/interpret?type=custom&custom_interpretation=${encodeURIComponent(item.editText)}`,
        {
          method: 'POST',
        }
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      // Update local state
      setFeatureInterpretations(prev => {
        const newMap = new Map(prev);
        const existing = newMap.get(nodeId);
        if (existing) {
          newMap.set(nodeId, {
            ...existing,
            interpretation: item.editText,
            editing: false,
          });
        }
        return newMap;
      });
    } catch (error) {
      console.error('Failed to save interpretation:', error);
      alert(`Save failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setSavingNodeId(null);
    }
  }, [featureInterpretations]);

  // Start editing
  const startEditing = useCallback((nodeId: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editing: true,
          editText: existing.interpretation || '',
        });
      }
      return newMap;
    });
  }, []);

  // Cancel editing
  const cancelEditing = useCallback((nodeId: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editing: false,
          editText: existing.interpretation || '',
        });
      }
      return newMap;
    });
  }, []);

  // Update edit text
  const updateEditText = useCallback((nodeId: string, text: string) => {
    setFeatureInterpretations(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(nodeId);
      if (existing) {
        newMap.set(nodeId, {
          ...existing,
          editText: text,
        });
      }
      return newMap;
    });
  }, []);

  // Batch sync interpretations for all nodes
  const syncAllInterpretations = useCallback(async () => {
    if (!data || !data.nodes.length) {
      alert('⚠️ No available node data');
      return;
    }

    setSyncingAllInterpretations(true);
    try {
      // Clear cache for all loaded nodes
      loadedNodesRef.current.clear();
      setFeatureInterpretations(new Map());

      // Batch fetch interpretations for all nodes (use Promise.all for parallel requests, but limit concurrency)
      const allNodes = data.nodes;
      const batchSize = 10; // Process 10 nodes per batch
      let foundCount = 0;
      let notFoundCount = 0;

      console.log('🔄 Starting batch sync of all node interpretations:', {
        totalNodes: allNodes.length,
      });

      // Process in batches to avoid too many concurrent requests
      for (let i = 0; i < allNodes.length; i += batchSize) {
        const batch = allNodes.slice(i, i + batchSize);
        await Promise.all(
          batch.map(async (node) => {
            try {
              const parts = node.nodeId.split('_');
              const layer = parseInt(parts[0]) || 0;
              const featureIndex = node.featureIndex !== undefined ? node.featureIndex : (parseInt(parts[2]) || 0);
              const isLorsa = node.feature_type?.toLowerCase() === 'lorsa';
              
              const dictionary = getDictionaryNameForNode(layer, isLorsa);
              const feature = await fetchFeature(dictionary, layer, featureIndex);
              
              if (feature?.interpretation?.text) {
                foundCount++;
                setFeatureInterpretations(prev => {
                  const newMap = new Map(prev);
                  newMap.set(node.nodeId, {
                    node,
                    interpretation: feature.interpretation!.text!,
                    feature,
                    loading: false,
                    editing: false,
                    editText: feature.interpretation!.text!,
                  });
                  return newMap;
                });
              } else {
                notFoundCount++;
                setFeatureInterpretations(prev => {
                  const newMap = new Map(prev);
                  newMap.set(node.nodeId, {
                    node,
                    interpretation: null,
                    feature,
                    loading: false,
                    editing: false,
                    editText: '',
                  });
                  return newMap;
                });
              }
            } catch (error) {
              console.error(`❌ Failed to get interpretation for node ${node.nodeId}:`, error);
              notFoundCount++;
            }
          })
        );

        // Show progress
        console.log(`✅ Processed ${Math.min(i + batchSize, allNodes.length)}/${allNodes.length} nodes`);
      }

      console.log('✅ Batch sync completed:', {
        total: allNodes.length,
        found: foundCount,
        notFound: notFoundCount
      });

      alert(`✅ Sync completed! Found ${foundCount} interpretations, ${notFoundCount} not found.\n\nNote: Interpretations have been synced from backend MongoDB, please click nodes to view.`);
    } catch (error) {
      console.error('❌ Batch sync failed:', error);
      alert(`❌ Sync failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setSyncingAllInterpretations(false);
    }
  }, [data, getDictionaryNameForNode]);

  // Render feature row
  const renderFeatureRow = useCallback((node: Node) => {
    const item = featureInterpretations.get(node.nodeId);
    const isHovered = node.nodeId === hoveredId;
    const isEditing = item?.editing || false;
    const interpretation = item?.interpretation || null;
    const loading = item?.loading || false;
    const editText = item?.editText || '';

    return (
      <div
        key={node.nodeId}
        className={`p-2 border rounded mb-2 cursor-pointer transition-colors ${
          isHovered ? 'bg-blue-50 border-blue-300' : 'bg-gray-50 border-gray-200'
        }`}
        onClick={() => onFeatureClick(node)}
        onMouseEnter={() => onFeatureHover(node.nodeId)}
        onMouseLeave={() => onFeatureHover(null)}
      >
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="text-sm font-mono text-gray-700 mb-1">
              {formatNodeId(node)}
            </div>
            {loading ? (
              <div className="text-xs text-gray-500">Loading...</div>
            ) : isEditing ? (
              <div className="mt-2">
                <textarea
                  value={editText}
                  onChange={(e) => updateEditText(node.nodeId, e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded text-sm"
                  rows={2}
                  onClick={(e) => e.stopPropagation()}
                />
                <div className="flex gap-2 mt-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      saveInterpretation(node.nodeId);
                    }}
                    disabled={savingNodeId === node.nodeId}
                    className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 disabled:opacity-50"
                  >
                    {savingNodeId === node.nodeId ? 'Saving...' : 'Save'}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      cancelEditing(node.nodeId);
                    }}
                    className="px-2 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-xs text-gray-600 mt-1">
                {interpretation || 'No interpretation'}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    startEditing(node.nodeId);
                  }}
                  className="ml-2 text-blue-600 hover:text-blue-800 underline"
                >
                  Edit
                </button>
              </div>
            )}
          </div>
          <div className="text-right ml-2">
            <div className="text-sm font-mono text-gray-500">1.000</div>
            <div className="text-xs text-gray-400">100.0%</div>
          </div>
        </div>
      </div>
    );
  }, [featureInterpretations, hoveredId, formatNodeId, onFeatureClick, onFeatureHover, saveInterpretation, cancelEditing, startEditing, updateEditText, savingNodeId]);

  // Get interpretation info for clicked node (must be before early return)
  const clickedNodeInterpretation = useMemo(() => {
    if (!clickedNode) return null;
    return featureInterpretations.get(clickedNode.nodeId);
  }, [clickedNode, featureInterpretations]);

  // Render interpretation editing area for clicked node (must be before early return)
  const renderClickedNodeInterpretation = useCallback(() => {
    if (!clickedNode || !clickedNodeInterpretation) {
      return (
        <div className="text-xs text-gray-500 mt-2">Loading...</div>
      );
    }

    const { interpretation, editing, editText, loading } = clickedNodeInterpretation;
    const isEditing = editing || false;
    const currentText = editText || '';
    const isSaving = savingNodeId === clickedNode.nodeId;

    if (loading) {
      return (
        <div className="text-xs text-gray-500 mt-2">Loading...</div>
      );
    }

    if (isEditing) {
      return (
        <div className="mt-2">
          <textarea
            value={currentText}
            onChange={(e) => updateEditText(clickedNode.nodeId, e.target.value)}
            className="w-full p-2 border border-gray-300 rounded text-sm"
            rows={3}
            placeholder="Enter feature interpretation..."
          />
          <div className="flex gap-2 mt-1">
            <button
              onClick={() => saveInterpretation(clickedNode.nodeId)}
              disabled={isSaving}
              className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 disabled:opacity-50"
            >
              {isSaving ? 'Saving...' : 'Save'}
            </button>
            <button
              onClick={() => cancelEditing(clickedNode.nodeId)}
              className="px-3 py-1 bg-gray-300 text-gray-700 rounded text-xs hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="mt-2">
        <div className="text-xs text-gray-600">
          {interpretation || 'No interpretation'}
        </div>
        <button
          onClick={() => startEditing(clickedNode.nodeId)}
          className="mt-1 text-xs text-blue-600 hover:text-blue-800 underline"
        >
          Edit
        </button>
      </div>
    );
  }, [clickedNode, clickedNodeInterpretation, savingNodeId, saveInterpretation, cancelEditing, startEditing, updateEditText]);

  // Early return must be after all hooks
  if (!clickedNode) {
    return (
      <div className="flex flex-col h-full overflow-y-auto">
        <div className="text-gray-500 text-center py-8">
          Click a node on the left to view connected features
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="mb-4 p-2 bg-gray-100 rounded">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-mono">{formatNodeId(clickedNode)}</div>
          <button
            onClick={syncAllInterpretations}
            disabled={syncingAllInterpretations}
            className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-1"
            title="Batch sync all feature interpretations from backend MongoDB"
          >
            {syncingAllInterpretations ? (
              <>
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                Syncing...
              </>
            ) : (
              <>
                <span>🔄</span>
                Sync All Interpretations
              </>
            )}
          </button>
        </div>
        <div className="text-sm text-gray-600 mb-2">{clickedNode.localClerp || ''}</div>
        <div className="border-t pt-2 mt-2">
          <div className="text-xs font-semibold text-gray-700 mb-1">Interpretation:</div>
          {renderClickedNodeInterpretation()}
        </div>
      </div>

      {/* Input Features and Output Features */}
      <div className="flex-1 flex gap-4">
        {/* Input Features */}
        <div className="flex-1">
          <div className="text-lg font-semibold mb-2">Input Features</div>
          <div className="space-y-2">
            {inputNodes.length === 0 ? (
              <div className="text-sm text-gray-500 text-center py-4">No input features</div>
            ) : (
              inputNodes.map(node => renderFeatureRow(node))
            )}
          </div>
        </div>

        {/* Output Features */}
        <div className="flex-1">
          <div className="text-lg font-semibold mb-2">Output Features</div>
          <div className="space-y-2">
            {outputNodes.length === 0 ? (
              <div className="text-sm text-gray-500 text-center py-4">No output features</div>
            ) : (
              outputNodes.map(node => renderFeatureRow(node))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

