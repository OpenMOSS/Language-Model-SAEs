import { useCallback, useRef } from "react";
import { LinkGraph, exportLinkGraphAsSvg } from "./link-graph/link-graph";
import { LinkGraphData, VisState, Node } from "./link-graph/types";
import { extractLayerAndFeature } from "./link-graph/utils";
import { fetchFeature, getDictionaryName } from "@/utils/api";
import { Feature } from "@/types/feature";

interface LinkGraphContainerProps {
  data: LinkGraphData;
  onNodeClick?: (node: Node, isMetaKey: boolean) => void;
  onNodeHover?: (nodeId: string | null) => void;
  onFeatureSelect?: (feature: Feature | null) => void;
  onConnectedFeaturesSelect?: (features: Feature[]) => void;
  onConnectedFeaturesLoading?: (loading: boolean) => void;
  clickedId?: string | null;
  hoveredId?: string | null;
  pinnedIds?: string[];
  hideEmbLogit?: boolean; // Whether to hide Emb and Logit layers (for interaction circuit mode)
}

export const LinkGraphContainer: React.FC<LinkGraphContainerProps> = ({ 
  data, 
  onNodeClick,
  onNodeHover,
  onFeatureSelect,
  onConnectedFeaturesSelect,
  onConnectedFeaturesLoading,
  clickedId,
  hoveredId,
  pinnedIds = [],
  hideEmbLogit = false
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  // Create visState from props
  const visState: VisState = {
    pinnedIds,
    clickedId: clickedId || null,
    hoveredId: hoveredId || null,
    isShowAllLinks: false,
  };

  const handleNodeClick = useCallback(async (nodeId: string, metaKey: boolean) => {
    const isMultiFile = !!((data as any)?.metadata?.sourceFileNames?.length > 1);
    console.log("[LinkGraphContainer] handleNodeClick 收到:", { nodeId, metaKey, isMultiFile, nodesCount: data.nodes.length });

    // Background click (empty id): clear selection-related UI without looking up a node
    if (!nodeId) {
      onFeatureSelect?.(null);
      onConnectedFeaturesSelect?.([]);
      onConnectedFeaturesLoading?.(false);
      // Propagate a dummy node with empty nodeId so parent can clear clickedId
      onNodeClick?.({ nodeId: "", id: "", feature_type: "" } as any, metaKey);
      return;
    }

    // Node lookup: try nodeId and id (coerce to string for robustness)
    const node = data.nodes.find(n => String(n.nodeId ?? n.id ?? "") === String(nodeId));
    if (!node) {
      const sampleIds = data.nodes.slice(0, 5).map(n => ({ nodeId: n.nodeId, id: n.id }));
      console.warn("[LinkGraphContainer] 节点未找到:", { nodeId, sampleIds });
      return;
    }

    console.log("[LinkGraphContainer] 节点已找到，调用 onNodeClick:", { nodeId: node.nodeId, feature_type: node.feature_type });
    // Always call parent handler first to update global state
    onNodeClick?.(node, metaKey);
    
    // If not meta key (not pinning), handle feature selection
    if (!metaKey) {
      const names = (data as any)?.metadata?.sourceFileNames as string[] | undefined;
      const isMultiFile = !!(names && names.length > 1);

      // In single-file mode, keep the original "click same node to deselect" behavior.
      // In multi-file mode, do NOT treat clicking the same node as deselect, to avoid flicker.
      if (!isMultiFile && clickedId === nodeId) {
        onFeatureSelect?.(null);
        onConnectedFeaturesSelect?.([]);
        onConnectedFeaturesLoading?.(false);
        return;
      }

      // Only fetch feature data for supported node types
      if (node.feature_type === 'cross layer transcoder' || node.feature_type === 'lorsa') {
        const layerAndFeature = extractLayerAndFeature(nodeId);
        if (layerAndFeature) {
          const { layer, featureId, isLorsa } = layerAndFeature;
          const dictionaryName = getDictionaryName(data.metadata, layer, isLorsa);

          if (dictionaryName) {
            try {
              onConnectedFeaturesLoading?.(true);
              const feature = await fetchFeature(dictionaryName, layer, featureId);
              if (feature) {
                onFeatureSelect?.(feature);
              } else {
                onFeatureSelect?.(null);
                onConnectedFeaturesSelect?.([]);
                onConnectedFeaturesLoading?.(false);
              }
            } catch (error) {
              console.error('Failed to fetch feature:', error);
              onFeatureSelect?.(null);
              onConnectedFeaturesSelect?.([]);
              onConnectedFeaturesLoading?.(false);
            }
          } else {
            onFeatureSelect?.(null);
            onConnectedFeaturesSelect?.([]);
            onConnectedFeaturesLoading?.(false);
          }
        } else {
          onFeatureSelect?.(null);
          onConnectedFeaturesSelect?.([]);
          onConnectedFeaturesLoading?.(false);
        }
      } else {
        // For unsupported node types, clear the selection
        onFeatureSelect?.(null);
        onConnectedFeaturesSelect?.([]);
        onConnectedFeaturesLoading?.(false);
      }
    }
  }, [onNodeClick, clickedId, data, onFeatureSelect, onConnectedFeaturesSelect, onConnectedFeaturesLoading]);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    if (onNodeHover) {
      onNodeHover(nodeId);
    }
  }, [onNodeHover]);

  // Export SVG function
  const handleExportSvg = useCallback(() => {
    if (typeof window === 'undefined' || !containerRef.current) return;
    
    try {
      // Find the SVG element within the container
      const svgElement = containerRef.current.querySelector('svg') as SVGSVGElement | null;
      if (!svgElement) {
        console.error('SVG element not found');
        return;
      }
      
      // Export the SVG
      const svgString = exportLinkGraphAsSvg(svgElement);
      if (!svgString) {
        console.error('Failed to export SVG');
        return;
      }
      
      // Create a blob and download
      const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `link-graph-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.svg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting SVG:', error);
    }
  }, []);

  return (
    <div ref={containerRef} className="link-graph-container-wrapper w-full h-full overflow-hidden">
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold">Link Graph Visualization</h3>
          <button
            onClick={handleExportSvg}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors text-sm font-medium"
            title="Export graph as SVG"
          >
            Export SVG
          </button>
        </div>
        {visState.pinnedIds.length > 0 && (
          <div className="mb-2">
            <span className="text-sm font-medium">Pinned Nodes: </span>
            <span className="text-sm text-gray-600">{visState.pinnedIds.join(", ")}</span>
          </div>
        )}
      </div>
      
      <div className="w-full h-full overflow-hidden relative">
        <div className="absolute inset-0 overflow-hidden">
          <div className="w-full h-full overflow-hidden">
            <LinkGraph
              data={data}
              visState={visState}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              hideEmbLogit={hideEmbLogit}
            />
          </div>
        </div>
      </div>


    </div>
  );
}; 