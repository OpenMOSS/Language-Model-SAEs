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
  hideEmbLogit?: boolean; // 是否隐藏 Emb 和 Logit 层（用于 interaction circuit 模式）
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
    // Try both id and nodeId to handle different node formats
    const node = data.nodes.find(n => n.nodeId === nodeId || n.id === nodeId);
    if (!node) {
      console.warn('⚠️ LinkGraphContainer: Node not found for nodeId:', nodeId);
      return;
    }

    // Always call parent handler first to update global state
    onNodeClick?.(node, metaKey);
    
    // If not meta key (not pinning), handle feature selection
    if (!metaKey) {
      if (clickedId === nodeId) {
        // Deselecting the same node
        onFeatureSelect?.(null);
        onConnectedFeaturesSelect?.([]);
        onConnectedFeaturesLoading?.(false);
      } else {
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
    }
  }, [onNodeClick, clickedId, data.nodes, data.metadata, onFeatureSelect, onConnectedFeaturesSelect, onConnectedFeaturesLoading]);

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