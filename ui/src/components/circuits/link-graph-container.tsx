import { useState, useCallback } from "react";
import { LinkGraph } from "./link-graph/link-graph";
import { LinkGraphData, VisState } from "./link-graph/types";

interface LinkGraphContainerProps {
  data: LinkGraphData;
}

export const LinkGraphContainer: React.FC<LinkGraphContainerProps> = ({ data }) => {
  const [visState, setVisState] = useState<VisState>({
    pinnedIds: [],
    clickedId: null,
    hoveredId: null,
    linkType: "either",
    isShowAllLinks: false,
    isHideLayer: false,
  });

  const handleNodeClick = useCallback((nodeId: string, metaKey: boolean) => {
    if (metaKey) {
      // Toggle pinned state
      setVisState(prev => ({
        ...prev,
        pinnedIds: prev.pinnedIds.includes(nodeId)
          ? prev.pinnedIds.filter(id => id !== nodeId)
          : [...prev.pinnedIds, nodeId],
        clickedId: nodeId,
      }));
    } else {
      // Set clicked node
      setVisState(prev => ({
        ...prev,
        clickedId: prev.clickedId === nodeId ? null : nodeId,
      }));
    }
  }, []);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setVisState(prev => ({
      ...prev,
      hoveredId: nodeId,
    }));
  }, []);

  // Find the clicked node data
  const clickedNode = visState.clickedId 
    ? data.nodes.find(node => node.id === visState.clickedId)
    : null;

  return (
    <div className="link-graph-container-wrapper">
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Link Graph Visualization</h3>
        <div className="flex gap-2 mb-4">
          <select
            value={visState.linkType}
            onChange={(e) => setVisState(prev => ({ ...prev, linkType: e.target.value as any }))}
            className="px-3 py-1 border rounded"
          >
            <option value="input">Input Links</option>
            <option value="output">Output Links</option>
            <option value="either">Either</option>
            <option value="both">Both</option>
          </select>
          <button
            onClick={() => setVisState(prev => ({ ...prev, isShowAllLinks: !prev.isShowAllLinks }))}
            className={`px-3 py-1 border rounded ${visState.isShowAllLinks ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
          >
            Show All Links
          </button>
          <button
            onClick={() => setVisState(prev => ({ ...prev, pinnedIds: [] }))}
            className="px-3 py-1 border rounded bg-red-100 hover:bg-red-200"
          >
            Clear Pinned
          </button>
        </div>
        {visState.pinnedIds.length > 0 && (
          <div className="mb-2">
            <span className="text-sm font-medium">Pinned Nodes: </span>
            <span className="text-sm text-gray-600">{visState.pinnedIds.join(", ")}</span>
          </div>
        )}
      </div>
      
      <LinkGraph
        data={data}
        visState={visState}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
      />

      {/* Node Details Section */}
      {clickedNode && (
        <div className="mt-6 p-4 bg-gray-50 border rounded-lg">
          <h4 className="text-lg font-semibold mb-3 text-gray-800">Node Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div>
                <span className="font-medium text-gray-700">Node ID:</span>
                <span className="ml-2 text-gray-900">{clickedNode.nodeId}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Feature ID:</span>
                <span className="ml-2 text-gray-900">{clickedNode.featureId}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Feature Type:</span>
                <span className="ml-2 text-gray-900">{clickedNode.feature_type}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Layer Index:</span>
                <span className="ml-2 text-gray-900">{clickedNode.layerIdx}</span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div>
                <span className="font-medium text-gray-700">Context Index:</span>
                <span className="ml-2 text-gray-900">{clickedNode.ctx_idx}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Position:</span>
                <span className="ml-2 text-gray-900">({clickedNode.pos[0].toFixed(2)}, {clickedNode.pos[1].toFixed(2)})</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Offset:</span>
                <span className="ml-2 text-gray-900">({clickedNode.xOffset}, {clickedNode.yOffset})</span>
              </div>
              {clickedNode.featureIndex !== undefined && (
                <div>
                  <span className="font-medium text-gray-700">Feature Index:</span>
                  <span className="ml-2 text-gray-900">{clickedNode.featureIndex}</span>
                </div>
              )}
            </div>
            
            <div className="space-y-2">
              {clickedNode.logitPct !== undefined && (
                <div>
                  <span className="font-medium text-gray-700">Logit %:</span>
                  <span className="ml-2 text-gray-900">{(clickedNode.logitPct * 100).toFixed(2)}%</span>
                </div>
              )}
              {clickedNode.logitToken && (
                <div>
                  <span className="font-medium text-gray-700">Logit Token:</span>
                  <span className="ml-2 text-gray-900">{clickedNode.logitToken}</span>
                </div>
              )}
              {clickedNode.localClerp && (
                <div>
                  <span className="font-medium text-gray-700">Local Clerp:</span>
                  <span className="ml-2 text-gray-900">{clickedNode.localClerp}</span>
                </div>
              )}
              {clickedNode.remoteClerp && (
                <div>
                  <span className="font-medium text-gray-700">Remote Clerp:</span>
                  <span className="ml-2 text-gray-900">{clickedNode.remoteClerp}</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Connection Information */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h5 className="font-medium text-gray-700 mb-2">Connections</h5>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="font-medium text-gray-700">Input Links:</span>
                <span className="ml-2 text-gray-900">{clickedNode.sourceLinks?.length || 0}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Output Links:</span>
                <span className="ml-2 text-gray-900">{clickedNode.targetLinks?.length || 0}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 