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
    </div>
  );
}; 