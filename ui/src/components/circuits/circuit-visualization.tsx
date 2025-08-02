import { useEffect, useState } from "react";
import { LinkGraphContainer } from "./link-graph-container";
import { LinkGraphData } from "./link-graph/types";
import { loadDefaultCircuitData } from "./link-graph/utils";

export const CircuitVisualization = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [linkGraphData, setLinkGraphData] = useState<LinkGraphData | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const data = await loadDefaultCircuitData();
        setLinkGraphData(data);
      } catch (err) {
        console.error('Failed to load circuit data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load circuit data');
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-red-600 mb-2">Failed to load circuit visualization</h3>
          <p className="text-gray-600">{error}</p>
          <button 
            onClick={() => {
              setError(null);
              setIsLoading(true);
            }}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (isLoading || !linkGraphData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading circuit visualization...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Circuit Visualization</h2>
        <div className="text-sm text-gray-500">
          {linkGraphData.metadata.prompt_tokens.join(' ')}
        </div>
      </div>

      {/* Link Graph Component */}
      <div className="border rounded-lg p-4 bg-white">
        <LinkGraphContainer data={linkGraphData} />
      </div>

      {/* Circuit Information */}
      <div className="border rounded-lg p-4 bg-white">
        <h3 className="text-lg font-semibold mb-4">Circuit Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium mb-2">Nodes ({linkGraphData.nodes.length})</h4>
            <ul className="space-y-1 text-sm max-h-40 overflow-y-auto">
              {linkGraphData.nodes.slice(0, 10).map((node) => (
                <li key={node.id} className="flex justify-between">
                  <span>{node.feature_type} ({node.nodeId})</span>
                  <span className="text-gray-500">ctx: {node.ctx_idx}</span>
                </li>
              ))}
              {linkGraphData.nodes.length > 10 && (
                <li className="text-gray-500 text-xs">... and {linkGraphData.nodes.length - 10} more nodes</li>
              )}
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Links ({linkGraphData.links.length})</h4>
            <ul className="space-y-1 text-sm max-h-40 overflow-y-auto">
              {linkGraphData.links.slice(0, 10).map((link, index) => (
                <li key={index} className="flex justify-between">
                  <span>{link.source} → {link.target}</span>
                  <span className="text-gray-500">width: {link.strokeWidth.toFixed(1)}</span>
                </li>
              ))}
              {linkGraphData.links.length > 10 && (
                <li className="text-gray-500 text-xs">... and {linkGraphData.links.length - 10} more links</li>
              )}
            </ul>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="border rounded-lg p-4 bg-blue-50">
        <h3 className="text-lg font-semibold mb-2 text-blue-800">How to Use</h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Click on nodes to select them</li>
          <li>• Hold Ctrl/Cmd and click to pin nodes</li>
          <li>• Use the link type selector to filter connections</li>
          <li>• Hover over nodes to see details</li>
        </ul>
      </div>
    </div>
  );
}; 