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
    </div>
  );
}; 