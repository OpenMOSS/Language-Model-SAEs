import { AppNavbar } from "@/components/app/navbar";
import { SearchCircuitsVisualization } from "@/components/search-circuits/search-circuits-visualization";

export const SearchCircuitsPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Search Trace Visualization</h1>
          <p className="text-gray-600 mt-2">
            Upload search trace JSON files to visualize MCTS search tree and node details.
          </p>
        </div>

        <SearchCircuitsVisualization />
      </div>
    </div>
  );
};
