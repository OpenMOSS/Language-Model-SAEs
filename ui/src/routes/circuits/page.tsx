import { AppNavbar } from "@/components/app/navbar";
import { CircuitVisualization } from "@/components/circuits/circuit-visualization";

export const CircuitsPage = () => {

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto p-4">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Circuit Tracing</h1>
          <p className="text-gray-600 mt-2">
            Upload circuit data and click on nodes to view detailed feature information.
          </p>
        </div>

        <CircuitVisualization />
      </div>
    </div>
  );
}; 