import { createFileRoute } from '@tanstack/react-router'
import { CircuitVisualization } from '@/components/circuits/circuit-visualization'

export const Route = createFileRoute('/circuits/')({
  component: CircuitsPage,
})

/**
 * Circuit tracing page component.
 */
function CircuitsPage() {
  return (
    <div className="h-full overflow-y-auto pt-4 pb-20 px-8 flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Circuit Tracing</h1>
        <p className="text-gray-600 mt-2">
          Upload circuit data and click on nodes to view detailed feature
          information.
        </p>
      </div>

      <CircuitVisualization />
    </div>
  )
}
