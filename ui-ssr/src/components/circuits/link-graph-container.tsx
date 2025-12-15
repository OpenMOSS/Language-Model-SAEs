import { LinkGraph } from './link-graph/link-graph'
import type { CircuitData, VisState } from '@/types/circuit'

interface LinkGraphContainerProps {
  data: CircuitData
  visState: VisState
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

export const LinkGraphContainer: React.FC<LinkGraphContainerProps> = ({
  data,
  visState,
  onNodeClick,
  onNodeHover,
}) => {
  return (
    <div className="w-full h-full overflow-hidden">
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Link Graph Visualization</h3>
        {visState.pinnedIds.length > 0 && (
          <div className="mt-2 p-2 bg-amber-100 rounded text-sm">
            <span className="font-medium">Pinned Nodes: </span>
            <span className="text-gray-600">
              {visState.pinnedIds.join(', ')}
            </span>
          </div>
        )}
      </div>

      <div className="w-full h-full overflow-hidden relative">
        <div className="absolute inset-0 overflow-hidden">
          <div className="w-full h-full overflow-hidden">
            <LinkGraph
              data={data}
              visState={visState}
              onNodeClick={onNodeClick}
              onNodeHover={onNodeHover}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
