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
    <LinkGraph
      data={data}
      visState={visState}
      onNodeClick={onNodeClick}
      onNodeHover={onNodeHover}
    />
  )
}
