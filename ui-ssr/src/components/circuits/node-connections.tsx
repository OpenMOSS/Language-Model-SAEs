import { memo, useMemo } from 'react'
import { Link } from '@tanstack/react-router'
import { Send } from 'lucide-react'
import { Card } from '../ui/card'
import { Info } from '../ui/info'
import { Button } from '../ui/button'
import type { Node } from '@/types/circuit'
import type { RawEdgeIndex, RawNodeIndex } from '@/utils/circuit-index'
import { cn } from '@/lib/utils'
import { formatFeatureId } from '@/utils/circuit'
import { getWeightStyle } from '@/utils/style'
import { getEdgesBySource, getEdgesByTarget } from '@/utils/circuit-index'

interface FeatureRowProps {
  node: Node
  weight: number
  isHidden: boolean
  isHovered: boolean
  isClicked: boolean
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

const FeatureRow = memo(
  ({
    node,
    weight,
    isHidden,
    isHovered,
    isClicked,
    onNodeClick,
    onNodeHover,
  }: FeatureRowProps) => {
    const weightStyle = getWeightStyle(weight)

    return (
      <div
        className={cn(
          'py-2 px-2 mx-1 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200',
          isHidden && 'opacity-50',
          isHovered && 'ring-2 ring-blue-300',
          isClicked && 'ring-2 ring-blue-500',
        )}
        style={weightStyle}
        onClick={() => onNodeClick(node.nodeId, false)}
        onMouseEnter={() => onNodeHover(node.nodeId)}
        onMouseLeave={() => onNodeHover(null)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-gray-600">
              {formatFeatureId(node, false)}
            </span>
            <span className="text-xs font-medium">
              {(node.featureType === 'cross layer transcoder' ||
                node.featureType === 'lorsa') &&
                node.feature.interpretation?.text}
            </span>
          </div>
          <div className="text-right">
            <div className="text-xs font-mono">
              {weight > 0 ? '+' : ''}
              {weight.toFixed(3)}
            </div>
          </div>
        </div>
      </div>
    )
  },
)

FeatureRow.displayName = 'FeatureRow'

interface NodeConnectionsProps {
  nodeIndex: RawNodeIndex
  edgeIndex: RawEdgeIndex
  clickedId: string
  hoveredId: string | null
  hiddenIds: string[]
  className?: string
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

export const NodeConnections = memo(
  ({
    nodeIndex,
    edgeIndex,
    clickedId,
    hoveredId,
    hiddenIds,
    onNodeClick,
    onNodeHover,
    className,
  }: NodeConnectionsProps) => {
    const clickedNode = useMemo(
      () => nodeIndex.byId.get(clickedId)!,
      [nodeIndex, clickedId],
    )

    const inputNodes = useMemo(() => {
      const incomingEdges = getEdgesByTarget(edgeIndex, clickedId)
      return incomingEdges
        .map((edge) => {
          const node = nodeIndex.byId.get(edge.source)
          if (!node) return null
          return { node, weight: edge.weight }
        })
        .filter((item): item is { node: Node; weight: number } => item !== null)
        .sort((a, b) => b.weight - a.weight)
    }, [edgeIndex, nodeIndex, clickedId])

    const outputNodes = useMemo(() => {
      const outgoingEdges = getEdgesBySource(edgeIndex, clickedId)
      return outgoingEdges
        .map((edge) => {
          const node = nodeIndex.byId.get(edge.target)
          if (!node) return null
          return { node, weight: edge.weight }
        })
        .filter((item): item is { node: Node; weight: number } => item !== null)
        .sort((a, b) => b.weight - a.weight)
    }, [edgeIndex, nodeIndex, clickedId])

    const hiddenIdsSet = useMemo(() => new Set(hiddenIds), [hiddenIds])

    return (
      <Card
        className={cn('flex flex-col basis-1/2 min-w-0 gap-4 p-4', className)}
      >
        <div className="flex items-center justify-between">
          <div className="flex justify-between items-center space-x-2 w-full">
            <span className="text-sm font-medium">
              {clickedNode.featureType === 'cross layer transcoder' ||
              clickedNode.featureType === 'lorsa'
                ? clickedNode.feature.interpretation?.text
                : formatFeatureId(clickedNode, true)}
            </span>
            {(clickedNode.featureType === 'cross layer transcoder' ||
              clickedNode.featureType === 'lorsa') && (
              <Link
                to="/dictionaries/$dictionaryName/features/$featureIndex"
                params={{
                  dictionaryName: clickedNode.saeName,
                  featureIndex: clickedNode.feature.featureIndex.toString(),
                }}
              >
                <Button size="sm" className="h-8 px-4 text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono text-gray-600">
                      {formatFeatureId(clickedNode, true)}
                    </span>
                    <Send className="w-3.5 h-3.5 text-gray-400" />
                  </div>
                </Button>
              </Link>
            )}
          </div>
        </div>
        <div className="flex gap-4 flex-1 overflow-hidden">
          <div className="flex flex-col w-1/2 gap-2 min-h-0">
            <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
              <span>INPUT NODES</span>
              <Info iconSize={14}>
                Nodes (features/embeddings) that influence the clicked node.
              </Info>
            </div>
            <div className="flex flex-col gap-2 overflow-y-auto no-scrollbar">
              {inputNodes.map((item) => (
                <FeatureRow
                  key={item.node.nodeId}
                  node={item.node}
                  weight={item.weight}
                  isHidden={hiddenIdsSet.has(item.node.nodeId)}
                  isHovered={item.node.nodeId === hoveredId}
                  isClicked={item.node.nodeId === clickedId}
                  onNodeClick={onNodeClick}
                  onNodeHover={onNodeHover}
                />
              ))}
            </div>
          </div>
          <div className="flex flex-col w-1/2 gap-2 min-h-0">
            <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
              <span>OUTPUT NODES</span>
              <Info iconSize={14}>
                Nodes (features/logits) that the clicked node influences.
              </Info>
            </div>
            <div className="flex flex-col gap-2 overflow-y-auto no-scrollbar">
              {outputNodes.map((item) => (
                <FeatureRow
                  key={item.node.nodeId}
                  node={item.node}
                  weight={item.weight}
                  isHidden={hiddenIdsSet.has(item.node.nodeId)}
                  isHovered={item.node.nodeId === hoveredId}
                  isClicked={item.node.nodeId === clickedId}
                  onNodeClick={onNodeClick}
                  onNodeHover={onNodeHover}
                />
              ))}
            </div>
          </div>
        </div>
      </Card>
    )
  },
)

NodeConnections.displayName = 'NodeConnections'
