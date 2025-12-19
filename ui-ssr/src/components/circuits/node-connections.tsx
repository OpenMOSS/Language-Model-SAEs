import React, { memo, useMemo } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Send } from 'lucide-react'
import { Card } from '../ui/card'
import { Info } from '../ui/info'
import { Button } from '../ui/button'
import type { CircuitData, Node } from '@/types/circuit'
import { cn } from '@/lib/utils'
import { formatFeatureId } from '@/utils/circuit'
import { getWeightStyle } from '@/utils/style'

interface NodeConnectionsProps {
  data: CircuitData
  clickedId: string
  hoveredId: string | null
  hiddenIds: string[]
  className?: string
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

export const NodeConnections = memo(
  ({
    data,
    clickedId,
    hoveredId,
    hiddenIds,
    onNodeClick,
    onNodeHover,
    className,
  }: NodeConnectionsProps) => {
    const clickedNode = useMemo(
      () => data.nodes.find((node) => node.nodeId === clickedId)!,
      [data.nodes, clickedId],
    )

    const navigate = useNavigate()

    const inputNodes = useMemo(
      () =>
        data.nodes
          .flatMap((node) => {
            const edge = data.edges.find(
              (edge) =>
                edge.source === node.nodeId &&
                edge.target === clickedNode.nodeId,
            )
            if (!edge) return []
            return [{ node, weight: edge.weight }]
          })
          .sort((a, b) => b.weight - a.weight),
      [data.nodes, data.edges, clickedNode],
    )

    const outputNodes = useMemo(
      () =>
        data.nodes
          .flatMap((node) => {
            const edge = data.edges.find(
              (edge) =>
                edge.source === clickedNode.nodeId &&
                edge.target === node.nodeId,
            )
            if (!edge) return []
            return [{ node, weight: edge.weight }]
          })
          .sort((a, b) => b.weight - a.weight),
      [data.nodes, data.edges, clickedNode],
    )

    const renderFeatureRow = (node: { node: Node; weight: number }) => {
      const isHidden = hiddenIds.includes(String(node.node.nodeId))
      const isHovered = node.node.nodeId === hoveredId
      const isClicked = node.node.nodeId === clickedId
      const weightStyle = getWeightStyle(node.weight)

      return (
        <div
          key={node.node.nodeId}
          className={cn(
            'py-2 px-2 mx-1 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200',
            isHidden && 'opacity-50',
            isHovered && 'ring-2 ring-blue-300',
            isClicked && 'ring-2 ring-blue-500',
          )}
          style={weightStyle}
          onClick={() => onNodeClick(node.node.nodeId, false)}
          onMouseEnter={() => onNodeHover(node.node.nodeId)}
          onMouseLeave={() => onNodeHover(null)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-gray-600">
                {formatFeatureId(node.node, false)}
              </span>
              <span className="text-xs font-medium">
                {(node.node.featureType === 'cross layer transcoder' ||
                  node.node.featureType === 'lorsa') &&
                  node.node.feature.interpretation?.text}
              </span>
            </div>
            <div className="text-right">
              <div className="text-xs font-mono">
                {node.weight > 0 ? '+' : ''}
                {node.weight.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      )
    }

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
              <Button
                size="sm"
                className="h-8 px-4 text-xs"
                onClick={() =>
                  navigate({
                    to: `/dictionaries/$dictionaryName/features/$featureIndex`,
                    params: {
                      dictionaryName: clickedNode.saeName,
                      featureIndex: clickedNode.feature.featureIndex.toString(),
                    },
                  })
                }
              >
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono text-gray-600">
                    {formatFeatureId(clickedNode, true)}
                  </span>
                  <Send className="w-3.5 h-3.5 text-gray-400" />
                </div>
              </Button>
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
              {inputNodes.map((node) => renderFeatureRow(node))}
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
              {outputNodes.map((node) => renderFeatureRow(node))}
            </div>
          </div>
        </div>
      </Card>
    )
  },
)

NodeConnections.displayName = 'NodeConnections'
