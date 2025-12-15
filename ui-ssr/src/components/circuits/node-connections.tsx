import React, { useMemo } from 'react'
import type { CircuitData, Node } from '@/types/circuit'
import { findEdgeWeight, formatFeatureId } from '@/utils/circuit'

interface NodeConnectionsProps {
  data: CircuitData
  clickedId: string | null
  hoveredId: string | null
  pinnedIds: string[]
  hiddenIds: string[]
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

interface ConnectionSection {
  title: string
  nodes: Node[]
}

interface ConnectionType {
  id: 'input' | 'output'
  title: string
  sections: ConnectionSection[]
}

export const NodeConnections: React.FC<NodeConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  pinnedIds,
  hiddenIds,
  onNodeClick,
  onNodeHover,
}) => {
  const clickedNode = useMemo(
    () => data.nodes.find((node) => node.nodeId === clickedId),
    [data.nodes, clickedId],
  )

  const connectionTypes = useMemo((): ConnectionType[] => {
    if (!clickedNode) return []

    const inputNodes = data.nodes.filter((node) => {
      if (node.nodeId === clickedNode.nodeId) return false
      return data.edges.some(
        (edge) =>
          edge.source === node.nodeId && edge.target === clickedNode.nodeId,
      )
    })

    const outputNodes = data.nodes.filter((node) => {
      if (node.nodeId === clickedNode.nodeId) return false
      return data.edges.some(
        (edge) =>
          edge.source === clickedNode.nodeId && edge.target === node.nodeId,
      )
    })

    return [
      {
        id: 'input',
        title: 'Input Features',
        sections: ['Positive', 'Negative'].map((title) => {
          const nodes = inputNodes.filter((node) => {
            const weight = findEdgeWeight(
              data.edges,
              node.nodeId,
              clickedNode.nodeId,
            )
            if (weight === undefined) return false
            return title === 'Positive' ? weight > 0 : weight < 0
          })

          nodes.sort((a, b) => {
            const weightA = Math.abs(
              findEdgeWeight(data.edges, a.nodeId, clickedNode.nodeId) || 0,
            )
            const weightB = Math.abs(
              findEdgeWeight(data.edges, b.nodeId, clickedNode.nodeId) || 0,
            )
            return weightB - weightA
          })

          return { title, nodes }
        }),
      },
      {
        id: 'output',
        title: 'Output Features',
        sections: ['Positive', 'Negative'].map((title) => {
          const nodes = outputNodes.filter((node) => {
            const weight = findEdgeWeight(
              data.edges,
              clickedNode.nodeId,
              node.nodeId,
            )
            if (weight === undefined) return false
            return title === 'Positive' ? weight > 0 : weight < 0
          })

          nodes.sort((a, b) => {
            const weightA = Math.abs(
              findEdgeWeight(data.edges, clickedNode.nodeId, a.nodeId) || 0,
            )
            const weightB = Math.abs(
              findEdgeWeight(data.edges, clickedNode.nodeId, b.nodeId) || 0,
            )
            return weightB - weightA
          })

          return { title, nodes }
        }),
      },
    ]
  }, [data.nodes, data.edges, clickedNode?.nodeId])

  const renderFeatureRow = (node: Node, type: 'input' | 'output') => {
    if (!clickedNode) return null

    const weight =
      type === 'input'
        ? findEdgeWeight(data.edges, node.nodeId, clickedNode.nodeId)
        : findEdgeWeight(data.edges, clickedNode.nodeId, node.nodeId)

    if (weight === undefined) return null

    const isPinned = pinnedIds.includes(node.nodeId)
    const isHidden = hiddenIds.includes(String(node.feature))
    const isHovered = node.nodeId === hoveredId
    const isClicked = node.nodeId === clickedId

    return (
      <div
        key={node.nodeId}
        className={`py-0.5 px-1 border rounded cursor-pointer transition-colors ${
          isPinned
            ? 'bg-yellow-100 border-yellow-300'
            : 'bg-gray-50 border-gray-200'
        } ${isHidden ? 'opacity-50' : ''} ${isHovered ? 'ring-2 ring-blue-300' : ''} ${
          isClicked ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => onNodeClick(node.nodeId, false)}
        onMouseEnter={() => onNodeHover(node.nodeId)}
        onMouseLeave={() => onNodeHover(null)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-gray-600">
              {formatFeatureId(node, false)}
            </span>
            <span className="text-xs font-medium">{node.clerp || ''}</span>
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
  }

  const headerClassName = useMemo(
    () =>
      `mb-3 cursor-pointer p-2 rounded-lg border ${
        clickedNode && pinnedIds.includes(clickedNode.nodeId)
          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
          : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
      }`,
    [pinnedIds, clickedNode?.nodeId],
  )

  if (!clickedNode) {
    return (
      <div className="flex flex-col h-full overflow-y-auto">
        <div className="mb-3">Click a feature on the left for details</div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className={headerClassName}>
        <span className="inline-block mr-2 font-mono tabular-nums w-20 text-sm">
          {formatFeatureId(clickedNode)}
        </span>
        <span className="font-medium text-sm">{clickedNode.clerp || ''}</span>
      </div>

      <div className="flex-1 flex overflow-hidden gap-5">
        {connectionTypes.map((type) => (
          <div key={type.id} className="flex-1">
            <div className="text-lg font-semibold mb-2 text-gray-800">
              {type.title}
            </div>

            <div className="space-y-1 overflow-y-auto h-full">
              {type.sections.map((section) => (
                <div key={section.title}>
                  <h4
                    className={`text-xs font-medium mb-0.5 px-2 py-0.5 rounded ${
                      section.title === 'Positive'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {section.title}
                  </h4>
                  <div className="space-y-0.5">
                    {section.nodes.map((node) =>
                      renderFeatureRow(node, type.id),
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
