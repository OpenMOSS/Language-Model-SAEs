import React, { useCallback, useMemo } from 'react'
import type { Link, LinkGraphData, Node } from '@/types/circuit'
import type { Feature } from '@/types/feature'
import {
  extractLayerAndFeature,
  formatFeatureId,
  getDictionaryName,
} from '@/utils/circuit'
import { fetchFeature } from '@/api/features'

interface NodeConnectionsProps {
  data: LinkGraphData
  clickedId: string | null
  hoveredId: string | null
  pinnedIds: string[]
  hiddenIds: string[]
  onFeatureClick: (node: Node, isMetaKey: boolean) => void
  onFeatureSelect: (feature: Feature | null) => void
  onFeatureHover: (nodeId: string | null) => void
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

/**
 * Component to display connections for a selected node.
 */
export const NodeConnections: React.FC<NodeConnectionsProps> = ({
  data,
  clickedId,
  hoveredId,
  pinnedIds,
  hiddenIds,
  onFeatureClick,
  onFeatureSelect,
  onFeatureHover,
}) => {
  // Memoize the clicked node
  const clickedNode = useMemo(
    () => data.nodes.find((node) => node.nodeId === clickedId),
    [data.nodes, clickedId],
  )

  // Memoize the connection types computation
  const connectionTypes = useMemo((): ConnectionType[] => {
    if (!clickedNode || !clickedNode.sourceLinks || !clickedNode.targetLinks) {
      return []
    }

    // Input features: nodes that have links TO the clicked node
    const inputNodes = data.nodes.filter(
      (node) =>
        node.nodeId !== clickedNode.nodeId &&
        node.sourceLinks &&
        node.sourceLinks.some((link) => link.target === clickedNode.nodeId),
    )

    // Output features: nodes that the clicked node has links TO
    const outputNodes = data.nodes.filter(
      (node) =>
        node.nodeId !== clickedNode.nodeId &&
        clickedNode.sourceLinks &&
        clickedNode.sourceLinks.some((link) => link.target === node.nodeId),
    )

    return [
      {
        id: 'input',
        title: 'Input Features',
        sections: ['Positive', 'Negative'].map((title) => {
          const nodes = inputNodes.filter((node) => {
            const link = node.sourceLinks?.find(
              (l) => l.target === clickedNode.nodeId,
            )
            if (!link || link.weight === undefined) return false
            return title === 'Positive' ? link.weight > 0 : link.weight < 0
          })

          // Sort by absolute weight
          nodes.sort((a, b) => {
            const linkA = a.sourceLinks?.find(
              (l) => l.target === clickedNode.nodeId,
            )
            const linkB = b.sourceLinks?.find(
              (l) => l.target === clickedNode.nodeId,
            )
            const weightA = Math.abs(linkA?.weight || 0)
            const weightB = Math.abs(linkB?.weight || 0)
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
            const link = clickedNode.sourceLinks?.find(
              (l) => l.target === node.nodeId,
            )
            if (!link || link.weight === undefined) return false
            return title === 'Positive' ? link.weight > 0 : link.weight < 0
          })

          // Sort by absolute weight
          nodes.sort((a, b) => {
            const linkA = clickedNode.sourceLinks?.find(
              (l) => l.target === a.nodeId,
            )
            const linkB = clickedNode.sourceLinks?.find(
              (l) => l.target === b.nodeId,
            )
            const weightA = Math.abs(linkA?.weight || 0)
            const weightB = Math.abs(linkB?.weight || 0)
            return weightB - weightA
          })

          return { title, nodes }
        }),
      },
    ]
  }, [
    data.nodes,
    clickedNode?.nodeId,
    clickedNode?.sourceLinks,
    clickedNode?.targetLinks,
  ])

  const handleNodeClick = useCallback(
    async (nodeId: string, metaKey: boolean) => {
      const node = data.nodes.find((n) => n.id === nodeId)
      if (!node) return

      onFeatureClick?.(node, metaKey)

      if (!metaKey) {
        if (clickedId === nodeId) {
          onFeatureSelect?.(null)
        } else {
          if (
            node.feature_type === 'cross layer transcoder' ||
            node.feature_type === 'lorsa'
          ) {
            const layerAndFeature = extractLayerAndFeature(nodeId)
            if (layerAndFeature) {
              const { layer, featureId, isLorsa } = layerAndFeature
              const dictionaryName = getDictionaryName(
                data.metadata,
                layer,
                isLorsa,
              )

              if (dictionaryName) {
                try {
                  const feature = await fetchFeature({
                    data: {
                      dictionary: dictionaryName,
                      featureIndex: featureId,
                    },
                  })
                  if (feature) {
                    onFeatureSelect?.(feature)
                  } else {
                    onFeatureSelect?.(null)
                  }
                } catch (error) {
                  console.error('Failed to fetch feature:', error)
                  onFeatureSelect?.(null)
                }
              } else {
                onFeatureSelect?.(null)
              }
            } else {
              onFeatureSelect?.(null)
            }
          } else {
            onFeatureSelect?.(null)
          }
        }
      }
    },
    [onFeatureClick, clickedId, data.nodes, data.metadata, onFeatureSelect],
  )

  // Render a feature row
  const renderFeatureRow = useCallback(
    (node: Node, type: 'input' | 'output') => {
      if (!clickedNode) return null

      const link =
        type === 'input'
          ? node.sourceLinks?.find((l) => l.target === clickedNode.nodeId)
          : clickedNode.sourceLinks?.find((l) => l.target === node.nodeId)

      if (!link || link.weight === undefined) return null

      const weight = link.weight
      const isPinned = pinnedIds.includes(node.nodeId)
      const isHidden = hiddenIds.includes(node.featureId)
      const isHovered = node.nodeId === hoveredId
      const isClicked = node.nodeId === clickedId

      return (
        <div
          key={node.nodeId}
          className={`feature-row py-0.5 px-1 border rounded cursor-pointer transition-colors ${
            isPinned
              ? 'bg-yellow-100 border-yellow-300'
              : 'bg-gray-50 border-gray-200'
          } ${isHidden ? 'opacity-50' : ''} ${isHovered ? 'ring-2 ring-blue-300' : ''} ${
            isClicked ? 'ring-2 ring-blue-500' : ''
          }`}
          onClick={() => {
            onFeatureClick(node, false)
            handleNodeClick(node.nodeId, false)
          }}
          onMouseEnter={() => onFeatureHover(node.nodeId)}
          onMouseLeave={() => onFeatureHover(null)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-gray-600">
                {formatFeatureId(node, false)}
              </span>
              <span className="text-xs font-medium">
                {node.localClerp || node.remoteClerp || ''}
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
    [
      clickedNode,
      pinnedIds,
      hiddenIds,
      hoveredId,
      clickedId,
      onFeatureClick,
      onFeatureHover,
      handleNodeClick,
    ],
  )

  // Header styling
  const headerClassName = useMemo(
    () =>
      `header-top-row section-title mb-3 cursor-pointer p-2 rounded-lg border ${
        clickedNode && pinnedIds.includes(clickedNode.nodeId)
          ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
          : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
      }`,
    [pinnedIds, clickedNode?.nodeId],
  )

  if (!clickedNode) {
    return (
      <div className="node-connections flex flex-col h-full overflow-y-auto">
        <div className="header-top-row section-title mb-3">
          Click a feature on the left for details
        </div>
      </div>
    )
  }

  return (
    <div className="node-connections flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className={headerClassName}>
        <span className="inline-block mr-2 font-mono tabular-nums w-20 text-sm">
          {formatFeatureId(clickedNode)}
        </span>
        <span className="feature-title font-medium text-sm">
          {clickedNode.localClerp || clickedNode.remoteClerp || ''}
        </span>
      </div>

      {/* Connections */}
      <div className="connections flex-1 flex overflow-hidden gap-5">
        {connectionTypes.map((type) => (
          <div
            key={type.id}
            className={`features flex-1 ${type.id === 'output' ? 'output' : 'input'}`}
          >
            <div className="section-title text-lg font-semibold mb-2 text-gray-800">
              {type.title}
            </div>

            <div className="effects space-y-1 overflow-y-auto h-full">
              {type.sections.map((section) => (
                <div key={section.title} className="section">
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
