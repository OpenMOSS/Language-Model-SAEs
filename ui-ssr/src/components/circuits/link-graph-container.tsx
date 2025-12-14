import { useCallback } from 'react'
import { LinkGraph } from './link-graph/link-graph'
import type { LinkGraphData, Node, VisState } from '@/types/circuit'
import type { Feature } from '@/types/feature'
import { extractLayerAndFeature, getDictionaryName } from '@/utils/circuit'
import { fetchFeature } from '@/api/features'

interface LinkGraphContainerProps {
  data: LinkGraphData
  onNodeClick?: (node: Node, isMetaKey: boolean) => void
  onNodeHover?: (nodeId: string | null) => void
  onFeatureSelect?: (feature: Feature | null) => void
  onConnectedFeaturesSelect?: (features: Feature[]) => void
  onConnectedFeaturesLoading?: (loading: boolean) => void
  clickedId?: string | null
  hoveredId?: string | null
  pinnedIds?: string[]
}

/**
 * Container component for the LinkGraph with feature fetching logic.
 */
export const LinkGraphContainer: React.FC<LinkGraphContainerProps> = ({
  data,
  onNodeClick,
  onNodeHover,
  onFeatureSelect,
  onConnectedFeaturesSelect,
  onConnectedFeaturesLoading,
  clickedId,
  hoveredId,
  pinnedIds = [],
}) => {
  // Create visState from props
  const visState: VisState = {
    pinnedIds,
    clickedId: clickedId || null,
    hoveredId: hoveredId || null,
    isShowAllLinks: false,
  }

  const handleNodeClick = useCallback(
    async (nodeId: string, metaKey: boolean) => {
      const node = data.nodes.find((n) => n.id === nodeId)
      if (!node) return

      // Always call parent handler first
      onNodeClick?.(node, metaKey)

      // If not meta key, handle feature selection
      if (!metaKey) {
        if (clickedId === nodeId) {
          // Deselecting the same node
          onFeatureSelect?.(null)
          onConnectedFeaturesSelect?.([])
          onConnectedFeaturesLoading?.(false)
        } else {
          // Only fetch for supported node types
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
                  onConnectedFeaturesLoading?.(true)
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
                    onConnectedFeaturesSelect?.([])
                    onConnectedFeaturesLoading?.(false)
                  }
                } catch (error) {
                  console.error('Failed to fetch feature:', error)
                  onFeatureSelect?.(null)
                  onConnectedFeaturesSelect?.([])
                  onConnectedFeaturesLoading?.(false)
                }
              } else {
                onFeatureSelect?.(null)
                onConnectedFeaturesSelect?.([])
                onConnectedFeaturesLoading?.(false)
              }
            } else {
              onFeatureSelect?.(null)
              onConnectedFeaturesSelect?.([])
              onConnectedFeaturesLoading?.(false)
            }
          } else {
            onFeatureSelect?.(null)
            onConnectedFeaturesSelect?.([])
            onConnectedFeaturesLoading?.(false)
          }
        }
      }
    },
    [
      onNodeClick,
      clickedId,
      data.nodes,
      data.metadata,
      onFeatureSelect,
      onConnectedFeaturesSelect,
      onConnectedFeaturesLoading,
    ],
  )

  const handleNodeHover = useCallback(
    (nodeId: string | null) => {
      onNodeHover?.(nodeId)
    },
    [onNodeHover],
  )

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
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
