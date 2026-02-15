import { memo, useMemo } from 'react'
import type { VisState } from '@/types/circuit'
import type { EdgeIndex } from '@/utils/circuit-index'
import { getEdgeStrokeWidth } from '@/utils/circuit'
import { getConnectedEdges, getTopEdgesByWeight } from '@/utils/circuit-index'

interface LinksProps {
  edgeIndex: EdgeIndex
  visState: VisState
}

export const Links = memo(({ edgeIndex, visState }: LinksProps) => {
  const isFilteredView = !!visState.clickedId

  const connectedEdges = useMemo(() => {
    if (!visState.clickedId) return getTopEdgesByWeight(edgeIndex, 600)
    return getConnectedEdges(edgeIndex, visState.clickedId)
  }, [edgeIndex, visState.clickedId])

  // Styling based on view mode:
  // - "Overview" mode (top 600 edges by weight): subtle, low opacity
  // - "Connected" mode (~100 edges): prominent, color-coded by weight sign
  const opacity = isFilteredView ? 0.6 : 0.4
  const strokeWidthScale = isFilteredView ? 0.5 : 0.35

  return (
    <g>
      {connectedEdges.map((d: any) => (
        <path
          key={`${d.source}-${d.target}`}
          d={d.pathStr}
          fill="none"
          stroke="#94a3b8"
          strokeWidth={getEdgeStrokeWidth(d.weight) * strokeWidthScale}
          opacity={opacity}
          style={{
            pointerEvents: 'none',
            transition:
              'opacity 0.3s ease, stroke-width 0.3s ease, stroke 0.3s ease',
          }}
        />
      ))}
    </g>
  )
})

Links.displayName = 'Links'
