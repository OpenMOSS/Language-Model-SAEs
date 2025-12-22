import { memo, useEffect, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import type { VisState } from '@/types/circuit'
import type { EdgeIndex } from '@/utils/circuit-index'
import { getEdgeStrokeWidth } from '@/utils/circuit'
import { getConnectedEdges, getTopEdgesByWeight } from '@/utils/circuit-index'

interface LinksProps {
  edgeIndex: EdgeIndex
  visState: VisState
}

export const Links = memo(({ edgeIndex, visState }: LinksProps) => {
  const svgRef = useRef<SVGGElement>(null)

  const isFilteredView = !!visState.clickedId

  const connectedEdges = useMemo(() => {
    if (!visState.clickedId) return getTopEdgesByWeight(edgeIndex, 600)
    return getConnectedEdges(edgeIndex, visState.clickedId)
  }, [edgeIndex, visState.clickedId])

  useEffect(() => {
    if (!svgRef.current || !connectedEdges.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Styling based on view mode:
    // - "Overview" mode (top 600 edges by weight): subtle, low opacity
    // - "Connected" mode (~100 edges): prominent, color-coded by weight sign
    const opacity = isFilteredView ? 0.6 : 0.4
    const strokeWidthScale = isFilteredView ? 0.5 : 0.35

    const edgeSel = svg
      .selectAll('path')
      .data(connectedEdges, (d: any) => `${d.source}-${d.target}`)

    const edgeEnter = edgeSel
      .enter()
      .append('path')
      .attr('fill', 'none')
      .style('pointer-events', 'none')
      .style(
        'transition',
        'opacity 0.3s ease, stroke-width 0.3s ease, stroke 0.3s ease',
      )

    edgeSel
      .merge(edgeEnter as any)
      .attr('d', (d: any) => d.pathStr)
      .attr('stroke', '#94a3b8')
      .attr(
        'stroke-width',
        (d: any) => getEdgeStrokeWidth(d.weight) * strokeWidthScale,
      )
      .attr('opacity', opacity)
  }, [connectedEdges, isFilteredView])

  return <g ref={svgRef} />
})

Links.displayName = 'Links'
