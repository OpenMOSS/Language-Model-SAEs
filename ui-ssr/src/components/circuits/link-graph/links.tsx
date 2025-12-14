import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { PositionedEdge } from '@/types/circuit'
import { getEdgeStrokeWidth } from '@/utils/circuit'

interface LinksProps {
  positionedEdges: PositionedEdge[]
}

export const Links: React.FC<LinksProps> = React.memo(({ positionedEdges }) => {
  const svgRef = useRef<SVGGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !positionedEdges.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const edgeSel = svg
      .selectAll('path')
      .data(positionedEdges, (d: any) => `${d.source}-${d.target}`)

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
      .attr('stroke', '#666666')
      .attr('stroke-width', (d: any) => getEdgeStrokeWidth(d.weight))
      .attr('opacity', 0.03)
  }, [positionedEdges])

  return <g ref={svgRef} />
})

Links.displayName = 'Links'
