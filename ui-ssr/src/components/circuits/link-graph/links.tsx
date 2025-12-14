import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface LinksProps {
  positionedLinks: {
    source: string
    target: string
  }[]
}

export const Links: React.FC<LinksProps> = React.memo(({ positionedLinks }) => {
  const svgRef = useRef<SVGGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !positionedLinks.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Draw edges as SVG paths
    const linkSel = svg
      .selectAll('path')
      .data(positionedLinks, (d: any) => `${d.source}-${d.target}`)

    // Enter: create new paths
    const linkEnter = linkSel
      .enter()
      .append('path')
      .attr('fill', 'none')
      .style('pointer-events', 'none')
      .style(
        'transition',
        'opacity 0.3s ease, stroke-width 0.3s ease, stroke 0.3s ease',
      )

    // Merge enter and update selections
    linkSel
      .merge(linkEnter as any)
      .attr('d', (d: any) => d.pathStr)
      .attr('stroke', '#666666')
      .attr('stroke-width', (d: any) => d.strokeWidth || 1)
      .attr('opacity', 0.03)
  }, [positionedLinks])

  return <g ref={svgRef} />
})

Links.displayName = 'Links'
