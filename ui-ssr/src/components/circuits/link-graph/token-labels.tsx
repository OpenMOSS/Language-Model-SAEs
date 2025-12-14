import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface TokenLabelsProps {
  tokenData: { token: string; ctx_idx: number; x: number }[]
  dimensions: { width: number; height: number }
}

const BOTTOM_PADDING = 40

export const TokenLabels: React.FC<TokenLabelsProps> = React.memo(
  ({ tokenData, dimensions }) => {
    const svgRef = useRef<SVGGElement>(null)

    useEffect(() => {
      if (!svgRef.current || !tokenData.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      // Draw prompt tokens at the bottom
      svg
        .selectAll('text')
        .data(tokenData, (d: any) => d.ctx_idx)
        .enter()
        .append('text')
        .attr('x', (d: any) => d.x)
        .attr('y', dimensions.height - BOTTOM_PADDING + 10)
        .attr(
          'transform',
          (d: any) =>
            `rotate(-45, ${d.x}, ${dimensions.height - BOTTOM_PADDING + 10})`,
        )
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'hanging')
        .style('font-family', "'Courier New', monospace")
        .style('font-size', '11px')
        .style('font-weight', '500')
        .style('fill', '#374151')
        .text((d: any) => d.token)
    }, [tokenData, dimensions])

    return <g ref={svgRef} />
  },
)

TokenLabels.displayName = 'TokenLabels'
