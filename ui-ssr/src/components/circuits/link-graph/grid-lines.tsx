import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface GridLinesProps {
  dimensions: { width: number; height: number }
  calculatedCtxCounts: {
    ctx_idx: number
    maxCount: number
    cumsum: number
  }[]
  x: d3.ScaleLinear<number, number>
  positionedNodes: { ctx_idx: number }[]
}

const BOTTOM_PADDING = 40

export const GridLines: React.FC<GridLinesProps> = React.memo(
  ({ dimensions, calculatedCtxCounts, x, positionedNodes }) => {
    const svgRef = useRef<SVGGElement>(null)

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const earliestCtxWithNodes =
        d3.min(positionedNodes, (d) => d.ctx_idx) || 0

      calculatedCtxCounts.forEach((ctxData) => {
        if (ctxData.ctx_idx >= earliestCtxWithNodes) {
          const xPos = x(ctxData.ctx_idx)
          svg
            .append('line')
            .attr('x1', xPos)
            .attr('y1', 0)
            .attr('x2', xPos)
            .attr('y2', dimensions.height - BOTTOM_PADDING)
            .attr('stroke', 'rgba(255, 255, 255, 1)')
            .attr('stroke-width', '1')
        }
      })
    }, [dimensions, calculatedCtxCounts, x, positionedNodes])

    return <g ref={svgRef} />
  },
)

GridLines.displayName = 'GridLines'
