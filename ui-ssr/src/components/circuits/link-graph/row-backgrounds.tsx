import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface RowBackgroundsProps {
  dimensions: { width: number; height: number }
  positionedNodes: { layer: number }[]
  y: d3.ScaleBand<number>
}

export const RowBackgrounds: React.FC<RowBackgroundsProps> = React.memo(
  ({ dimensions, positionedNodes, y }) => {
    const svgRef = useRef<SVGGElement>(null)

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const yNumTicks = (d3.max(positionedNodes, (d) => d.layer) || 0) + 2

      d3.range(yNumTicks).forEach((layerIdx: number) => {
        const yPos = y(layerIdx) || 0
        const rowHeight = y.bandwidth()
        const backgroundColor = layerIdx % 2 === 0 ? '#F5F4EE' : '#EBE9E0'

        svg
          .append('rect')
          .attr('x', 0)
          .attr('y', yPos)
          .attr('width', dimensions.width)
          .attr('height', rowHeight)
          .attr('fill', backgroundColor)
      })
    }, [dimensions, positionedNodes, y])

    return <g ref={svgRef} />
  },
)

RowBackgrounds.displayName = 'RowBackgrounds'
