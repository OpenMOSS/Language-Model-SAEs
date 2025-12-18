import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface RowBackgroundsProps {
  dimensions: { width: number; height: number }
  positionedNodes: { layer: number }[]
  y: d3.ScaleBand<number>
}

const SIDE_PADDING = 70

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
        const rowHeight = y.bandwidth() - 1

        let backgroundColor: string
        if (layerIdx === 0) {
          backgroundColor = '#e5e7eb' // Embedding - slate with slight purple tint
        } else if (layerIdx === yNumTicks - 1) {
          backgroundColor = '#e8e4e0' // Logits - slate with slight warm tint
        } else if (layerIdx % 2 === 0) {
          backgroundColor = '#e2e8f0' // MLP - original slate
        } else {
          backgroundColor = '#e0e7ef' // Attention - slate with slight blue tint
        }

        svg
          .append('rect')
          .attr('x', SIDE_PADDING)
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
