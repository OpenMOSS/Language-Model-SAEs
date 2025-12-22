import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface YAxisProps {
  positionedNodes: { layer: number }[]
  y: d3.ScaleBand<number>
}

const SIDE_PADDING = 70

export const YAxis: React.FC<YAxisProps> = React.memo(
  ({ positionedNodes, y }) => {
    const svgRef = useRef<SVGGElement>(null)

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const yNumTicks = (d3.max(positionedNodes, (d) => d.layer) || 0) + 2

      d3.range(yNumTicks).forEach((layerIdx: number) => {
        const yPos = (y(layerIdx) || 0) + y.bandwidth() / 2

        let label: string
        let textColor: string
        if (layerIdx === 0) {
          label = 'Emb'
          textColor = '#6b21a8' // Embedding - purple
        } else if (layerIdx === yNumTicks - 1) {
          label = 'Logit'
          textColor = '#9a3412' // Logits - orange
        } else if (layerIdx % 2 === 0) {
          label = `M${Math.floor(layerIdx / 2) - 1}`
          textColor = '#166534' // MLP - green
        } else {
          label = `A${Math.floor(layerIdx / 2)}`
          textColor = '#1e40af' // Attention - blue
        }

        svg
          .append('text')
          .attr('x', SIDE_PADDING - 12)
          .attr('y', yPos + 4)
          .attr('text-anchor', 'end')
          .attr(
            'font-size',
            yNumTicks > 50
              ? '7px'
              : yNumTicks > 40
                ? '8px'
                : yNumTicks > 20
                  ? '10px'
                  : '12px',
          )
          .style(
            'font-family',
            "'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Consolas', 'Courier New', monospace",
          )
          .style('user-select', 'none')
          .attr('fill', textColor)
          .text(label)
      })
    }, [positionedNodes, y])

    return <g ref={svgRef} />
  },
)

YAxis.displayName = 'YAxis'
