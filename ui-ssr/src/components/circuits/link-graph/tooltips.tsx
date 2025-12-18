import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { PositionedNode } from '@/types/circuit'

interface TooltipsProps {
  positionedNodes: PositionedNode[]
  visState: { hoveredId: string | null }
  dimensions: { width: number; height: number }
}

const BOTTOM_PADDING = 40

export const Tooltips: React.FC<TooltipsProps> = React.memo(
  ({ positionedNodes, visState, dimensions }) => {
    const svgRef = useRef<SVGGElement>(null)
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    useEffect(() => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }, [visState.hoveredId])

    useEffect(() => {
      if (visState.hoveredId && !timeoutRef.current) {
        timeoutRef.current = setTimeout(() => {
          timeoutRef.current = null
        }, 10000)
      }

      return () => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
          timeoutRef.current = null
        }
      }
    }, [visState.hoveredId])

    useEffect(() => {
      if (!svgRef.current) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      if (!visState.hoveredId) {
        return
      }

      const hoveredNode = positionedNodes.find(
        (d) => d.nodeId === visState.hoveredId,
      )

      if (hoveredNode) {
        const tooltip = svg.append('g')

        const tooltipText =
          hoveredNode.clerp ||
          `Feature: ${hoveredNode.feature} (Layer ${hoveredNode.layer})`

        const textWidth = tooltipText.length * 6
        const tooltipWidth = Math.max(120, textWidth + 20)
        const tooltipHeight = 20
        const padding = 10

        let tooltipX = hoveredNode.pos[0] + padding
        let tooltipY = hoveredNode.pos[1] - 15

        if (tooltipX + tooltipWidth > dimensions.width - padding) {
          tooltipX = hoveredNode.pos[0] - tooltipWidth - padding
        }

        if (tooltipY < padding) {
          tooltipY = hoveredNode.pos[1] + padding
        }

        if (
          tooltipY + tooltipHeight >
          dimensions.height - BOTTOM_PADDING - padding
        ) {
          tooltipY = hoveredNode.pos[1] - tooltipHeight - padding
        }

        tooltip
          .append('rect')
          .attr('x', tooltipX)
          .attr('y', tooltipY)
          .attr('width', tooltipWidth)
          .attr('height', tooltipHeight)
          .attr('fill', 'rgba(0, 0, 0, 0.8)')
          .attr('rx', 2)

        tooltip
          .append('text')
          .attr('x', tooltipX + 5)
          .attr('y', tooltipY + 13)
          .attr('fill', 'white')
          .attr('font-size', '10px')
          .text(tooltipText)
      }
    }, [positionedNodes, visState.hoveredId, dimensions])

    return <g ref={svgRef} />
  },
)

Tooltips.displayName = 'Tooltips'
