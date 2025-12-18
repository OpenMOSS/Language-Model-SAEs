import React, { useCallback, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { PositionedEdge, PositionedNode } from '@/types/circuit'
import { getNodeColor } from '@/utils/circuit'

interface NodesProps {
  positionedNodes: PositionedNode[]
  positionedEdges: PositionedEdge[]
  visState: {
    clickedId: string | null
    hoveredId: string | null
  }
}

export const Nodes: React.FC<NodesProps> = React.memo(
  ({ positionedNodes, positionedEdges, visState }) => {
    const svgRef = useRef<SVGGElement>(null)

    const isConnected = useCallback(
      (nodeId: string) => {
        if (!visState.clickedId) return false
        return positionedEdges.some(
          (edge) =>
            (edge.source === visState.clickedId && edge.target === nodeId) ||
            (edge.target === visState.clickedId && edge.source === nodeId),
        )
      },
      [visState.clickedId, positionedEdges],
    )

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      svg
        .selectAll('circle.node')
        .data(positionedNodes, (d: any) => d.nodeId)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', (d: any) => d.pos[0])
        .attr('cy', (d: any) => d.pos[1])
        .attr('r', 3)
        .attr('fill', (d: any) => getNodeColor(d.featureType))
        .attr('stroke', (d: any) => {
          if (d.nodeId === visState.clickedId) return '#ef4444'
          if (isConnected(d.nodeId)) return '#10b981'
          return '#000'
        })
        .attr('stroke-width', (d: any) => {
          if (d.nodeId === visState.clickedId || isConnected(d.nodeId))
            return '2'
          return '0.5'
        })
        .style('pointer-events', 'none')
        .style('transition', 'all 0.2s ease')

      svg
        .selectAll('circle.hover-indicator')
        .data(positionedNodes, (d: any) => d.nodeId)
        .enter()
        .append('circle')
        .attr('class', 'hover-indicator')
        .attr('cx', (d: any) => d.pos[0])
        .attr('cy', (d: any) => d.pos[1])
        .attr('r', 6)
        .attr('stroke', '#f0f')
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .style('pointer-events', 'none')
        .style('opacity', (d: any) => (d.nodeId === visState.hoveredId ? 1 : 0))
    }, [positionedNodes, positionedEdges, visState, isConnected])

    return <g ref={svgRef} />
  },
)

Nodes.displayName = 'Nodes'
