import React, { useCallback, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { PositionedNode } from '@/types/circuit'
import type { EdgeIndex } from '@/utils/circuit-index'
import { getNodeColor } from '@/utils/circuit'
import { isNodeConnected } from '@/utils/circuit-index'

interface NodesProps {
  positionedNodes: PositionedNode[]
  edgeIndex: EdgeIndex
  visState: {
    clickedId: string | null
    hoveredId: string | null
  }
}

export const Nodes: React.FC<NodesProps> = React.memo(
  ({ positionedNodes, edgeIndex, visState }) => {
    const svgRef = useRef<SVGGElement>(null)

    const isConnected = useCallback(
      (nodeId: string) => {
        if (!visState.clickedId) return false
        return isNodeConnected(edgeIndex, visState.clickedId, nodeId)
      },
      [visState.clickedId, edgeIndex],
    )

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const isErrorNode = (d: any) =>
        d.featureType === 'lorsa error' ||
        d.featureType === 'mlp reconstruction error'

      const regularNodes = positionedNodes.filter((d) => !isErrorNode(d))
      const errorNodes = positionedNodes.filter((d) => isErrorNode(d))

      svg
        .selectAll('circle.node')
        .data(regularNodes, (d: any) => d.nodeId)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', (d: any) => d.pos[0])
        .attr('cy', (d: any) => d.pos[1])
        .attr('r', 3)
        .attr('fill', (d: any) => getNodeColor(d.featureType))
        .attr('stroke', (d: any) => {
          if (d.nodeId === visState.clickedId) return '#ef4444'
          return '#000'
        })
        .attr('stroke-width', (d: any) => {
          if (d.nodeId === visState.clickedId || isConnected(d.nodeId))
            return '1.5'
          return '0.5'
        })
        .style('pointer-events', 'none')
        .style('transition', 'all 0.2s ease')

      svg
        .selectAll('rect.node')
        .data(errorNodes, (d: any) => d.nodeId)
        .enter()
        .append('rect')
        .attr('class', 'node')
        .attr('x', (d: any) => d.pos[0] - 3)
        .attr('y', (d: any) => d.pos[1] - 3)
        .attr('width', 6)
        .attr('height', 6)
        .attr('fill', (d: any) => getNodeColor(d.featureType))
        .attr('stroke', (d: any) => {
          if (d.nodeId === visState.clickedId) return '#ef4444'
          return '#000'
        })
        .attr('stroke-width', (d: any) => {
          if (d.nodeId === visState.clickedId || isConnected(d.nodeId))
            return '1.5'
          return '0.5'
        })
        .style('pointer-events', 'none')
        .style('transition', 'all 0.2s ease')

      svg
        .selectAll('circle.hover-indicator')
        .data(regularNodes, (d: any) => d.nodeId)
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

      svg
        .selectAll('rect.hover-indicator')
        .data(errorNodes, (d: any) => d.nodeId)
        .enter()
        .append('rect')
        .attr('class', 'hover-indicator')
        .attr('x', (d: any) => d.pos[0] - 6)
        .attr('y', (d: any) => d.pos[1] - 6)
        .attr('width', 12)
        .attr('height', 12)
        .attr('stroke', '#f0f')
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .style('pointer-events', 'none')
        .style('opacity', (d: any) => (d.nodeId === visState.hoveredId ? 1 : 0))
    }, [positionedNodes, edgeIndex, visState, isConnected])

    return <g ref={svgRef} />
  },
)

Nodes.displayName = 'Nodes'
