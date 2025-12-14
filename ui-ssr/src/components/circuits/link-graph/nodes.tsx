import React, { useCallback, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { Link, Node } from '@/types/circuit'

interface NodesProps {
  positionedNodes: Node[]
  positionedLinks: Link[]
  visState: {
    clickedId: string | null
    hoveredId: string | null
  }
  onNodeMouseEnter: (nodeId: string) => void
  onNodeMouseLeave: () => void
  onNodeClick: (nodeId: string, metaKey: boolean) => void
}

export const Nodes: React.FC<NodesProps> = React.memo(
  ({
    positionedNodes,
    positionedLinks,
    visState,
    onNodeMouseEnter,
    onNodeMouseLeave,
    onNodeClick,
  }) => {
    const svgRef = useRef<SVGGElement>(null)

    const handleMouseEnter = useCallback(
      (nodeId: string) => {
        onNodeMouseEnter(nodeId)
      },
      [onNodeMouseEnter],
    )

    const handleMouseLeave = useCallback(() => {
      onNodeMouseLeave()
    }, [onNodeMouseLeave])

    const handleClick = useCallback(
      (nodeId: string, metaKey: boolean) => {
        onNodeClick(nodeId, metaKey)
      },
      [onNodeClick],
    )

    const isConnected = useCallback(
      (nodeId: string) => {
        if (!visState.clickedId) return false
        return positionedLinks.some(
          (link) =>
            (link.source === visState.clickedId && link.target === nodeId) ||
            (link.target === visState.clickedId && link.source === nodeId),
        )
      },
      [visState.clickedId, positionedLinks],
    )

    useEffect(() => {
      if (!svgRef.current || !positionedNodes.length) return

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const nodeSel = svg
        .selectAll('circle.node')
        .data(positionedNodes, (d: any) => d.nodeId)

      const nodeEnter = nodeSel
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', (d: any) => d.pos[0])
        .attr('cy', (d: any) => d.pos[1])
        .attr('r', (d: any) => (isConnected(d.nodeId) ? 6 : 4))
        .attr('fill', (d: any) => d.nodeColor)
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
        .style('cursor', 'pointer')
        .style('transition', 'all 0.2s ease')
        .on('mouseenter', function (_event: any, d: any) {
          handleMouseEnter(d.nodeId)
        })
        .on('mouseleave', function () {
          handleMouseLeave()
        })
        .on('click', function (event: any, d: any) {
          event.stopPropagation()
          const metaKey = event.metaKey || event.ctrlKey
          handleClick(d.nodeId, metaKey)
        })

      nodeSel
        .merge(nodeEnter as any)
        .attr('cx', (d: any) => d.pos[0])
        .attr('cy', (d: any) => d.pos[1])
        .attr('r', (d: any) => (isConnected(d.nodeId) ? 6 : 4))
        .attr('fill', (d: any) => d.nodeColor)
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
        .style('cursor', 'pointer')
        .style('transition', 'all 0.2s ease')
        .on('mouseenter', function (_event: any, d: any) {
          handleMouseEnter(d.nodeId)
        })
        .on('mouseleave', function () {
          handleMouseLeave()
        })
        .on('click', function (event: any, d: any) {
          event.stopPropagation()
          const metaKey = event.metaKey || event.ctrlKey
          handleClick(d.nodeId, metaKey)
        })

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

      return () => {
        handleMouseLeave()
      }
    }, [
      positionedNodes,
      positionedLinks,
      visState,
      handleMouseEnter,
      handleMouseLeave,
      handleClick,
      isConnected,
    ])

    return <g ref={svgRef} />
  },
)

Nodes.displayName = 'Nodes'
