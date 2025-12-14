import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import {
  GridLines,
  Links,
  Nodes,
  RowBackgrounds,
  TokenLabels,
  Tooltips,
  YAxis,
} from './index'
import type { Link, LinkGraphData, Node, VisState } from '@/types/circuit'

interface LinkGraphProps {
  data: LinkGraphData
  visState: VisState
  onNodeClick: (nodeId: string, metaKey: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

const BOTTOM_PADDING = 50
const SIDE_PADDING = 20

const LinkGraphComponent: React.FC<LinkGraphProps> = ({
  data,
  visState,
  onNodeClick,
  onNodeHover,
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  // Memoize expensive calculations
  const { calculatedCtxCounts, x, y, positionedNodes, positionedLinks } =
    useMemo(() => {
      if (!data.nodes.length) {
        return {
          calculatedCtxCounts: [],
          x: null,
          y: null,
          positionedNodes: [],
          positionedLinks: [],
        }
      }

      const { nodes } = data
      const earliestCtxWithNodes = d3.min(nodes, (d) => d.ctx_idx) || 0

      let cumsum = 0
      const calculatedCtxCounts = d3
        .range((d3.max(nodes, (d) => d.ctx_idx) || 0) + 1)
        .map((ctx_idx: number) => {
          if (ctx_idx >= earliestCtxWithNodes) {
            const group = nodes.filter((d) => d.ctx_idx === ctx_idx)
            const layerGroups = d3.group(group, (d) => d.layerIdx)
            const maxNodesPerLayer =
              d3.max(
                Array.from(layerGroups.values()),
                (layerNodes) => layerNodes.length,
              ) || 1
            const maxCount = Math.max(1, maxNodesPerLayer)
            cumsum += maxCount
            return { ctx_idx, maxCount, cumsum, layerGroups }
          }
          return { ctx_idx, maxCount: 0, cumsum, layerGroups: new Map() }
        })

      const xDomain = [-1].concat(calculatedCtxCounts.map((d) => d.ctx_idx))
      const xRange = [SIDE_PADDING].concat(
        calculatedCtxCounts.map(
          (d) =>
            SIDE_PADDING +
            (d.cumsum * (dimensions.width - 2 * SIDE_PADDING)) / cumsum,
        ),
      )
      const x = d3
        .scaleLinear()
        .domain(xDomain.map((d) => d + 1))
        .range(xRange)

      const yNumTicks = (d3.max(nodes, (d) => d.layerIdx) || 0) + 1
      const y = d3.scaleBand<number>(d3.range(yNumTicks), [
        dimensions.height - BOTTOM_PADDING,
        0,
      ])

      // Position nodes
      calculatedCtxCounts.forEach((d: any) => {
        d.width = x(d.ctx_idx + 1) - x(d.ctx_idx)
      })

      const padR =
        Math.min(
          8,
          d3.min(calculatedCtxCounts.slice(1), (d: any) => d.width / 2) || 8,
        ) + 0

      // Create a copy of nodes to avoid mutating the original data
      const positionedNodes: Node[] = nodes.map((node) => ({ ...node }))

      // Position nodes within each context and layer
      calculatedCtxCounts.forEach((ctxData: any) => {
        if (ctxData.layerGroups.size === 0) return

        const ctxWidth = x(ctxData.ctx_idx + 1) - x(ctxData.ctx_idx) - padR

        ctxData.layerGroups.forEach((layerNodes: Node[]) => {
          const sortedNodes = [...layerNodes].sort(
            (a, b) => -(a.logitPct || 0) + (b.logitPct || 0),
          )

          const maxNodesInContext = ctxData.maxCount
          const spacing = ctxWidth / maxNodesInContext

          sortedNodes.forEach((node, i) => {
            const totalWidth = (sortedNodes.length - 1) * spacing
            const startX = ctxWidth - totalWidth
            // Find the node in positionedNodes and update it
            const posNode = positionedNodes.find(
              (n) => n.nodeId === node.nodeId,
            )
            if (posNode) {
              posNode.xOffset = startX + i * spacing
              posNode.yOffset = 0
            }
          })
        })
      })

      positionedNodes.forEach((d) => {
        d.pos = [
          x(d.ctx_idx) + d.xOffset,
          (y(d.layerIdx) || 0) + y.bandwidth() / 2 + d.yOffset,
        ]
      })

      // Update link paths and populate node link references
      const positionedLinks: Link[] = data.links
        .map((d) => {
          const sourceNode = positionedNodes.find((n) => n.nodeId === d.source)
          const targetNode = positionedNodes.find((n) => n.nodeId === d.target)
          if (sourceNode && targetNode) {
            const [x1, y1] = sourceNode.pos
            const [x2, y2] = targetNode.pos
            return {
              ...d,
              pathStr: `M${x1},${y1}L${x2},${y2}`,
            }
          }
          return null
        })
        .filter((link): link is Link => link !== null)

      return { calculatedCtxCounts, x, y, positionedNodes, positionedLinks }
    }, [data.nodes, data.links, dimensions.width, dimensions.height])

  // Handle mouse enter/leave
  const handleNodeMouseEnter = useCallback(
    (nodeId: string) => {
      onNodeHover(nodeId)
    },
    [onNodeHover],
  )

  const handleNodeMouseLeave = useCallback(() => {
    onNodeHover(null)
  }, [onNodeHover])

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setDimensions({
          width: rect.width,
          height: rect.height,
        })
      }
    }

    updateDimensions()

    const resizeObserver = new ResizeObserver(updateDimensions)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  // Memoize token data calculation
  const tokenData = useMemo(() => {
    if (!data.metadata?.prompt_tokens || !positionedNodes.length || !x)
      return []

    const maxCtxIdx = d3.max(positionedNodes, (d) => d.ctx_idx) || 0

    return data.metadata.prompt_tokens
      .slice(0, maxCtxIdx + 1)
      .map((token: string, index: number) => {
        const contextNodes = positionedNodes.filter((d) => d.ctx_idx === index)

        if (contextNodes.length === 0) {
          return {
            token,
            ctx_idx: index,
            x: x(index + 1) - (x(index + 1) - x(index)) / 2,
          }
        }

        const nodeXPositions = contextNodes.map((d) => d.pos[0])
        const rightX = Math.max(...nodeXPositions)

        return {
          token,
          ctx_idx: index,
          x: rightX,
        }
      })
  }, [data.metadata?.prompt_tokens, positionedNodes, x])

  // Early return if no data or scales
  if (!positionedNodes.length || !x || !y) {
    return (
      <div
        ref={containerRef}
        className="relative w-full h-[600px] border border-gray-200 rounded-lg overflow-hidden"
      >
        <div>Loading...</div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[600px] border border-gray-200 rounded-lg overflow-hidden"
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            onNodeClick('', false)
          }
        }}
        className="relative z-[1]"
        style={{ pointerEvents: 'auto' }}
      >
        <RowBackgrounds
          dimensions={dimensions}
          positionedNodes={positionedNodes}
          y={y}
        />

        <GridLines
          dimensions={dimensions}
          calculatedCtxCounts={calculatedCtxCounts}
          x={x}
          positionedNodes={positionedNodes}
        />

        <YAxis positionedNodes={positionedNodes} y={y} />

        <Links positionedLinks={positionedLinks} />

        <Nodes
          positionedNodes={positionedNodes}
          positionedLinks={positionedLinks}
          visState={{
            clickedId: visState.clickedId,
            hoveredId: visState.hoveredId,
          }}
          onNodeMouseEnter={handleNodeMouseEnter}
          onNodeMouseLeave={handleNodeMouseLeave}
          onNodeClick={onNodeClick}
        />

        <Tooltips
          positionedNodes={positionedNodes}
          visState={{ hoveredId: visState.hoveredId }}
          dimensions={dimensions}
        />

        <TokenLabels tokenData={tokenData} dimensions={dimensions} />
      </svg>
    </div>
  )
}

export const LinkGraph = React.memo(LinkGraphComponent)
