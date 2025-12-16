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
import type {
  CircuitData,
  PositionedEdge,
  PositionedNode,
  VisState,
} from '@/types/circuit'

interface LinkGraphProps {
  data: CircuitData
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

  const { calculatedCtxCounts, x, y, positionedNodes, positionedEdges } =
    useMemo(() => {
      if (!data.nodes.length) {
        return {
          calculatedCtxCounts: [],
          x: null,
          y: null,
          positionedNodes: [],
          positionedEdges: [],
        }
      }

      const { nodes } = data
      const earliestCtxWithNodes = d3.min(nodes, (d) => d.ctxIdx) || 0

      let cumsum = 0
      const calculatedCtxCounts = d3
        .range((d3.max(nodes, (d) => d.ctxIdx) || 0) + 1)
        .map((ctxIdx: number) => {
          if (ctxIdx >= earliestCtxWithNodes) {
            const group = nodes.filter((d) => d.ctxIdx === ctxIdx)
            const layerGroups = d3.group(group, (d) => d.layer)
            const maxNodesPerLayer =
              d3.max(
                Array.from(layerGroups.values()),
                (layerNodes) => layerNodes.length,
              ) || 1
            const maxCount = Math.max(1, maxNodesPerLayer)
            cumsum += maxCount
            return { ctxIdx, maxCount, cumsum, layerGroups }
          }
          return { ctxIdx, maxCount: 0, cumsum, layerGroups: new Map() }
        })

      const xDomain = [-1].concat(calculatedCtxCounts.map((d) => d.ctxIdx))
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

      const yNumTicks = (d3.max(nodes, (d) => d.layer) || 0) + 2
      const y = d3.scaleBand<number>(d3.range(yNumTicks), [
        dimensions.height - BOTTOM_PADDING,
        0,
      ])

      calculatedCtxCounts.forEach((d: any) => {
        d.width = x(d.ctxIdx + 1) - x(d.ctxIdx)
      })

      const padR =
        Math.min(
          8,
          d3.min(calculatedCtxCounts.slice(1), (d: any) => d.width / 2) || 8,
        ) + 0

      const positionedNodes: PositionedNode[] = nodes.map((node) => ({
        ...node,
        pos: [0, 0] as [number, number],
      }))

      const xOffsets = new Map<string, number>()

      calculatedCtxCounts.forEach((ctxData: any) => {
        if (ctxData.layerGroups.size === 0) return

        const ctxWidth = x(ctxData.ctxIdx + 1) - x(ctxData.ctxIdx) - padR

        ctxData.layerGroups.forEach(
          (layerNodes: typeof nodes, _layerIdx: number) => {
            const sortedNodes = [...layerNodes].sort(
              (a, b) => -(a.tokenProb || 0) + (b.tokenProb || 0),
            )

            const maxNodesInContext = ctxData.maxCount
            const spacing = ctxWidth / maxNodesInContext

            sortedNodes.forEach((node, i) => {
              const totalWidth = (sortedNodes.length - 1) * spacing
              const startX = ctxWidth - totalWidth
              xOffsets.set(node.nodeId, startX + i * spacing)
            })
          },
        )
      })

      positionedNodes.forEach((d) => {
        const xOffset = xOffsets.get(d.nodeId) || 0
        d.pos = [
          x(d.ctxIdx) + xOffset,
          (y(d.layer + 1) || 0) + y.bandwidth() / 2,
        ]
      })

      const positionedEdges: PositionedEdge[] = data.edges
        .map((edge) => {
          const sourceNode = positionedNodes.find(
            (n) => n.nodeId === edge.source,
          )
          const targetNode = positionedNodes.find(
            (n) => n.nodeId === edge.target,
          )
          if (sourceNode && targetNode) {
            const [x1, y1] = sourceNode.pos
            const [x2, y2] = targetNode.pos
            return {
              ...edge,
              pathStr: `M${x1},${y1}L${x2},${y2}`,
            }
          }
          return null
        })
        .filter((edge): edge is PositionedEdge => edge !== null)

      return { calculatedCtxCounts, x, y, positionedNodes, positionedEdges }
    }, [data.nodes, data.edges, dimensions.width, dimensions.height])

  const handleNodeMouseEnter = useCallback(
    (nodeId: string) => {
      onNodeHover(nodeId)
    },
    [onNodeHover],
  )

  const handleNodeMouseLeave = useCallback(() => {
    onNodeHover(null)
  }, [onNodeHover])

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

  const tokenData = useMemo(() => {
    if (!data.metadata?.promptTokens || !positionedNodes.length || !x) return []

    const maxCtxIdx = d3.max(positionedNodes, (d) => d.ctxIdx) || 0

    return data.metadata.promptTokens
      .slice(0, maxCtxIdx + 1)
      .map((token: string, index: number) => {
        const contextNodes = positionedNodes.filter((d) => d.ctxIdx === index)

        if (contextNodes.length === 0) {
          return {
            token,
            ctxIdx: index,
            x: x(index + 1) - (x(index + 1) - x(index)) / 2,
          }
        }

        const nodeXPositions = contextNodes.map((d) => d.pos[0])
        const rightX = Math.max(...nodeXPositions)

        return {
          token,
          ctxIdx: index,
          x: rightX,
        }
      })
  }, [data.metadata?.promptTokens, positionedNodes, x])

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
        className="relative z-1"
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

        <Links positionedEdges={positionedEdges} />

        <Nodes
          positionedNodes={positionedNodes}
          positionedEdges={positionedEdges}
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
