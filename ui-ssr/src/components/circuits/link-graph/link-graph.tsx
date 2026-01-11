import React, { useCallback, useEffect, useMemo, useRef } from 'react'
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
  Node,
  PositionedNode,
  VisState,
} from '@/types/circuit'
import {
  createEdgeIndex,
  createNodeIndex,
  createPositionedEdges,
  createSpatialIndex,
  findNearestNode,
  getNodesByCtxIdx,
} from '@/utils/circuit-index'
import { useLocalStorage } from '@/hooks/useLocalStorage'

interface LinkGraphProps {
  data: CircuitData
  visState: VisState
  onNodeClick: (nodeId: string, isMultiSelect: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

const BOTTOM_PADDING = 50
const SIDE_PADDING = 70
const MIN_UNIT_WIDTH = 8

function topologicalSort(
  nodes: Node[],
  outgoingEdges: Map<string, Set<string>>,
): Node[] {
  const nodeIds = new Set(nodes.map((n) => n.nodeId))
  const inDegree = new Map<string, number>()
  const localOutgoing = new Map<string, string[]>()

  for (const node of nodes) {
    inDegree.set(node.nodeId, 0)
    localOutgoing.set(node.nodeId, [])
  }

  for (const node of nodes) {
    const targets = outgoingEdges.get(node.nodeId)
    if (targets) {
      for (const target of targets) {
        if (nodeIds.has(target)) {
          localOutgoing.get(node.nodeId)!.push(target)
          inDegree.set(target, (inDegree.get(target) || 0) + 1)
        }
      }
    }
  }

  // Kahn's algorithm
  const queue: Node[] = []
  const nodeMap = new Map(nodes.map((n) => [n.nodeId, n]))

  for (const node of nodes) {
    if (inDegree.get(node.nodeId) === 0) {
      queue.push(node)
    }
  }

  const sorted: Node[] = []
  while (queue.length > 0) {
    const node = queue.shift()!
    sorted.push(node)

    for (const targetId of localOutgoing.get(node.nodeId) || []) {
      const newDegree = (inDegree.get(targetId) || 1) - 1
      inDegree.set(targetId, newDegree)
      if (newDegree === 0) {
        queue.push(nodeMap.get(targetId)!)
      }
    }
  }

  // Handle cycles by adding remaining nodes
  if (sorted.length < nodes.length) {
    for (const node of nodes) {
      if (!sorted.includes(node)) {
        sorted.push(node)
      }
    }
  }

  return sorted
}

const LinkGraphComponent: React.FC<LinkGraphProps> = ({
  data: rawData,
  visState,
  onNodeClick,
  onNodeHover,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerDimensions, setContainerDimensions] = useLocalStorage(
    'linkGraphDimensions',
    {
      width: 800,
      height: 600,
    },
  )

  const data = useMemo(() => {
    const nodes = rawData.nodes.filter((n) => !n.isFromQkTracing)
    const nodeIds = new Set(nodes.map((n) => n.nodeId))
    const edges = rawData.edges.filter(
      (e) => nodeIds.has(e.source) && nodeIds.has(e.target),
    )
    return { ...rawData, nodes, edges }
  }, [rawData])

  // 1. Calculate stats about context counts and total units needed
  const { calculatedCtxCounts, totalUnits } = useMemo(() => {
    if (!data.nodes.length) {
      return { calculatedCtxCounts: [], totalUnits: 0 }
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

    const totalUnits = cumsum + 2 * calculatedCtxCounts.length

    return { calculatedCtxCounts, totalUnits }
  }, [data.nodes])

  // 2. Calculate graph dimensions based on data density and container size
  const dimensions = useMemo(() => {
    const minWidth = totalUnits * MIN_UNIT_WIDTH + SIDE_PADDING
    return {
      width: Math.max(containerDimensions.width, minWidth),
      height: containerDimensions.height,
    }
  }, [totalUnits, containerDimensions])

  // 3. Calculate scales and node positions
  const { x, y, positionedNodes } = useMemo(() => {
    if (!calculatedCtxCounts.length) {
      return {
        x: null,
        y: null,
        positionedNodes: [],
      }
    }

    const { nodes, edges } = data

    // Build outgoing edges map for topological sorting
    const outgoingEdges = new Map<string, Set<string>>()
    for (const edge of edges) {
      if (!outgoingEdges.has(edge.source)) {
        outgoingEdges.set(edge.source, new Set())
      }
      outgoingEdges.get(edge.source)!.add(edge.target)
    }

    const xDomain = [-1].concat(calculatedCtxCounts.map((d: any) => d.ctxIdx))
    const xRange = [SIDE_PADDING].concat(
      calculatedCtxCounts.map(
        (d: any, i: number) =>
          SIDE_PADDING +
          ((d.cumsum + 2 * (i + 1)) * (dimensions.width - SIDE_PADDING)) /
            totalUnits,
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
          // Topological sort first
          let sortedNodes = topologicalSort(layerNodes, outgoingEdges)

          // Then sort for logits by token probability
          sortedNodes = sortedNodes.sort((a, b) => {
            if (a.featureType === 'logit' && b.featureType === 'logit') {
              return -(a.tokenProb || 0) + (b.tokenProb || 0)
            }
            return 0
          })

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
      d.pos = [x(d.ctxIdx) + xOffset, (y(d.layer + 1) || 0) + y.bandwidth() / 2]
    })

    return { x, y, positionedNodes }
  }, [data.nodes, data.edges, dimensions, calculatedCtxCounts, totalUnits])

  const nodeIndex = useMemo(
    () => createNodeIndex(positionedNodes),
    [positionedNodes],
  )

  const positionedEdges = useMemo(
    () => createPositionedEdges(data.edges, nodeIndex),
    [data.edges, nodeIndex],
  )

  const edgeIndex = useMemo(
    () => createEdgeIndex(positionedEdges),
    [positionedEdges],
  )

  const spatialIndex = useMemo(
    () => createSpatialIndex(positionedNodes),
    [positionedNodes],
  )

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const MAGNET_THRESHOLD = 30
      const rect = event.currentTarget.getBoundingClientRect()
      const mouseX = event.clientX - rect.left
      const mouseY = event.clientY - rect.top

      const nearestNode = findNearestNode(
        spatialIndex,
        mouseX,
        mouseY,
        MAGNET_THRESHOLD,
      )
      onNodeHover(nearestNode?.nodeId ?? null)
    },
    [spatialIndex, onNodeHover],
  )

  const handleClick = useCallback(
    (event: React.MouseEvent<SVGSVGElement>) => {
      const MAGNET_THRESHOLD = 30
      const rect = event.currentTarget.getBoundingClientRect()
      const mouseX = event.clientX - rect.left
      const mouseY = event.clientY - rect.top

      const nearestNode = findNearestNode(
        spatialIndex,
        mouseX,
        mouseY,
        MAGNET_THRESHOLD,
      )

      if (nearestNode) {
        onNodeClick(nearestNode.nodeId, event.metaKey || event.ctrlKey)
      }
    },
    [spatialIndex, onNodeClick],
  )

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        const newWidth = rect.width
        const newHeight = rect.height - 20

        setContainerDimensions((prev) => {
          if (
            Math.abs(prev.width - newWidth) < 1 &&
            Math.abs(prev.height - newHeight) < 1
          ) {
            return prev
          }
          return {
            width: newWidth,
            height: newHeight,
          }
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
  }, [setContainerDimensions])

  const tokenData = useMemo(() => {
    if (!data.metadata?.promptTokens || !positionedNodes.length || !x) return []

    const maxCtxIdx = d3.max(positionedNodes, (d) => d.ctxIdx) || 0

    return data.metadata.promptTokens
      .slice(0, maxCtxIdx + 1)
      .map((token: string, index: number) => {
        const contextNodes = getNodesByCtxIdx(nodeIndex, index)

        if (contextNodes.length === 0) {
          return {
            token,
            ctxIdx: index,
            x: x(index + 1) - (x(index + 1) - x(index)) / 2,
          }
        }

        let rightX = -Infinity
        for (const node of contextNodes) {
          if (node.pos[0] > rightX) rightX = node.pos[0]
        }

        return {
          token,
          ctxIdx: index,
          x: rightX,
        }
      })
  }, [data.metadata?.promptTokens, positionedNodes, nodeIndex, x])

  if (!positionedNodes.length || !x || !y) {
    return (
      <div ref={containerRef} className="relative w-full h-[400px]">
        <div className="flex items-center justify-center h-full text-slate-500">
          Loading circuit graph...
        </div>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full min-h-[420px] h-[420px] overflow-x-auto overflow-y-hidden"
    >
      <svg
        width={dimensions.width}
        height={dimensions.height}
        className="relative z-1"
        style={{ pointerEvents: 'auto' }}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
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

        <Links edgeIndex={edgeIndex} visState={visState} />

        <Nodes
          positionedNodes={positionedNodes}
          edgeIndex={edgeIndex}
          visState={{
            clickedId: visState.clickedId,
            hoveredId: visState.hoveredId,
            selectedIds: visState.selectedIds,
          }}
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
