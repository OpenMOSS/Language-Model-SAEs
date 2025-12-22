import * as d3 from 'd3'
import type { Edge, PositionedEdge, PositionedNode } from '@/types/circuit'

export interface NodeIndex {
  byId: Map<string, PositionedNode>
  byCtxIdx: Map<number, PositionedNode[]>
}

export interface EdgeIndex {
  bySource: Map<string, PositionedEdge[]>
  byTarget: Map<string, PositionedEdge[]>
  byNode: Map<string, PositionedEdge[]>
  connectedNodes: Map<string, Set<string>>
}

export function createNodeIndex(nodes: PositionedNode[]): NodeIndex {
  const byId = new Map<string, PositionedNode>()
  const byCtxIdx = new Map<number, PositionedNode[]>()

  for (const node of nodes) {
    byId.set(node.nodeId, node)

    const ctxNodes = byCtxIdx.get(node.ctxIdx)
    if (ctxNodes) {
      ctxNodes.push(node)
    } else {
      byCtxIdx.set(node.ctxIdx, [node])
    }
  }

  return { byId, byCtxIdx }
}

export function createEdgeIndex(edges: PositionedEdge[]): EdgeIndex {
  const bySource = new Map<string, PositionedEdge[]>()
  const byTarget = new Map<string, PositionedEdge[]>()
  const byNode = new Map<string, PositionedEdge[]>()
  const connectedNodes = new Map<string, Set<string>>()

  for (const edge of edges) {
    // Index by source
    const sourceEdges = bySource.get(edge.source)
    if (sourceEdges) {
      sourceEdges.push(edge)
    } else {
      bySource.set(edge.source, [edge])
    }

    // Index by target
    const targetEdges = byTarget.get(edge.target)
    if (targetEdges) {
      targetEdges.push(edge)
    } else {
      byTarget.set(edge.target, [edge])
    }

    // Index by both nodes (for quick lookup of all edges connected to a node)
    const sourceNodeEdges = byNode.get(edge.source)
    if (sourceNodeEdges) {
      sourceNodeEdges.push(edge)
    } else {
      byNode.set(edge.source, [edge])
    }

    const targetNodeEdges = byNode.get(edge.target)
    if (targetNodeEdges) {
      targetNodeEdges.push(edge)
    } else {
      byNode.set(edge.target, [edge])
    }

    // Track connected nodes for quick isConnected checks
    let sourceConnected = connectedNodes.get(edge.source)
    if (!sourceConnected) {
      sourceConnected = new Set()
      connectedNodes.set(edge.source, sourceConnected)
    }
    sourceConnected.add(edge.target)

    let targetConnected = connectedNodes.get(edge.target)
    if (!targetConnected) {
      targetConnected = new Set()
      connectedNodes.set(edge.target, targetConnected)
    }
    targetConnected.add(edge.source)
  }

  return { bySource, byTarget, byNode, connectedNodes }
}

export type SpatialIndex = d3.Quadtree<PositionedNode>

export function createSpatialIndex(nodes: PositionedNode[]): SpatialIndex {
  return d3
    .quadtree<PositionedNode>()
    .x((d) => d.pos[0])
    .y((d) => d.pos[1])
    .addAll(nodes)
}

export function findNearestNode(
  quadtree: SpatialIndex,
  x: number,
  y: number,
  maxDistance: number,
): PositionedNode | null {
  let nearest: PositionedNode | null = null
  let nearestDistance = maxDistance

  quadtree.visit((quad, x0, y0, x1, y1) => {
    if (!quad.length) {
      const node = quad.data
      if (node) {
        const dx = x - node.pos[0]
        const dy = y - node.pos[1]
        const distance = Math.sqrt(dx * dx + dy * dy)
        if (distance < nearestDistance) {
          nearestDistance = distance
          nearest = node
        }
      }
    }
    // Skip this quadrant if the closest possible point is farther than current nearest
    const closestX = Math.max(x0, Math.min(x, x1))
    const closestY = Math.max(y0, Math.min(y, y1))
    const dx = x - closestX
    const dy = y - closestY
    return Math.sqrt(dx * dx + dy * dy) > nearestDistance
  })

  return nearest
}

export function isNodeConnected(
  edgeIndex: EdgeIndex,
  clickedId: string,
  nodeId: string,
): boolean {
  const connected = edgeIndex.connectedNodes.get(clickedId)
  return connected?.has(nodeId) ?? false
}

export function getConnectedEdges(
  edgeIndex: EdgeIndex,
  nodeId: string,
): PositionedEdge[] {
  return edgeIndex.byNode.get(nodeId) ?? []
}

export function createPositionedEdges(
  edges: Edge[],
  nodeIndex: NodeIndex,
): PositionedEdge[] {
  const result: PositionedEdge[] = []

  for (const edge of edges) {
    const sourceNode = nodeIndex.byId.get(edge.source)
    const targetNode = nodeIndex.byId.get(edge.target)

    if (sourceNode && targetNode) {
      const [x1, y1] = sourceNode.pos
      const [x2, y2] = targetNode.pos
      result.push({
        ...edge,
        pathStr: `M${x1},${y1}L${x2},${y2}`,
      })
    }
  }

  return result
}
