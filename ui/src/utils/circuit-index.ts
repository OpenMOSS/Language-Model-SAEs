import RBush from 'rbush'
import knn from 'rbush-knn'
import { MultiMap } from 'mnemonist'
import type {
  Edge,
  Node,
  PositionedEdge,
  PositionedNode,
} from '@/types/circuit'

interface SpatialItem {
  minX: number
  minY: number
  maxX: number
  maxY: number
  node: PositionedNode
}

export interface NodeIndex {
  byId: Map<string, PositionedNode>
  byCtxIdx: MultiMap<number, PositionedNode>
}

export interface RawNodeIndex {
  byId: Map<string, Node>
}

export interface EdgeIndex {
  bySource: MultiMap<string, PositionedEdge>
  byTarget: MultiMap<string, PositionedEdge>
  byNode: MultiMap<string, PositionedEdge>
  connectedNodes: MultiMap<string, string>
  sortedByWeight: PositionedEdge[]
}

export interface RawEdgeIndex {
  bySource: MultiMap<string, Edge>
  byTarget: MultiMap<string, Edge>
}

export function createNodeIndex(nodes: PositionedNode[]): NodeIndex {
  const byId = new Map<string, PositionedNode>()
  const byCtxIdx = new MultiMap<number, PositionedNode>()

  for (const node of nodes) {
    byId.set(node.nodeId, node)
    byCtxIdx.set(node.ctxIdx, node)
  }

  return { byId, byCtxIdx }
}

export function createRawNodeIndex(nodes: Node[]): RawNodeIndex {
  const byId = new Map<string, Node>()
  for (const node of nodes) {
    byId.set(node.nodeId, node)
  }
  return { byId }
}

export function createRawEdgeIndex(edges: Edge[]): RawEdgeIndex {
  const bySource = new MultiMap<string, Edge>()
  const byTarget = new MultiMap<string, Edge>()

  for (const edge of edges) {
    bySource.set(edge.source, edge)
    byTarget.set(edge.target, edge)
  }

  return { bySource, byTarget }
}

export function createEdgeIndex(edges: PositionedEdge[]): EdgeIndex {
  const bySource = new MultiMap<string, PositionedEdge>()
  const byTarget = new MultiMap<string, PositionedEdge>()
  const byNode = new MultiMap<string, PositionedEdge>()
  const connectedNodes = new MultiMap<string, string>()

  for (const edge of edges) {
    bySource.set(edge.source, edge)
    byTarget.set(edge.target, edge)
    byNode.set(edge.source, edge)
    byNode.set(edge.target, edge)
    connectedNodes.set(edge.source, edge.target)
    connectedNodes.set(edge.target, edge.source)
  }

  const sortedByWeight = [...edges].sort(
    (a, b) => Math.abs(b.weight) - Math.abs(a.weight),
  )

  return { bySource, byTarget, byNode, connectedNodes, sortedByWeight }
}

export type SpatialIndex = RBush<SpatialItem>

export function createSpatialIndex(nodes: PositionedNode[]): SpatialIndex {
  const tree = new RBush<SpatialItem>()

  const items: SpatialItem[] = nodes.map((node) => ({
    minX: node.pos[0],
    minY: node.pos[1],
    maxX: node.pos[0],
    maxY: node.pos[1],
    node,
  }))

  tree.load(items)
  return tree
}

export function findNearestNode(
  tree: SpatialIndex,
  x: number,
  y: number,
  maxDistance: number,
): PositionedNode | null {
  const results = knn(tree, x, y, 1, undefined, maxDistance)
  return results.length > 0 ? results[0].node : null
}

export function isNodeConnected(
  edgeIndex: EdgeIndex,
  clickedId: string,
  nodeId: string,
): boolean {
  const connected = edgeIndex.connectedNodes.get(clickedId)
  if (!connected) return false
  if (Array.isArray(connected)) {
    return connected.includes(nodeId)
  }
  return connected === nodeId
}

export function getConnectedEdges(
  edgeIndex: EdgeIndex,
  nodeId: string,
): PositionedEdge[] {
  const edges = edgeIndex.byNode.get(nodeId)
  if (!edges) return []
  return Array.isArray(edges) ? edges : [edges]
}

export function getAllEdges(edgeIndex: EdgeIndex): PositionedEdge[] {
  return Array.from(edgeIndex.bySource.values())
}

export function getTopEdgesByWeight(
  edgeIndex: EdgeIndex,
  limit: number,
): PositionedEdge[] {
  return edgeIndex.sortedByWeight.slice(0, limit)
}

export function getNodesByCtxIdx(
  nodeIndex: NodeIndex,
  ctxIdx: number,
): PositionedNode[] {
  const nodes = nodeIndex.byCtxIdx.get(ctxIdx)
  if (!nodes) return []
  return Array.isArray(nodes) ? nodes : [nodes]
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

export function getEdgesBySource(
  edgeIndex: RawEdgeIndex,
  source: string,
): Edge[] {
  const edges = edgeIndex.bySource.get(source)
  if (!edges) return []
  return Array.isArray(edges) ? edges : [edges]
}

export function getEdgesByTarget(
  edgeIndex: RawEdgeIndex,
  target: string,
): Edge[] {
  const edges = edgeIndex.byTarget.get(target)
  if (!edges) return []
  return Array.isArray(edges) ? edges : [edges]
}
