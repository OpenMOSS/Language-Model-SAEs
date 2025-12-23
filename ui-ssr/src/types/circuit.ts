import type { FeatureCompact } from './feature'

export type FeatureNode = {
  featureType: 'lorsa' | 'cross layer transcoder'
  nodeId: string
  layer: number
  ctxIdx: number
  isTargetLogit: boolean
  saeName: string
  feature: FeatureCompact
}

export type TokenNode = {
  featureType: 'embedding'
  nodeId: string
  layer: number
  ctxIdx: number
  token: string
}

export type ErrorNode = {
  featureType: 'lorsa error' | 'mlp reconstruction error'
  nodeId: string
  layer: number
  ctxIdx: number
}

export type LogitNode = {
  featureType: 'logit'
  nodeId: string
  layer: number
  ctxIdx: number
  tokenProb: number
  token: string
}

export type Node = FeatureNode | TokenNode | ErrorNode | LogitNode

export type Edge = {
  source: string
  target: string
  weight: number
}

export type PositionedNode = Node & {
  pos: [number, number]
}

export type PositionedEdge = Edge & {
  pathStr: string
}

export type CircuitData = {
  nodes: Node[]
  edges: Edge[]
  metadata: CircuitMetadata
}

export type CircuitMetadata = {
  promptTokens: string[]
}

export type VisState = {
  clickedId: string | null
  hoveredId: string | null
}
