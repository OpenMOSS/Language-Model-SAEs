// Base types from JSON data (after camelCase transformation)
export interface Node {
  nodeId: string
  feature: number
  layer: number
  ctxIdx: number
  featureType: string
  tokenProb: number
  isTargetLogit: boolean
  clerp: string
  saeName?: string
}

export interface Edge {
  source: string
  target: string
  weight: number
}

// Positioned types for visualization
export interface PositionedNode extends Node {
  pos: [number, number]
}

export interface PositionedEdge extends Edge {
  pathStr: string
}

// Circuit data container
export interface CircuitData {
  nodes: Node[]
  edges: Edge[]
  metadata: CircuitMetadata
}

export interface CircuitMetadata {
  promptTokens: string[]
}

export interface VisState {
  clickedId: string | null
  hoveredId: string | null
}

// Raw JSON data structure
export interface CircuitJsonData {
  metadata: {
    slug: string
    scan: string
    prompt_tokens: string[]
    prompt: string
  }
  qParams: {
    linkType: string
    pinnedIds: string[]
    clickedId: string
    supernodes: string[][]
    sg_pos: string
  }
  nodes: {
    node_id: string
    feature: number
    layer: number
    ctx_idx: number
    feature_type: string
    token_prob: number
    is_target_logit: boolean
    run_idx: number
    reverse_ctx_idx: number
    jsNodeId: string
    clerp: string
    sae_name?: string
  }[]
  links?: {
    source: string
    target: string
    weight: number
  }[]
}
