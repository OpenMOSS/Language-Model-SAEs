export interface Node {
  id: string
  nodeId: string
  featureId: string
  feature_type: string
  ctx_idx: number
  layerIdx: number
  pos: [number, number]
  xOffset: number
  yOffset: number
  nodeColor: string
  logitPct?: number
  logitToken?: string
  featureIndex?: number
  localClerp?: string
  remoteClerp?: string
  sourceLinks?: Link[]
  targetLinks?: Link[]
}

export interface Link {
  source: string
  target: string
  pathStr: string
  color: string
  strokeWidth: number
  weight?: number
  pctInput?: number
}

export interface LinkGraphData {
  nodes: Node[]
  links: Link[]
  metadata: CircuitMetadata
}

export interface CircuitMetadata {
  prompt_tokens: string[]
  lorsa_analysis_name?: string
  clt_analysis_name?: string
}

export interface VisState {
  pinnedIds: string[]
  clickedId: string | null
  hoveredId: string | null
  isShowAllLinks: boolean
}

export interface CircuitJsonData {
  metadata: {
    slug: string
    scan: string
    prompt_tokens: string[]
    prompt: string
    lorsa_analysis_name?: string
    clt_analysis_name?: string
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
  }[]
  links?: {
    source: string
    target: string
    weight: number
  }[]
}
