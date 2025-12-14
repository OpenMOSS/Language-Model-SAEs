/**
 * Type definitions for circuit visualization.
 */

/**
 * Represents a node in the circuit graph.
 */
export interface Node {
  /** Unique identifier for the node. */
  id: string
  /** Node ID used for graph operations. */
  nodeId: string
  /** Feature ID associated with this node. */
  featureId: string
  /** Type of feature (e.g., 'logit', 'embedding', 'cross layer transcoder', 'lorsa'). */
  feature_type: string
  /** Context index position. */
  ctx_idx: number
  /** Layer index in the model. */
  layerIdx: number
  /** Position coordinates [x, y]. */
  pos: [number, number]
  /** X offset for positioning. */
  xOffset: number
  /** Y offset for positioning. */
  yOffset: number
  /** Color for rendering the node. */
  nodeColor: string
  /** Logit percentage (optional). */
  logitPct?: number
  /** Logit token (optional). */
  logitToken?: string
  /** Feature index (optional). */
  featureIndex?: number
  /** Local clerp description (optional). */
  localClerp?: string
  /** Remote clerp description (optional). */
  remoteClerp?: string
  /** Links where this node is the source (optional). */
  sourceLinks?: Link[]
  /** Links where this node is the target (optional). */
  targetLinks?: Link[]
}

/**
 * Represents a link/edge between nodes.
 */
export interface Link {
  /** Source node ID. */
  source: string
  /** Target node ID. */
  target: string
  /** SVG path string for rendering. */
  pathStr: string
  /** Link color. */
  color: string
  /** Stroke width for rendering. */
  strokeWidth: number
  /** Weight of the connection (optional). */
  weight?: number
  /** Percentage of input (optional). */
  pctInput?: number
}

/**
 * Data structure for the link graph visualization.
 */
export interface LinkGraphData {
  /** All nodes in the graph. */
  nodes: Node[]
  /** All links/edges in the graph. */
  links: Link[]
  /** Metadata about the circuit. */
  metadata: CircuitMetadata
}

/**
 * Metadata about the circuit.
 */
export interface CircuitMetadata {
  /** Prompt tokens for display. */
  prompt_tokens: string[]
  /** LORSA analysis name (optional). */
  lorsa_analysis_name?: string
  /** CLT analysis name (optional). */
  clt_analysis_name?: string
}

/**
 * Visual state for the graph.
 */
export interface VisState {
  /** IDs of pinned nodes. */
  pinnedIds: string[]
  /** Currently clicked node ID. */
  clickedId: string | null
  /** Currently hovered node ID. */
  hoveredId: string | null
  /** Whether to show all links. */
  isShowAllLinks: boolean
}

/**
 * Raw JSON data structure from circuit file.
 */
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
