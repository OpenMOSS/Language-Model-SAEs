import type {
  CircuitData,
  CircuitJsonData,
  CircuitMetadata,
  Node,
  PositionedNode,
} from '@/types/circuit'

export function extractLayerAndFeature(
  nodeId: string,
): { layer: number; featureId: number; isLorsa: boolean } | null {
  try {
    const parts = nodeId.split('_')
    const layer = Math.floor(parseInt(parts[0]) / 2)
    const isLorsa = parseInt(parts[0]) % 2 === 0
    const featureId = parseInt(parts[1])
    if (isNaN(layer) || isNaN(featureId)) {
      return null
    }
    return { layer, featureId, isLorsa }
  } catch {
    console.error('Error extracting layer and feature from node ID')
    return null
  }
}

export function getNodeColor(featureType: string): string {
  switch (featureType) {
    case 'logit':
      return '#ff6b6b'
    case 'embedding':
      return '#69b3a2'
    case 'cross layer transcoder':
      return '#4ecdc4'
    case 'lorsa':
      return '#7a4cff'
    default:
      return '#95a5a6'
  }
}

export function getEdgeColor(weight: number): string {
  return weight > 0 ? '#4CAF50' : '#F44336'
}

export function getEdgeStrokeWidth(weight: number): number {
  return Math.max(0.5, Math.min(3, Math.abs(weight) * 10))
}

export function transformCircuitData(jsonData: CircuitJsonData): CircuitData {
  const nodes: Node[] = jsonData.nodes.map((node) => ({
    nodeId: node.node_id,
    feature: node.feature,
    layer: node.layer,
    ctxIdx: node.ctx_idx,
    featureType: node.feature_type,
    tokenProb: node.token_prob,
    isTargetLogit: node.is_target_logit,
    clerp: node.clerp,
    saeName: node.sae_name,
  }))

  const edges = (jsonData.links || []).map((edge) => ({
    source: edge.source,
    target: edge.target,
    weight: edge.weight,
  }))

  return {
    nodes,
    edges,
    metadata: {
      promptTokens: jsonData.metadata.prompt_tokens,
    },
  }
}

export function formatFeatureId(
  node: Node | PositionedNode,
  verbose: boolean = true,
): string {
  const layerIdx = node.layer + 1
  if (node.featureType === 'cross layer transcoder') {
    const mlpLayer = Math.floor(layerIdx / 2) - 1
    const featureId = node.nodeId.split('_')[1]
    return verbose ? `M${mlpLayer}#${featureId}@${node.ctxIdx}` : `M${mlpLayer}`
  } else if (node.featureType === 'lorsa') {
    const attnLayer = Math.floor(layerIdx / 2)
    const featureId = node.nodeId.split('_')[1]
    return verbose
      ? `A${attnLayer}#${featureId}@${node.ctxIdx}`
      : `A${attnLayer}`
  } else if (node.featureType === 'embedding') {
    return `Emb@${node.ctxIdx}`
  } else if (node.featureType === 'mlp reconstruction error') {
    return `M${Math.floor(layerIdx / 2) - 1}Error@${node.ctxIdx}`
  } else if (node.featureType === 'lorsa error') {
    return `A${Math.floor(layerIdx / 2)}Error@${node.ctxIdx}`
  }
  return ' '
}

export function findEdgeWeight(
  edges: { source: string; target: string; weight: number }[],
  source: string,
  target: string,
): number | undefined {
  const edge = edges.find((e) => e.source === source && e.target === target)
  return edge?.weight
}
