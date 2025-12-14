/**
 * Utility functions for circuit visualization.
 */

import type {
  CircuitJsonData,
  CircuitMetadata,
  Link,
  LinkGraphData,
  Node,
} from '@/types/circuit'

/**
 * Extract layer and feature ID from node ID.
 *
 * @param nodeId - The node ID string.
 * @returns Object with layer, featureId, and isLorsa, or null if invalid.
 */
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

/**
 * Convert feature type to display text.
 *
 * @param type - The feature type string.
 * @returns Display text for the feature type.
 */
export function featureTypeToText(type: string): string {
  switch (type) {
    case 'embedding':
      return 'E'
    case 'logit':
      return 'L'
    default:
      return type.charAt(0).toUpperCase()
  }
}

/**
 * Get dictionary name from circuit metadata based on layer and type.
 *
 * @param metadata - The circuit metadata.
 * @param layer - The layer index.
 * @param isLorsa - Whether this is a LORSA feature.
 * @returns The dictionary name or null.
 */
export function getDictionaryName(
  metadata: CircuitMetadata,
  layer: number,
  isLorsa: boolean,
): string | null {
  const analysisName = isLorsa
    ? metadata.lorsa_analysis_name
    : metadata.clt_analysis_name
  if (!analysisName) return null

  // The analysis name format is typically "dict_name/layer_X"
  // We need to construct the full dictionary path
  const parts = analysisName.split('/')
  if (parts.length < 1) return null

  // Return dictionary name with layer
  return `${parts[0]}/${isLorsa ? 'attn' : 'mlp'}_${layer}`
}

/**
 * Get color based on feature type.
 *
 * @param featureType - The type of feature.
 * @returns Hex color string.
 */
function getNodeColor(featureType: string): string {
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

/**
 * Transform raw circuit JSON data to LinkGraphData format.
 *
 * @param jsonData - Raw circuit JSON data.
 * @returns Transformed LinkGraphData.
 */
export function transformCircuitData(jsonData: CircuitJsonData): LinkGraphData {
  // Transform nodes
  const nodes: Node[] = jsonData.nodes.map((node) => {
    const transformedNode: Node = {
      id: node.node_id,
      nodeId: node.node_id,
      featureId: node.feature.toString(),
      feature_type: node.feature_type,
      ctx_idx: node.ctx_idx,
      layerIdx: node.layer + 1,
      pos: [0, 0], // Will be set by the component
      xOffset: 0,
      yOffset: 0,
      nodeColor: getNodeColor(node.feature_type),
      logitPct: node.token_prob,
      logitToken: node.is_target_logit ? 'target' : undefined,
      localClerp: node.clerp,
    }

    return transformedNode
  })

  // Extract edges from the 'links' array in the JSON
  const edges = jsonData.links || []

  const links: Link[] = edges.map(
    (edge: { source: string; target: string; weight: number }) => {
      // Calculate stroke width based on weight
      const strokeWidth = Math.max(0.5, Math.min(3, Math.abs(edge.weight) * 10))

      // Calculate color based on weight
      const color = edge.weight > 0 ? '#4CAF50' : '#F44336'

      return {
        source: edge.source,
        target: edge.target,
        pathStr: '', // Will be set by the component after positioning
        color,
        strokeWidth,
        weight: edge.weight,
        pctInput: Math.abs(edge.weight) * 100, // Convert weight to percentage
      }
    },
  )

  // Populate sourceLinks and targetLinks for each node
  nodes.forEach((node) => {
    node.sourceLinks = links.filter((link) => link.source === node.nodeId)
    node.targetLinks = links.filter((link) => link.target === node.nodeId)
  })

  return {
    nodes,
    links,
    metadata: {
      prompt_tokens: jsonData.metadata.prompt_tokens,
      lorsa_analysis_name: jsonData.metadata.lorsa_analysis_name,
      clt_analysis_name: jsonData.metadata.clt_analysis_name,
    },
  }
}

/**
 * Format a node's feature ID for display.
 *
 * @param node - The node to format.
 * @param verbose - Whether to include full details.
 * @returns Formatted feature ID string.
 */
export function formatFeatureId(node: Node, verbose: boolean = true): string {
  if (node.feature_type === 'cross layer transcoder') {
    const layerIdx = Math.floor(node.layerIdx / 2) - 1
    const featureId = node.id.split('_')[1]
    return verbose
      ? `M${layerIdx}#${featureId}@${node.ctx_idx}`
      : `M${layerIdx}`
  } else if (node.feature_type === 'lorsa') {
    const layerIdx = Math.floor(node.layerIdx / 2)
    const featureId = node.id.split('_')[1]
    return verbose
      ? `A${layerIdx}#${featureId}@${node.ctx_idx}`
      : `A${layerIdx}`
  } else if (node.feature_type === 'embedding') {
    return `Emb@${node.ctx_idx}`
  } else if (node.feature_type === 'mlp reconstruction error') {
    return `M${Math.floor(node.layerIdx / 2) - 1}Error@${node.ctx_idx}`
  } else if (node.feature_type === 'lorsa error') {
    return `A${Math.floor(node.layerIdx / 2)}Error@${node.ctx_idx}`
  }
  return ' '
}
