import { LinkGraphData, Node, Link, VisState } from "./types";
// @ts-ignore
import d3 from "../static_js/d3";

// Cantor pairing function utilities
// The cantor pairing function maps two integers (x, y) to a single integer z
// This is used to encode layer and feature IDs into a single node ID
export const cantorPair = (x: number, y: number): number => {
  return ((x + y) * (x + y + 1)) / 2 + y;
};

// Reverse the cantor pairing function to extract the original (x, y) from z
export const cantorUnpair = (z: number): [number, number] => {
  const w = Math.floor((Math.sqrt(8 * z + 1) - 1) / 2);
  const t = (w * w + w) / 2;
  const y = z - t;
  const x = w - y;
  return [x, y];
};

// Extract layer and feature ID from node ID using cantor unpairing
// Returns null if the node ID is not a valid cantor-paired number
export const extractLayerAndFeature = (nodeId: string): { layer: number; featureId: number; isLorsa: boolean } | null => {
  try {
    const parts = nodeId.split("_");
    const layer = Math.floor(parseInt(parts[0]) / 2);
    const isLorsa = parseInt(parts[0]) % 2 === 0;
    const featureId = parseInt(parts[1]);
    if (isNaN(layer) || isNaN(featureId)) {
      return null;
    }
    return { layer, featureId, isLorsa };
  } catch (error) {
    console.error('Error extracting layer and feature from node ID:', error);
    return null;
  }
};

export const featureTypeToText = (type: string): string => {
  switch (type) {
    case "embedding": return "E";
    case "logit": return "L";
    default: return type.charAt(0).toUpperCase();
  }
};

export interface CircuitJsonData {
  metadata: {
    slug: string;
    scan: string;
    prompt_tokens: string[];
    prompt: string;
    lorsa_analysis_name?: string;
    clt_analysis_name?: string;
  };
  qParams: {
    linkType: string;
    pinnedIds: string[];
    clickedId: string;
    supernodes: string[][];
    sg_pos: string;
  };
  nodes: Array<{
    node_id: string;
    feature: number;
    layer: number;
    ctx_idx: number;
    feature_type: string;
    token_prob: number;
    is_target_logit: boolean;
    run_idx: number;
    reverse_ctx_idx: number;
    jsNodeId: string;
    clerp: string;
  }>;
  // Edges are at the root level, not in an edges property
  [key: string]: any; // Allow for edge objects at root level
}

export function transformCircuitData(jsonData: CircuitJsonData): LinkGraphData {
  // Transform nodes
  const nodes: Node[] = jsonData.nodes.map((node) => {
    // Generate a color based on feature type
    const getNodeColor = (featureType: string): string => {
      switch (featureType) {
        case "logit":
          return "#ff6b6b";
        case "embedding":
          return "#69b3a2";
        case "cross layer transcoder":
          return "#4ecdc4";
        case "lorsa":
          return "#7a4cff";
        default:
          return "#95a5a6";
      }
    };

    // Handle null feature values
    const featureId = node.feature !== null && node.feature !== undefined
      ? node.feature.toString()
      : node.node_id; // Use node_id as fallback

    // Don't set initial positions - let the component handle positioning
    const transformedNode: Node = {
      id: node.node_id,
      nodeId: node.node_id,
      featureId: featureId,
      feature_type: node.feature_type,
      ctx_idx: node.ctx_idx,
      layerIdx: node.layer + 1,
      pos: [0, 0], // Will be set by the component
      xOffset: 0,
      yOffset: 0,
      nodeColor: getNodeColor(node.feature_type),
      logitPct: node.token_prob,
      logitToken: node.is_target_logit ? "target" : undefined,
      localClerp: node.clerp,
    };

    return transformedNode;
  });
  
  // Extract edges from the 'links' array in the JSON
  const edges = (jsonData as any).links || [];
  
  const links: Link[] = edges.map((edge: { source: string; target: string; weight: number }) => {
    // Calculate stroke width based on weight
    const strokeWidth = Math.max(0.5, Math.min(3, Math.abs(edge.weight) * 10));
    
    // Calculate color based on weight
    const color = edge.weight > 0 ? "#4CAF50" : "#F44336";

    return {
      source: edge.source,
      target: edge.target,
      pathStr: "", // Will be set by the component after positioning
      color,
      strokeWidth,
      weight: edge.weight,
      pctInput: Math.abs(edge.weight) * 100, // Convert weight to percentage
    };
  });

  // Populate sourceLinks and targetLinks for each node
  nodes.forEach(node => {
    node.sourceLinks = links.filter(link => link.source === node.nodeId);
    node.targetLinks = links.filter(link => link.target === node.nodeId);
  });

  return {
    nodes,
    links,
    metadata: {
      prompt_tokens: jsonData.metadata.prompt_tokens,
      lorsa_analysis_name: jsonData.metadata.lorsa_analysis_name,
      clt_analysis_name: jsonData.metadata.clt_analysis_name,
    },
  };
}
