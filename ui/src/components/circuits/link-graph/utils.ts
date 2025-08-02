import { LinkGraphData, Node, Link, VisState } from "./types";
// @ts-ignore
import d3 from "../static_js/d3";

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
  // Create a map of jsNodeId to node for easy lookup
  const nodeMap = new Map<string, any>();
  
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
        default:
          return "#95a5a6";
      }
    };

    // Don't set initial positions - let the component handle positioning
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
      logitToken: node.is_target_logit ? "target" : undefined,
      localClerp: node.clerp,
    };

    nodeMap.set(node.node_id, transformedNode);
    return transformedNode;
  });
  
  // Extract edges from the 'links' array in the JSON
  const edges = (jsonData as any).links || [];
  
  const links: Link[] = edges.map((edge: { source: string; target: string; weight: number }) => {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);
    
    if (!sourceNode || !targetNode) {
      console.warn(`Missing node for edge: ${edge.source} -> ${edge.target}`);
      return null;
    }

    // Calculate stroke width based on weight
    const strokeWidth = Math.max(0.5, Math.min(3, Math.abs(edge.weight) * 10));
    
    // Calculate color based on weight
    const color = edge.weight > 0 ? "#4CAF50" : "#F44336";

    return {
      source: edge.source,
      target: edge.target,
      sourceNode,
      targetNode,
      pathStr: "", // Will be set by the component after positioning
      color,
      strokeWidth,
    };
  }).filter(Boolean) as Link[];

  return {
    nodes,
    links,
    metadata: {
      prompt_tokens: jsonData.metadata.prompt_tokens,
    },
  };
}

export async function loadDefaultCircuitData(): Promise<LinkGraphData> {
  try {
    const response = await fetch('/circuits/example_data/capital-state-dallas.json');
    if (!response.ok) {
      throw new Error(`Failed to load circuit data: ${response.statusText}`);
    }
    
    const jsonData: CircuitJsonData = await response.json();
    return transformCircuitData(jsonData);
  } catch (error) {
    console.error('Error loading default circuit data:', error);
    throw error;
  }
}