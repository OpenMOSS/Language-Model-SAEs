export interface Node {
  id: string;
  nodeId: string;
  featureId: string;
  feature_type: string;
  ctx_idx: number;
  layerIdx: number;
  pos: [number, number];
  xOffset: number;
  yOffset: number;
  nodeColor: string;
  logitPct?: number;
  logitToken?: string;
  featureIndex?: number;
  localClerp?: string;
  remoteClerp?: string;
  sourceLinks?: Link[];
  targetLinks?: Link[];
}

export interface Link {
  source: string;
  target: string;
  sourceNode: Node;
  targetNode: Node;
  pathStr: string;
  color: string;
  strokeWidth: number;
}

export interface LinkGraphData {
  nodes: Node[];
  links: Link[];
  metadata: {
    prompt_tokens: string[];
  };
}

export interface VisState {
  pinnedIds: string[];
  clickedId: string | null;
  hoveredId: string | null;
  linkType: 'input' | 'output' | 'either' | 'both';
  isShowAllLinks: boolean;
  isHideLayer: boolean;
} 