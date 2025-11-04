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
  pathStr: string;
  color: string;
  strokeWidth: number;
  weight?: number;
  pctInput?: number;
}

export interface LinkGraphData {
  nodes: Node[];
  links: Link[];
  metadata: {
    prompt_tokens: string[];
    lorsa_analysis_name?: string;
    clt_analysis_name?: string;
  };
}

export interface VisState {
  pinnedIds: string[];
  clickedId: string | null;
  hoveredId: string | null;
  isShowAllLinks: boolean;
} 