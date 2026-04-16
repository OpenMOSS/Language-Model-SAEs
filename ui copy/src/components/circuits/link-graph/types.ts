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
  // Node origin: when multiple files are merged, record which source file indices this node comes from
  sourceIndex?: number; // Single source
  sourceIndices?: number[]; // Multiple sources
}

export interface Link {
  source: string;
  target: string;
  pathStr: string;
  color: string;
  strokeWidth: number;
  weight?: number;
  pctInput?: number;
  // Multi-file: per-source weights and percentages
  sources?: number[]; // Indices of source files in which this link appears
  weightsBySource?: Record<number, number>;
  pctBySource?: Record<number, number>;
}

export interface LinkGraphData {
  nodes: Node[];
  links: Link[];
  metadata: {
    prompt_tokens: string[];
    lorsa_analysis_name?: string;
    clt_analysis_name?: string;
    tc_analysis_name?: string;
    // When uploading multiple files, record file names; indices correspond to Node.sourceIndex
    sourceFileNames?: string[];
  };
}

export interface VisState {
  pinnedIds: string[];
  clickedId: string | null;
  hoveredId: string | null;
  isShowAllLinks: boolean;
} 