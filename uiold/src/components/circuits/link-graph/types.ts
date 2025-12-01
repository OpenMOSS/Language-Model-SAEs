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
  // 节点来源：当多文件合并时，记录该节点来自哪些源文件（索引）
  sourceIndex?: number; // 单一来源
  sourceIndices?: number[]; // 多个来源
}

export interface Link {
  source: string;
  target: string;
  pathStr: string;
  color: string;
  strokeWidth: number;
  weight?: number;
  pctInput?: number;
  // 多文件：每个来源文件的权重与占比
  sources?: number[]; // 出现于哪些源文件索引
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
    // 多文件上传时，记录各文件名，index 与 Node.sourceIndex 对应
    sourceFileNames?: string[];
  };
}

export interface VisState {
  pinnedIds: string[];
  clickedId: string | null;
  hoveredId: string | null;
  isShowAllLinks: boolean;
} 