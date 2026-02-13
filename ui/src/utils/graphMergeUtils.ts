/**
 * Graph merging utility functions
 * Used for merging multiple circuit graphs with color mixing
 */

import { transformCircuitData, CircuitJsonData } from "@/components/circuits/link-graph/utils";
import { mixHexColorsVivid } from "./colorUtils";

export const UNIQUE_GRAPH_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#C0392B"]; // Blue, Orange, Green, Red
export const POSITION_MAPPING_HIGHLIGHT_COLOR = "#8E44AD"; // Purple: position mapping highlight

/**
 * Get subset color for partially shared nodes
 * Uses stable coloring for subset combinations (e.g., 0-1, 0-2, 1-2 for 3 files)
 */
export const getSubsetColor = (
  sourceIndices: number[],
  colorCache: Map<string, string>
): string | null => {
  const sorted = [...sourceIndices].sort((a, b) => a - b);
  const key = sorted.join("-");
  const cached = colorCache.get(key);
  if (cached) return cached;
  
  const mix = mixHexColorsVivid(
    sorted.map((i) => UNIQUE_GRAPH_COLORS[i % UNIQUE_GRAPH_COLORS.length])
  );
  if (!mix) return null;
  
  colorCache.set(key, mix);
  return mix;
};

/**
 * Merge multiple circuit graphs into a single LinkGraphData
 * Nodes are merged by node_id, links are merged by (source, target)
 */
export const mergeCircuitGraphs = (
  jsons: CircuitJsonData[],
  fileNames?: string[]
): any => {
  // Transform each JSON to LinkGraphData
  const graphs = jsons.map(j => transformCircuitData(j));
  const totalSources = graphs.length;

  // Subset color cache for stable coloring
  const subsetColorCache = new Map<string, string>();

  // Merge metadata
  const mergedMetadata: any = {
    ...(graphs[0]?.metadata || {}),
    prompt_tokens: graphs.map((g, i) => 
      `[#${i + 1}] ` + (g?.metadata?.prompt_tokens?.join(' ') || '')
    ).filter(Boolean),
    sourceFileNames: fileNames && fileNames.length ? fileNames : undefined,
  };

  // Merge nodes
  type NodeAccum = {
    base: any; // Base node from any source (preserves feature_type, etc.)
    presentIn: number[]; // Indices of graphs where this node appears
  };

  const nodeMap = new Map<string, NodeAccum>();

  graphs.forEach((g, gi) => {
    g.nodes.forEach((n: any) => {
      const key = n.nodeId;
      if (!nodeMap.has(key)) {
        nodeMap.set(key, { base: { ...n }, presentIn: [gi] });
      } else {
        const acc = nodeMap.get(key)!;
        // Merge optional fields (use non-null value)
        acc.base.localClerp = acc.base.localClerp ?? n.localClerp;
        acc.base.remoteClerp = acc.base.remoteClerp ?? n.remoteClerp;
        // Accumulate sources
        if (!acc.presentIn.includes(gi)) acc.presentIn.push(gi);
      }
    });
  });

  // Set node colors (multi-file comparison):
  // - If presentIn.length === totalSources (all graphs share): use original feature_type color
  // - If presentIn.length === 1 (single graph only): use UNIQUE_GRAPH_COLORS[sourceIndex]
  // - If 1 < presentIn.length < totalSources (partially shared): use mixed color of those sources
  const mergedNodes: any[] = [];
  nodeMap.forEach(({ base, presentIn }) => {
    const isAllShared = presentIn.length === totalSources;
    const isPartiallyShared = presentIn.length > 1 && presentIn.length < totalSources;
    const isError = typeof base.feature_type === 'string' && 
                    base.feature_type.toLowerCase().includes('error');
    
    let nodeColor: string;
    if (isError) {
      nodeColor = '#95a5a6';
    } else if (isAllShared) {
      nodeColor = base.nodeColor;
    } else if (isPartiallyShared) {
      // Use stable subset color for partially shared nodes
      nodeColor =
        getSubsetColor(presentIn, subsetColorCache) ??
        UNIQUE_GRAPH_COLORS[presentIn[0] % UNIQUE_GRAPH_COLORS.length];
    } else {
      nodeColor = UNIQUE_GRAPH_COLORS[presentIn[0] % UNIQUE_GRAPH_COLORS.length];
    }
    
    const sourceIndices = presentIn.slice();
    const sourceFiles = (fileNames && fileNames.length)
      ? sourceIndices.map(i => fileNames[i]).filter(Boolean)
      : undefined;
    
    mergedNodes.push({
      ...base,
      nodeColor,
      sourceIndices,
      sourceIndex: sourceIndices.length === 1 ? sourceIndices[0] : undefined,
      sourceFiles,
    });
  });

  // Merge links: keyed by (source, target)
  // - If multiple graphs share: keep transform color (red/green based on weight sign) 
  //   and strokeWidth (max), sum/average weights
  // - If single graph: keep and color with corresponding UNIQUE_GRAPH_COLORS[gi]
  type LinkAccum = {
    sources: number[]; // Which graphs this link appears in
    weightSum: number;
    maxStroke: number;
    color: string; // Use first weight color (positive/negative)
    pctInputSum: number;
    weightsBySource: Record<number, number>;
    pctBySource: Record<number, number>;
  };

  const linkKey = (s: string, t: string) => `${s}__${t}`;
  const linkMap = new Map<string, LinkAccum>();

  graphs.forEach((g, gi) => {
    (g.links || []).forEach((e: any) => {
      const k = linkKey(e.source, e.target);
      if (!linkMap.has(k)) {
        linkMap.set(k, {
          sources: [gi],
          weightSum: e.weight ?? 0,
          maxStroke: e.strokeWidth ?? 1,
          color: e.color,
          pctInputSum: e.pctInput ?? 0,
          weightsBySource: { [gi]: e.weight ?? 0 },
          pctBySource: { [gi]: e.pctInput ?? (Math.abs(e.weight ?? 0) * 100) },
        });
      } else {
        const acc = linkMap.get(k)!;
        if (!acc.sources.includes(gi)) acc.sources.push(gi);
        acc.weightSum += (e.weight ?? 0);
        acc.maxStroke = Math.max(acc.maxStroke, e.strokeWidth ?? 1);
        // Color uses first positive/negative, don't override
        acc.pctInputSum += (e.pctInput ?? 0);
        acc.weightsBySource[gi] = (acc.weightsBySource[gi] || 0) + (e.weight ?? 0);
        acc.pctBySource[gi] = (acc.pctBySource[gi] || 0) + 
          (e.pctInput ?? (Math.abs(e.weight ?? 0) * 100));
      }
    });
  });

  const mergedLinks: any[] = [];
  linkMap.forEach((acc, k) => {
    const [source, target] = k.split("__");
    const isShared = acc.sources.length > 1;
    const avgWeight = acc.weightSum / acc.sources.length;
    const avgPct = acc.pctInputSum / acc.sources.length;
    // Use transform's positive/negative coloring: positive=green, negative=red
    const color = avgWeight > 0 ? "#4CAF50" : "#F44336";
    mergedLinks.push({
      source,
      target,
      pathStr: "",
      color: isShared ? color : color,
      strokeWidth: acc.maxStroke,
      weight: avgWeight,
      pctInput: avgPct,
      sources: acc.sources,
      weightsBySource: acc.weightsBySource,
      pctBySource: acc.pctBySource,
    });
  });

  // Rebuild sourceLinks/targetLinks for nodes
  const nodeById: Record<string, any> = {};
  mergedNodes.forEach(n => { 
    nodeById[n.nodeId] = { ...n, sourceLinks: [], targetLinks: [] }; 
  });
  mergedLinks.forEach(l => {
    if (nodeById[l.source]) nodeById[l.source].sourceLinks.push(l);
    if (nodeById[l.target]) nodeById[l.target].targetLinks.push(l);
  });

  const finalNodes = Object.values(nodeById);

  return {
    nodes: finalNodes,
    links: mergedLinks,
    metadata: mergedMetadata,
  };
};
