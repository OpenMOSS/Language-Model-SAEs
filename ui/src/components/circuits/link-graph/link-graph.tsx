import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { LinkGraphData, VisState } from "./types";
// @ts-ignore
import d3 from "../static_js/d3";
import "./link-graph.css";
import {
  GridLines,
  RowBackgrounds,
  YAxis,
  Links,
  Nodes,
  Tooltips,
  TokenLabels
} from "./atomic-components";

// Performance optimization: Replaced expensive closest node logic with exact hover detection.
// Hover tooltips and indicators are preserved but only trigger when mouse is directly over nodes.
// This provides the same UX with much better performance on large graphs.

interface LinkGraphProps {
  data: LinkGraphData;
  visState: VisState;
  onNodeClick: (nodeId: string, metaKey: boolean) => void;
  onNodeHover: (nodeId: string | null) => void;
}

const LinkGraphComponent: React.FC<LinkGraphProps> = ({
  data,
  visState,
  onNodeClick,
  onNodeHover,
}) => {
  // onNodeHover is kept for interface compatibility but not used for performance
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const BOTTOM_PADDING = 50; // Space for token labels
  const SIDE_PADDING = 20; // Space for left and right margins
  
  // Memoize expensive calculations
  const { calculatedCtxCounts, x, y, positionedNodes, positionedLinks } = useMemo(() => {
    if (!data.nodes.length) {
      return { calculatedCtxCounts: [], x: null, y: null, positionedNodes: [], positionedLinks: [] };
    }

    const { nodes } = data;
    const earliestCtxWithNodes = d3.min(nodes, (d: any) => d.ctx_idx) || 0;
    
    let cumsum = 0;
    const calculatedCtxCounts = d3.range(d3.max(nodes, (d: any) => d.ctx_idx) + 1).map((ctx_idx: number) => {
      if (ctx_idx >= earliestCtxWithNodes) {
        const group = nodes.filter((d: any) => d.ctx_idx === ctx_idx);
        const layerGroups = d3.group(group, (d: any) => d.layerIdx);
        const maxNodesPerLayer = d3.max(Array.from(layerGroups.values()), (layerNodes: any) => layerNodes.length) || 1;
        const maxCount = Math.max(1, maxNodesPerLayer);
        cumsum += maxCount;
        return { ctx_idx, maxCount, cumsum, layerGroups };
      }
      return { ctx_idx, maxCount: 0, cumsum, layerGroups: new Map() };
    });

    const xDomain = [-1].concat(calculatedCtxCounts.map((d: any) => d.ctx_idx));
    const xRange = [SIDE_PADDING].concat(calculatedCtxCounts.map((d: any) => SIDE_PADDING + d.cumsum * (dimensions.width - 2 * SIDE_PADDING) / cumsum));
    const x = d3.scaleLinear().domain(xDomain.map((d: any) => d + 1)).range(xRange);

    const yNumTicks = d3.max(nodes, (d: any) => d.layerIdx) + 1;
    const y = d3.scaleBand(d3.range(yNumTicks), [dimensions.height - BOTTOM_PADDING, 0]);

    // Position nodes
    calculatedCtxCounts.forEach((d: any) => {
      d.width = x(d.ctx_idx + 1) - x(d.ctx_idx);
    });

    const padR = Math.min(8, d3.min(calculatedCtxCounts.slice(1), (d: any) => d.width / 2)) + 0;

    // Create a copy of nodes to avoid mutating the original data
    const positionedNodes = nodes.map((node: any) => ({ ...node }));

    // Position nodes within each context and layer
    calculatedCtxCounts.forEach((ctxData: any) => {
      if (ctxData.layerGroups.size === 0) return;
      
      const ctxWidth = x(ctxData.ctx_idx + 1) - x(ctxData.ctx_idx) - padR;
      
      ctxData.layerGroups.forEach((layerNodes: any, layerIdx: number) => {
        // 自定义排序：共有 -> 按文件顺序的单文件 -> error
        const getIsError = (n: any) => typeof n.feature_type === 'string' && n.feature_type.toLowerCase().includes('error');
        const getIsShared = (n: any) => Array.isArray(n.sourceIndices) && n.sourceIndices.length > 1;
        const getFileIndex = (n: any) => (n.sourceIndex !== undefined ? n.sourceIndex : (Array.isArray(n.sourceIndices) && n.sourceIndices.length ? n.sourceIndices[0] : 0));
        const groupIndex = (n: any) => {
          if (getIsError(n)) return 1000; // error 放最后
          if (getIsShared(n)) return 0;   // 共有优先
          return 1 + getFileIndex(n);     // 各文件按上传顺序
        };
        layerNodes.sort((a: any, b: any) => {
          const ga = groupIndex(a);
          const gb = groupIndex(b);
          if (ga !== gb) return ga - gb;
          // 同组内保持稳定：可按 logitPct 次序作为次要键（降序）
          const la = a.logitPct || 0;
          const lb = b.logitPct || 0;
          return (lb - la);
        });
        
        const maxNodesInContext = ctxData.maxCount;
        const spacing = ctxWidth / maxNodesInContext;
        
        layerNodes.forEach((node: any, i: number) => {
          const totalWidth = (layerNodes.length - 1) * spacing;
          const startX = ctxWidth - totalWidth;
          node.xOffset = startX + i * spacing;
          node.yOffset = 0;
        });
      });
    });

    positionedNodes.forEach((d: any) => {
      d.pos = [
        x(d.ctx_idx) + d.xOffset,
        y(d.layerIdx) + y.bandwidth() / 2 + d.yOffset,
      ];
    });

    // Update link paths and populate node link references
    const positionedLinks = data.links.map((d: any) => {
      const sourceNode = positionedNodes.find((n: any) => n.nodeId === d.source);
      const targetNode = positionedNodes.find((n: any) => n.nodeId === d.target);
      if (sourceNode && targetNode) {
        const [x1, y1] = sourceNode.pos;
        const [x2, y2] = targetNode.pos;
        return {
          ...d,
          pathStr: `M${x1},${y1}L${x2},${y2}`
        };
      }
      // Skip invalid links
      return null;
    }).filter(Boolean);

    return { calculatedCtxCounts, x, y, positionedNodes, positionedLinks };
  }, [data.nodes, data.links, dimensions.width, dimensions.height]);

  // Handle mouse enter/leave for exact hover detection (much more performant than closest node)
  const handleNodeMouseEnter = useCallback((nodeId: string) => {
    onNodeHover(nodeId);
  }, [onNodeHover]);

  const handleNodeMouseLeave = useCallback(() => {
    onNodeHover(null);
  }, [onNodeHover]);

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width,
          height: rect.height
        });
      }
    };

    // Initial size
    updateDimensions();

    // Set up resize observer
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    // Cleanup
    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Memoize token data calculation
  const tokenData = useMemo(() => {
    if (!data.metadata?.prompt_tokens || !positionedNodes.length) return [];
    
    const maxCtxIdx = d3.max(positionedNodes, (d: any) => d.ctx_idx) || 0;
    
    return data.metadata.prompt_tokens
      .slice(0, maxCtxIdx + 1)
      .map((token: string, index: number) => {
        const contextNodes = positionedNodes.filter((d: any) => d.ctx_idx === index);
        
        if (contextNodes.length === 0) {
          return {
            token,
            ctx_idx: index,
            x: x(index + 1) - (x(index + 1) - x(index)) / 2
          };
        }
        
        const nodeXPositions = contextNodes.map((d: any) => d.pos[0]);
        const rightX = Math.max(...nodeXPositions);
        
        return {
          token,
          ctx_idx: index,
          x: rightX
        };
      });
  }, [data.metadata?.prompt_tokens, positionedNodes, x]);

  // Early return if no data or scales
  if (!positionedNodes.length || !x || !y) {
    return (
      <div ref={containerRef} className="link-graph-container">
        <div>Loading...</div>
      </div>
    );
  }

  return (
    <div 
      ref={containerRef} 
      className="link-graph-container"
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onClick={(event) => {
          // Only clear selection if clicking on the SVG background (not on nodes)
          if (event.target === event.currentTarget) {
            onNodeClick("", false);
          }
        }}
        style={{ position: "relative", zIndex: 1 }}
      >
        {/* Atomic components - each manages its own rendering */}
        <RowBackgrounds 
          dimensions={dimensions}
          positionedNodes={positionedNodes}
          y={y}
        />
        
        <GridLines 
          dimensions={dimensions}
          calculatedCtxCounts={calculatedCtxCounts}
          x={x}
          positionedNodes={positionedNodes}
        />
        
        <YAxis 
          positionedNodes={positionedNodes}
          y={y}
        />
        
        <Links 
          positionedLinks={positionedLinks}
        />
        
        <Nodes 
          positionedNodes={positionedNodes}
          positionedLinks={positionedLinks}
          visState={{
            clickedId: visState.clickedId,
            hoveredId: visState.hoveredId
          }}
          onNodeMouseEnter={handleNodeMouseEnter}
          onNodeMouseLeave={handleNodeMouseLeave}
          onNodeClick={onNodeClick}
        />
        
        <Tooltips 
          positionedNodes={positionedNodes}
          visState={{ hoveredId: visState.hoveredId }}
          dimensions={dimensions}
        />
        
        <TokenLabels 
          tokenData={tokenData}
          dimensions={dimensions}
        />
      </svg>
    </div>
  );
};

// Memoize the component to prevent unnecessary re-renders
export const LinkGraph = React.memo(LinkGraphComponent); 