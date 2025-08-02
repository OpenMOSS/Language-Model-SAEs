import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { LinkGraphData, VisState } from "./types";
// @ts-ignore
import d3 from "../static_js/d3";
import "./link-graph.css";

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
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [ctxCounts, setCtxCounts] = useState<any[]>([]);
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
        layerNodes.sort((a: any, b: any) => -(a.logitPct || 0) + (b.logitPct || 0));
        
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

    // Update link paths
    const positionedLinks = data.links.map((d: any) => {
      const sourceNode = positionedNodes.find((n: any) => n.nodeId === d.source);
      const targetNode = positionedNodes.find((n: any) => n.nodeId === d.target);
      if (sourceNode && targetNode) {
        const [x1, y1] = sourceNode.pos;
        const [x2, y2] = targetNode.pos;
        return {
          ...d,
          sourceNode,
          targetNode,
          pathStr: `M${x1},${y1}L${x2},${y2}`
        };
      }
      return d;
    });

    return { calculatedCtxCounts, x, y, positionedNodes, positionedLinks };
  }, [data.nodes, data.links, dimensions.width, dimensions.height]);

  // Memoize hover detection logic with throttling
  const findClosestNode = useCallback((mouseX: number, mouseY: number, maxDistance: number = 30) => {
    let closestNode: any = null;
    let closestDistance = Infinity;
    
    // Use a more efficient distance calculation (squared distance to avoid sqrt)
    positionedNodes.forEach((node) => {
      const dx = mouseX - node.pos[0];
      const dy = mouseY - node.pos[1];
      const distSquared = dx * dx + dy * dy;
      if (distSquared < closestDistance) {
        closestNode = node;
        closestDistance = distSquared;
      }
    });
    
    // Only apply sqrt to the final closest distance
    return Math.sqrt(closestDistance) <= maxDistance ? closestNode : null;
  }, [positionedNodes]);

  // Throttled mouse move handler for better performance
  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (event.shiftKey) return;
    
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Throttle hover updates to reduce lag
    requestAnimationFrame(() => {
      const closestNode = findClosestNode(mouseX, mouseY);
      const hoveredId = closestNode?.nodeId || null;
      
      if (hoveredId !== visState.hoveredId) {
        onNodeHover(hoveredId);
      }
    });
  }, [findClosestNode, visState.hoveredId, onNodeHover]);

  const handleClick = useCallback((event: React.MouseEvent) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    const closestNode = findClosestNode(mouseX, mouseY);
    
    if (!closestNode) {
      onNodeClick("", false);
    } else {
      onNodeClick(closestNode.nodeId, event.metaKey || event.ctrlKey);
    }
  }, [findClosestNode, onNodeClick]);

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

  // Memoize SVG rendering
  const renderSVG = useCallback(() => {
    if (!svgRef.current || !positionedNodes.length) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    
    // Draw background
    svg.append("rect")
      .attr("width", dimensions.width)
      .attr("height", dimensions.height - BOTTOM_PADDING)
      .attr("fill", "#F5F4EE");

    // Draw grid lines
    const earliestCtxWithNodes = d3.min(positionedNodes, (d: any) => d.ctx_idx) || 0;
    
    calculatedCtxCounts.forEach((ctxData: any) => {
      if (ctxData.ctx_idx >= earliestCtxWithNodes) {
        const xPos = x(ctxData.ctx_idx);
        svg.append("line")
          .attr("x1", xPos)
          .attr("y1", 0)
          .attr("x2", xPos)
          .attr("y2", dimensions.height - BOTTOM_PADDING)
          .attr("stroke", "rgba(255, 255, 255, 1)")
          .attr("stroke-width", "1")
      }
    });

    const yNumTicks = d3.max(positionedNodes, (d: any) => d.layerIdx) + 1;
    d3.range(yNumTicks).forEach((layerIdx: number) => {
      const yPos = y(layerIdx) + y.bandwidth() / 2;
      svg.append("line")
        .attr("x1", 0)
        .attr("y1", yPos)
        .attr("x2", dimensions.width)
        .attr("y2", yPos)
        .attr("stroke", "rgba(255, 255, 255, 1)")
        .attr("stroke-width", "1")
    });

    // Draw Y-axis ticks and labels
    d3.range(yNumTicks).forEach((layerIdx: number) => {
      const yPos = y(layerIdx) + y.bandwidth() / 2;
      
      let label: string;
      if (layerIdx === 0) {
        label = "Emb";
      } else if (layerIdx === yNumTicks - 1) {
        label = "Logit";
      } else {
        label = `L${layerIdx}`;
      }
      
      svg.append("line")
        .attr("x1", 0)
        .attr("y1", yPos)
        .attr("x2", 8)
        .attr("y2", yPos)
        .attr("stroke", "#666")
        .attr("stroke-width", "1");
      
      svg.append("text")
        .attr("x", 12)
        .attr("y", yPos + 4)
        .attr("text-anchor", "start")
        .attr("font-size", "12px")
        .attr("font-family", "Arial, sans-serif")
        .attr("fill", "#666")
        .text(label);
    });

    // Draw edges as SVG paths
    const linkSel = svg.selectAll(".link").data(positionedLinks, (d: any) => `${d.source}-${d.target}`);
    
    // Enter: create new paths
    const linkEnter = linkSel.enter().append("path")
      .attr("class", "link")
      .attr("fill", "none")
      .style("pointer-events", "none");
    
    // Merge enter and update selections to apply attributes to both new and existing elements
    linkSel.merge(linkEnter)
      .attr("d", (d: any) => d.pathStr)
      .attr("stroke", (d: any) => {
        const isConnected = visState.clickedId && 
          (d.sourceNode.nodeId === visState.clickedId || d.targetNode.nodeId === visState.clickedId);
        
        if (isConnected) {
          return d.color || "#4CAF50";
        } else {
          return "#666666";
        }
      })
      .attr("stroke-width", (d: any) => {
        const isConnected = visState.clickedId && 
          (d.sourceNode.nodeId === visState.clickedId || d.targetNode.nodeId === visState.clickedId);
        
        if (isConnected) {
          return Math.max(3, (d.strokeWidth || 1) * 2);
        } else {
          return d.strokeWidth || 1;
        }
      })
      .attr("opacity", (d: any) => {
        const isConnected = visState.clickedId && 
          (d.sourceNode.nodeId === visState.clickedId || d.targetNode.nodeId === visState.clickedId);
        
        if (isConnected) {
          return 0.9;
        } else {
          return 0.05;
        }
      });

    // Draw nodes
    const nodeSel = svg.selectAll(".node").data(positionedNodes, (d: any) => d.nodeId);
    
    nodeSel.enter().append("circle")
      .attr("class", "node")
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("r", 4)
      .attr("fill", (d: any) => d.nodeColor)
      .attr("stroke", "#000")
      .attr("stroke-width", "0.5")
      .classed("pinned", (d: any) => visState.pinnedIds.includes(d.nodeId))
      .classed("clicked", (d: any) => d.nodeId === visState.clickedId);

    // Draw hover indicators
    svg.selectAll(".hover-indicator").data(positionedNodes, (d: any) => d.nodeId)
      .enter().append("circle")
      .attr("class", "hover-indicator")
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("r", 6)
      .attr("stroke", "#f0f")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "2 2")
      .attr("fill", "none")
      .style("display", (d: any) => d.nodeId === visState.hoveredId ? "" : "none");

    // Draw prompt tokens at the bottom
    if (tokenData.length > 0) {

      svg.selectAll(".token-label").remove();

      svg.selectAll(".token-label").data(tokenData, (d: any) => d.ctx_idx)
        .enter().append("text")
        .attr("class", "token-label")
        .attr("x", (d: any) => d.x)
        .attr("y", dimensions.height - BOTTOM_PADDING + 10)
        .attr("transform", (d: any) => `rotate(-45, ${d.x}, ${dimensions.height - BOTTOM_PADDING + 10})`)
        .attr("text-anchor", "end")
        .text((d: any) => d.token);
    }
  }, [positionedNodes, positionedLinks, visState, dimensions, calculatedCtxCounts, x, y, tokenData]);

  // Update ctxCounts for debug info
  useEffect(() => {
    setCtxCounts(calculatedCtxCounts);
  }, [calculatedCtxCounts]);

  // Render the visualization
  useEffect(() => {
    // Only render SVG since it's what's actually visible
    renderSVG();
  }, [renderSVG]);

  return (
    <div ref={containerRef} className="link-graph-container">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => onNodeHover(null)}
        onClick={handleClick}
        style={{ position: "relative", zIndex: 1 }}
      />
      
      {/* Debug Info Overlay */}
      {debugInfo && (
        <div 
          style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            background: "rgba(0, 0, 0, 0.9)",
            color: "white",
            padding: "15px",
            borderRadius: "8px",
            fontSize: "12px",
            fontFamily: "monospace",
            maxWidth: "300px",
            zIndex: 10,
            border: "1px solid #666"
          }}
        >
          <div style={{ marginBottom: "10px", fontWeight: "bold", borderBottom: "1px solid #666", paddingBottom: "5px" }}>
            🐛 Node Debug Info
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>ID:</strong> {debugInfo.nodeId}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Feature:</strong> {debugInfo.featureId}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Type:</strong> {debugInfo.feature_type}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Context:</strong> {debugInfo.ctx_idx}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Layer:</strong> {debugInfo.layerIdx}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Position:</strong> [{debugInfo.pos[0].toFixed(1)}, {debugInfo.pos[1].toFixed(1)}]
          </div>
          {debugInfo.logitPct && (
            <div style={{ marginBottom: "5px" }}>
              <strong>Logit %:</strong> {(debugInfo.logitPct * 100).toFixed(2)}%
            </div>
          )}
          {debugInfo.logitToken && (
            <div style={{ marginBottom: "5px" }}>
              <strong>Logit Token:</strong> {debugInfo.logitToken}
            </div>
          )}
          {debugInfo.localClerp && (
            <div style={{ marginBottom: "5px" }}>
              <strong>Local Clerp:</strong> {debugInfo.localClerp}
            </div>
          )}
          <div style={{ marginBottom: "5px" }}>
            <strong>Links:</strong> {debugInfo.sourceLinks?.length || 0} → {debugInfo.targetLinks?.length || 0}
          </div>
          <div style={{ marginBottom: "5px" }}>
            <strong>Column Width:</strong> {ctxCounts.find((d: any) => d.ctx_idx === debugInfo.ctx_idx)?.maxCount || 0} nodes max
          </div>
          <button 
            onClick={() => setDebugInfo(null)}
            style={{
              background: "#666",
              color: "white",
              border: "none",
              padding: "5px 10px",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "10px"
            }}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
};

// Memoize the component to prevent unnecessary re-renders
export const LinkGraph = React.memo(LinkGraphComponent); 