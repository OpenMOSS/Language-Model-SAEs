import React, { useEffect, useRef, useCallback } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface NodesProps {
  positionedNodes: any[];
  positionedLinks: any[];
  visState: { 
    clickedId: string | null; 
    hoveredId: string | null;
  };
  onNodeMouseEnter: (nodeId: string) => void;
  onNodeMouseLeave: () => void;
  onNodeClick: (nodeId: string, metaKey: boolean) => void;
}

export const Nodes: React.FC<NodesProps> = React.memo(({
  positionedNodes,
  positionedLinks,
  visState,
  onNodeMouseEnter,
  onNodeMouseLeave,
  onNodeClick
}) => {

  const svgRef = useRef<SVGSVGElement>(null);

  // Memoize the event handlers to prevent them from being recreated on every render
  const handleMouseEnter = useCallback((nodeId: string) => {
    onNodeMouseEnter(nodeId);
  }, [onNodeMouseEnter]);

  const handleMouseLeave = useCallback(() => {
    onNodeMouseLeave();
  }, [onNodeMouseLeave]);

  const handleClick = useCallback((nodeId: string, metaKey: boolean) => {
    onNodeClick(nodeId, metaKey);
  }, [onNodeClick]);

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Draw nodes
    const nodeSel = svg.selectAll(".node").data(positionedNodes, (d: any) => d.nodeId);
    
    // Enter: create new nodes
    const nodeEnter = nodeSel.enter().append("circle")
      .attr("class", "node")
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("r", 4)
      .attr("fill", (d: any) => d.nodeColor)
      .attr("stroke", "#000")
      .attr("stroke-width", "0.5")
      .classed("clicked", (d: any) => d.nodeId === visState.clickedId)
      .classed("connected", (d: any) => {
        if (!visState.clickedId) return false;
        // Check if this node is connected to the clicked node
        return positionedLinks.some(link => 
          (link.source === visState.clickedId && link.target === d.nodeId) ||
          (link.target === visState.clickedId && link.source === d.nodeId)
        );
      })
      .style("cursor", "pointer")
      .on("mouseenter", function(event: any, d: any) {
        handleMouseEnter(d.nodeId);
      })
      .on("mouseleave", function(event: any, d: any) {
        handleMouseLeave();
      })
      .on("click", function(event: any, d: any) {
        event.stopPropagation(); // Prevent event bubbling
        const metaKey = event.metaKey || event.ctrlKey;
        handleClick(d.nodeId, metaKey);
      });
    
    // Merge enter and update selections to apply attributes to both new and existing elements
    nodeSel.merge(nodeEnter)
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("fill", (d: any) => d.nodeColor)
      .classed("clicked", (d: any) => d.nodeId === visState.clickedId)
      .classed("connected", (d: any) => {
        if (!visState.clickedId) return false;
        // Check if this node is connected to the clicked node
        return positionedLinks.some(link => 
          (link.source === visState.clickedId && link.target === d.nodeId) ||
          (link.target === visState.clickedId && link.source === d.nodeId)
        );
      })
      .style("cursor", "pointer")
      .on("mouseenter", function(event: any, d: any) {
        handleMouseEnter(d.nodeId);
      })
      .on("mouseleave", function(event: any, d: any) {
        handleMouseLeave();
      })
      .on("click", function(event: any, d: any) {
        event.stopPropagation(); // Prevent event bubbling
        const metaKey = event.metaKey || event.ctrlKey;
        handleClick(d.nodeId, metaKey);
      });

    // Draw hover indicators (only when hovering over nodes)
    svg.selectAll(".hover-indicator").data(positionedNodes, (d: any) => d.nodeId)
      .enter().append("circle")
      .attr("class", "hover-indicator")
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("r", 6)
      .attr("stroke", "#f0f")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .style("opacity", (d: any) => d.nodeId === visState.hoveredId ? 1 : 0);

    // Cleanup function to clear hover state when component unmounts or nodes change
    return () => {
      handleMouseLeave();
    };
  }, [positionedNodes, positionedLinks, visState, handleMouseEnter, handleMouseLeave, handleClick]);

  return (
    <g 
      ref={svgRef} 
      className="nodes"
    />
  );
});

Nodes.displayName = 'Nodes'; 