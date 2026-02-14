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

  const isErrorNode = (d: any) =>
    typeof d.feature_type === "string" && d.feature_type.toLowerCase().includes("error");

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const regularNodes = positionedNodes.filter((d: any) => !isErrorNode(d));
    const errorNodes = positionedNodes.filter((d: any) => isErrorNode(d));

    const isConnected = (d: any) =>
      visState.clickedId &&
      positionedLinks.some(
        (link: any) =>
          (link.source === visState.clickedId && link.target === d.nodeId) ||
          (link.target === visState.clickedId && link.source === d.nodeId)
      );
    const getStroke = (d: any) => {
      if (d.nodeId === visState.clickedId) return "#ef4444";
      if (isConnected(d)) return "#22c55e";
      return "#000";
    };
    const getStrokeWidth = (d: any) =>
      d.nodeId === visState.clickedId || isConnected(d) ? "1.5" : "0.5";
    const nodeEvents = (sel: any) =>
      sel
        .style("cursor", "pointer")
        .on("mouseenter", function (_event: any, d: any) {
          handleMouseEnter(d.nodeId);
        })
        .on("mouseleave", function () {
          handleMouseLeave();
        })
        .on("click", function (event: any, d: any) {
          event.stopPropagation();
          handleClick(d.nodeId, event.metaKey || event.ctrlKey);
        });

    // Regular nodes: circles - append first, then set all attrs on merge
    const regularSel = svg.selectAll(".node.node-regular").data(regularNodes, (d: any) => d.nodeId);
    const regularEnter = regularSel.enter().append("circle");
    const regularMerged = regularSel.merge(regularEnter)
      .attr("class", "node node-regular")
      .attr("r", 4)
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("fill", (d: any) => d.nodeColor)
      .attr("stroke", getStroke)
      .attr("stroke-width", getStrokeWidth);
    nodeEvents(regularMerged);

    // Error nodes: squares (rect)
    const size = 6;
    const errorSel = svg.selectAll(".node.node-error").data(errorNodes, (d: any) => d.nodeId);
    const errorEnter = errorSel.enter().append("rect");
    const errorMerged = errorSel.merge(errorEnter)
      .attr("class", "node node-error")
      .attr("width", size)
      .attr("height", size)
      .attr("x", (d: any) => d.pos[0] - size / 2)
      .attr("y", (d: any) => d.pos[1] - size / 2)
      .attr("fill", (d: any) => d.nodeColor)
      .attr("stroke", getStroke)
      .attr("stroke-width", getStrokeWidth);
    nodeEvents(errorMerged);

    // Hover indicators: circles for regular, rects for error
    const hoverReg = svg.selectAll(".hover-indicator.hover-regular").data(regularNodes, (d: any) => d.nodeId);
    const hoverRegEnter = hoverReg.enter().append("circle");
    hoverReg.merge(hoverRegEnter)
      .attr("class", "hover-indicator hover-regular")
      .attr("r", 6)
      .attr("cx", (d: any) => d.pos[0])
      .attr("cy", (d: any) => d.pos[1])
      .attr("stroke", "#f0f")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .style("opacity", (d: any) => (d.nodeId === visState.hoveredId ? 1 : 0));

    const hSize = 12;
    const hoverErr = svg.selectAll(".hover-indicator.hover-error").data(errorNodes, (d: any) => d.nodeId);
    const hoverErrEnter = hoverErr.enter().append("rect");
    hoverErr.merge(hoverErrEnter)
      .attr("class", "hover-indicator hover-error")
      .attr("width", hSize)
      .attr("height", hSize)
      .attr("x", (d: any) => d.pos[0] - hSize / 2)
      .attr("y", (d: any) => d.pos[1] - hSize / 2)
      .attr("stroke", "#f0f")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .style("opacity", (d: any) => (d.nodeId === visState.hoveredId ? 1 : 0));

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