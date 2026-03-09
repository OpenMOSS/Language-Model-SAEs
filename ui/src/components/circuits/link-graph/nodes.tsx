import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface NodesProps {
  positionedNodes: any[];
  positionedLinks: any[];
  clickedId: string | null;
  hoveredId: string | null;
  onNodeMouseEnter: (nodeId: string) => void;
  onNodeMouseLeave: () => void;
  onNodeClick: (nodeId: string, metaKey: boolean) => void;
}

export const Nodes: React.FC<NodesProps> = React.memo(({
  positionedNodes,
  positionedLinks,
  clickedId,
  hoveredId,
  onNodeMouseEnter,
  onNodeMouseLeave,
  onNodeClick
}) => {

  const svgRef = useRef<SVGSVGElement>(null);

  // Use refs for callbacks so effect doesn't re-run when parent re-renders (e.g. on hoveredId change).
  // In multi-file mode, handleFeatureHover depends on hoveredId, so the callback chain gets new refs
  // on every hover, which would cause this effect to run, destroy DOM, recreate nodes, and trigger
  // mouseenter again -> infinite loop. Refs avoid that.
  const onNodeMouseEnterRef = useRef(onNodeMouseEnter);
  const onNodeMouseLeaveRef = useRef(onNodeMouseLeave);
  const onNodeClickRef = useRef(onNodeClick);
  onNodeMouseEnterRef.current = onNodeMouseEnter;
  onNodeMouseLeaveRef.current = onNodeMouseLeave;
  onNodeClickRef.current = onNodeClick;

  const isErrorNode = (d: any) =>
    typeof d.feature_type === "string" && d.feature_type.toLowerCase().includes("error");

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const regularNodes = positionedNodes.filter((d: any) => !isErrorNode(d));
    const errorNodes = positionedNodes.filter((d: any) => isErrorNode(d));

    const isConnected = (d: any) =>
      clickedId &&
      positionedLinks.some(
        (link: any) =>
          (link.source === clickedId && link.target === d.nodeId) ||
          (link.target === clickedId && link.source === d.nodeId)
      );
    const getStroke = (d: any) => {
      if (d.nodeId === clickedId) return "#ef4444";
      if (isConnected(d)) return "#22c55e";
      return "#000";
    };
    const getStrokeWidth = (d: any) =>
      d.nodeId === clickedId || isConnected(d) ? "1.5" : "0.5";
    const nodeEvents = (sel: any) =>
      sel
        .style("cursor", "pointer")
        .on("mouseenter", function (_event: any, d: any) {
          onNodeMouseEnterRef.current(d.nodeId);
        })
        .on("mouseleave", function () {
          onNodeMouseLeaveRef.current();
        })
        .on("click", function (event: any, d: any) {
          event.stopPropagation();
          const nodeId = d?.nodeId ?? d?.id;
          onNodeClickRef.current(nodeId, event.metaKey || event.ctrlKey);
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
      .style("opacity", (d: any) => (d.nodeId === hoveredId ? 1 : 0));

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
      .style("opacity", (d: any) => (d.nodeId === hoveredId ? 1 : 0));

    // Do NOT call handleMouseLeave in cleanup - it causes flicker when effect re-runs
    // (e.g. on clickedId change). Hover is cleared by actual mouseleave on nodes.
  }, [positionedNodes, positionedLinks, clickedId]);

  // Separate effect: update hover indicator opacity when hoveredId changes (no DOM recreation)
  useEffect(() => {
    if (!svgRef.current) return;
    const sel = d3.select(svgRef.current);
    sel.selectAll(".hover-indicator").style("opacity", (d: any) => (d.nodeId === hoveredId ? 1 : 0));
  }, [hoveredId]);

  return (
    <g 
      ref={svgRef} 
      className="nodes"
    />
  );
});

Nodes.displayName = 'Nodes'; 