import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface LinksProps {
  positionedLinks: any[];
}

export const Links: React.FC<LinksProps> = React.memo(({
  positionedLinks,
}) => {

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !positionedLinks.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

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
      .attr("stroke", "#666666")
      .attr("stroke-width", (d: any) => d.strokeWidth || 1)
      .attr("opacity", 0.03);
  }, [positionedLinks]);

  return (
    <g ref={svgRef} className="links" />
  );
});

Links.displayName = 'Links'; 