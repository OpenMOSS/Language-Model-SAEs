import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface TokenLabelsProps {
  tokenData: any[];
  dimensions: { width: number; height: number };
}

const BOTTOM_PADDING = 40;

export const TokenLabels: React.FC<TokenLabelsProps> = React.memo(({
  tokenData,
  dimensions
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !tokenData.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Draw position index only (no token text)
    // Use ctx_idx as key to prevent duplicates
    const labels = svg.selectAll(".position-index")
      .data(tokenData, (d: any) => d.ctx_idx);
    
    // Remove old labels that are no longer in data
    labels.exit().remove();
    
    // Add new labels
    const labelEnter = labels.enter()
      .append("text")
      .attr("class", "position-index");
    
    // Update all labels (both new and existing)
    labelEnter.merge(labels as any)
      .attr("x", (d: any) => d.x)
      .attr("y", dimensions.height - BOTTOM_PADDING + 15)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px")
      .attr("fill", "#374151")
      .text((d: any) => d.ctx_idx);
  }, [tokenData, dimensions]);

  return (
    <g ref={svgRef} className="token-labels" />
  );
});

TokenLabels.displayName = 'TokenLabels'; 