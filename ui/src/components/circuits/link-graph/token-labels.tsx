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
  console.log('🔄 TokenLabels component recomputed', { 
    tokenDataCount: tokenData.length,
    dimensions
  });

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !tokenData.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Draw prompt tokens at the bottom
    svg.selectAll(".token-label").data(tokenData, (d: any) => d.ctx_idx)
      .enter().append("text")
      .attr("class", "token-label")
      .attr("x", (d: any) => d.x)
      .attr("y", dimensions.height - BOTTOM_PADDING + 10)
      .attr("transform", (d: any) => `rotate(-45, ${d.x}, ${dimensions.height - BOTTOM_PADDING + 10})`)
      .attr("text-anchor", "end")
      .text((d: any) => d.token);
  }, [tokenData, dimensions]);

  return (
    <g ref={svgRef} className="token-labels" />
  );
});

TokenLabels.displayName = 'TokenLabels'; 