import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface GridLinesProps {
  dimensions: { width: number; height: number };
  calculatedCtxCounts: any[];
  x: d3.ScaleLinear<number, number>;
  positionedNodes: any[];
}

const BOTTOM_PADDING = 40;

export const GridLines: React.FC<GridLinesProps> = React.memo(({
  dimensions,
  calculatedCtxCounts,
  x,
  positionedNodes
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

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
          .attr("stroke-width", "1");
      }
    });
  }, [dimensions, calculatedCtxCounts, x, positionedNodes]);

  return (
    <g ref={svgRef} className="grid-lines" />
  );
});

GridLines.displayName = 'GridLines'; 