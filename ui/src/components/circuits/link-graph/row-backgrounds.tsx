import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface RowBackgroundsProps {
  dimensions: { width: number; height: number };
  positionedNodes: any[];
  y: d3.ScaleBand<number>;
}

export const RowBackgrounds: React.FC<RowBackgroundsProps> = React.memo(({
  dimensions,
  positionedNodes,
  y
}) => {
  console.log('🔄 RowBackgrounds component recomputed', { 
    dimensions,
    positionedNodesCount: positionedNodes.length
  });

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const yNumTicks = d3.max(positionedNodes, (d: any) => d.layerIdx) + 1;
    
    // Draw alternating row backgrounds
    d3.range(yNumTicks).forEach((layerIdx: number) => {
      const yPos = y(layerIdx);
      const rowHeight = y.bandwidth();
      
      // Alternate between two subtle background colors
      const backgroundColor = layerIdx % 2 === 0 ? "#F5F4EE" : "#EBE9E0";
      
      svg.append("rect")
        .attr("x", 0)
        .attr("y", yPos)
        .attr("width", dimensions.width)
        .attr("height", rowHeight)
        .attr("fill", backgroundColor);
    });
  }, [dimensions, positionedNodes, y]);

  return (
    <g ref={svgRef} className="row-backgrounds" />
  );
});

RowBackgrounds.displayName = 'RowBackgrounds'; 