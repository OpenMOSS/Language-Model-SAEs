import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface YAxisProps {
  positionedNodes: any[];
  y: d3.ScaleBand<number>;
}

export const YAxis: React.FC<YAxisProps> = React.memo(({
  positionedNodes,
  y
}) => {
  console.log('🔄 YAxis component recomputed', { 
    positionedNodesCount: positionedNodes.length
  });

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !positionedNodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const yNumTicks = d3.max(positionedNodes, (d: any) => d.layerIdx) + 1;
    
    // Draw Y-axis ticks and labels
    d3.range(yNumTicks).forEach((layerIdx: number) => {
      const yPos = y(layerIdx) + y.bandwidth() / 2;
      
      let label: string;
      if (layerIdx === 0) {
        label = "Emb";
      } else if (layerIdx === yNumTicks - 1) {
        label = "Logit";
      } else if (layerIdx % 2 === 0) {
        label = `M${Math.floor(layerIdx / 2) - 1}`;
      } else {
        label = `A${Math.floor(layerIdx / 2)}`;
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
  }, [positionedNodes, y]);

  return (
    <g ref={svgRef} className="y-axis" />
  );
});

YAxis.displayName = 'YAxis'; 