import React, { useEffect, useRef } from 'react';
// @ts-ignore
import d3 from "../static_js/d3";

interface TooltipsProps {
  positionedNodes: any[];
  visState: { hoveredId: string | null };
  dimensions: { width: number; height: number };
}

const BOTTOM_PADDING = 40;

export const Tooltips: React.FC<TooltipsProps> = React.memo(({
  positionedNodes,
  visState,
  dimensions
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Clear any existing timeout when component unmounts or hover state changes
  useEffect(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, [visState.hoveredId]);

  // Set a fallback timeout to clear tooltips if mouse leave events fail
  useEffect(() => {
    if (visState.hoveredId && !timeoutRef.current) {
      timeoutRef.current = setTimeout(() => {
        console.log('🔄 Tooltips: Fallback timeout clearing tooltip');
        // This will trigger a re-render and clear the tooltip
        timeoutRef.current = null;
      }, 10000); // 10 second fallback
    }

    // Cleanup function
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [visState.hoveredId]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    
    // Clear existing tooltips
    svg.selectAll("*").remove();

    // If no hovered ID, don't show any tooltip
    if (!visState.hoveredId) {
      console.log('🔄 Tooltips: No hovered ID, clearing tooltips');
      return;
    }

    const hoveredNode = positionedNodes.find((d: any) => d.nodeId === visState.hoveredId);
    
    if (hoveredNode) {
      console.log('🔄 Tooltips: Creating tooltip for node:', visState.hoveredId);
      const tooltip = svg.append("g")
        .attr("class", "clerp-tooltip");
      
      // Determine tooltip content
      let tooltipText = "";
      if (hoveredNode.localClerp) {
        tooltipText = hoveredNode.localClerp;
      } else if (hoveredNode.remoteClerp) {
        tooltipText = hoveredNode.remoteClerp;
      } else {
        // Fallback: show basic node info
        tooltipText = `Feature: ${hoveredNode.featureId} (Layer ${hoveredNode.layerIdx})`;
      }

      // 追加来源文件信息（如果有）
      if (hoveredNode.sourceFiles && hoveredNode.sourceFiles.length) {
        const nodeType = (hoveredNode.feature_type || '').toLowerCase();
        // 对于 logit 和 embedding 节点，不显示来源文件信息
        if (nodeType !== 'logit' && nodeType !== 'embedding') {
          const files = hoveredNode.sourceFiles.join(', ');
          tooltipText = `${tooltipText} | Source: ${files}`;
        }
      }
 
      // Calculate tooltip dimensions based on text content
      const textWidth = tooltipText.length * 6; // Approximate character width
      const tooltipWidth = Math.max(120, textWidth + 20); // Minimum 120px, or text width + padding
      const tooltipHeight = 20;
      const padding = 10;
      
      let tooltipX = hoveredNode.pos[0] + padding;
      let tooltipY = hoveredNode.pos[1] - 15;
      
      // If tooltip would go off the right edge, position it to the left of the node
      if (tooltipX + tooltipWidth > dimensions.width - padding) {
        tooltipX = hoveredNode.pos[0] - tooltipWidth - padding;
      }
      
      // If tooltip would go off the top edge, position it below the node
      if (tooltipY < padding) {
        tooltipY = hoveredNode.pos[1] + padding;
      }
      
      // If tooltip would go off the bottom edge, position it above the node
      if (tooltipY + tooltipHeight > dimensions.height - BOTTOM_PADDING - padding) {
        tooltipY = hoveredNode.pos[1] - tooltipHeight - padding;
      }
      
      // Background rectangle
      tooltip.append("rect")
        .attr("x", tooltipX)
        .attr("y", tooltipY)
        .attr("width", tooltipWidth)
        .attr("height", tooltipHeight)
        .attr("fill", "rgba(0, 0, 0, 0.8)")
        .attr("rx", 2);
      
      // Tooltip text
      tooltip.append("text")
        .attr("x", tooltipX + 5)
        .attr("y", tooltipY + 13)
        .attr("fill", "white")
        .attr("font-size", "10px")
        .text(tooltipText);
    }
  }, [positionedNodes, visState.hoveredId, dimensions]);

  return (
    <g ref={svgRef} className="tooltips" />
  );
});

Tooltips.displayName = 'Tooltips'; 