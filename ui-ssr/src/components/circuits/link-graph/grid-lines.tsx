import { memo } from 'react'
import * as d3 from 'd3'

interface GridLinesProps {
  dimensions: { width: number; height: number }
  calculatedCtxCounts: {
    ctxIdx: number
    maxCount: number
    cumsum: number
  }[]
  x: d3.ScaleLinear<number, number>
  positionedNodes: { ctxIdx: number }[]
}

const BOTTOM_PADDING = 40

export const GridLines = memo(
  ({ dimensions, calculatedCtxCounts, x, positionedNodes }: GridLinesProps) => {
    const earliestCtxWithNodes = d3.min(positionedNodes, (d) => d.ctxIdx) || 0

    return (
      <g>
        {calculatedCtxCounts
          .filter((ctxData) => ctxData.ctxIdx >= earliestCtxWithNodes)
          .map((ctxData) => {
            const xPos = x(ctxData.ctxIdx)
            return (
              <line
                key={ctxData.ctxIdx}
                x1={xPos}
                y1={0}
                x2={xPos}
                y2={dimensions.height - BOTTOM_PADDING}
                stroke="rgba(255, 255, 255, 1)"
                strokeWidth="1"
              />
            )
          })}
      </g>
    )
  },
)

GridLines.displayName = 'GridLines'
