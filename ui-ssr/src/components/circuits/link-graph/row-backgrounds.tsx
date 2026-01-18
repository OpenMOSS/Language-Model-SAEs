import React from 'react'
import * as d3 from 'd3'

interface RowBackgroundsProps {
  dimensions: { width: number; height: number }
  positionedNodes: { layer: number; featureType: string }[]
  y: d3.ScaleBand<number>
}

const SIDE_PADDING = 70

export const RowBackgrounds: React.FC<RowBackgroundsProps> = React.memo(
  ({ dimensions, positionedNodes, y }) => {
    const hasEmbedding = positionedNodes.some(
      (d) => d.featureType === 'embedding',
    )
    const hasLogit = positionedNodes.some((d) => d.featureType === 'logit')
    const yNumTicks = (d3.max(positionedNodes, (d) => d.layer) || 0) + 2

    return (
      <g>
        {d3.range(yNumTicks).map((layerIdx: number) => {
          const yPos = y(layerIdx) || 0
          const rowHeight = y.bandwidth() - 1

          let backgroundColor: string
          if (hasEmbedding && layerIdx === 0) {
            backgroundColor = '#e5e7eb' // Embedding - slate with slight purple tint
          } else if (hasLogit && layerIdx === yNumTicks - 1) {
            backgroundColor = '#e8e4e0' // Logits - slate with slight warm tint
          } else if ((layerIdx - (hasEmbedding ? 1 : 0)) % 2 === 1) {
            backgroundColor = '#e2e8f0' // MLP - original slate
          } else {
            backgroundColor = '#e0e7ef' // Attention - slate with slight blue tint
          }

          return (
            <rect
              key={layerIdx}
              x={SIDE_PADDING}
              y={yPos}
              width={dimensions.width}
              height={rowHeight}
              fill={backgroundColor}
            />
          )
        })}
      </g>
    )
  },
)

RowBackgrounds.displayName = 'RowBackgrounds'
