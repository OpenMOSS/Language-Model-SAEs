import React from 'react'
import * as d3 from 'd3'

interface YAxisProps {
  positionedNodes: { layer: number; featureType: string }[]
  y: d3.ScaleBand<number>
}

const SIDE_PADDING = 70

export const YAxis: React.FC<YAxisProps> = React.memo(
  ({ positionedNodes, y }) => {
    const hasEmbedding = positionedNodes.some(
      (d) => d.featureType === 'embedding',
    )
    const hasLogit = positionedNodes.some((d) => d.featureType === 'logit')
    const yNumTicks = (d3.max(positionedNodes, (d) => d.layer) || 0) + 2

    return (
      <g>
        {d3.range(yNumTicks).map((layerIdx: number) => {
          const yPos = (y(layerIdx) || 0) + y.bandwidth() / 2

          let label: string
          let textColor: string
          if (hasEmbedding && layerIdx === 0) {
            label = 'Emb'
            textColor = '#6b21a8' // Embedding - purple
          } else if (hasLogit && layerIdx === yNumTicks - 1) {
            label = 'Logit'
            textColor = '#9a3412' // Logits - orange
          } else if ((layerIdx - (hasEmbedding ? 1 : 0)) % 2 === 1) {
            label = `M${Math.floor((layerIdx - (hasEmbedding ? 1 : 0)) / 2)}`
            textColor = '#166534' // MLP - green
          } else {
            label = `A${Math.floor((layerIdx - (hasEmbedding ? 1 : 0)) / 2)}`
            textColor = '#1e40af' // Attention - blue
          }

          const fontSize =
            yNumTicks > 50
              ? '7px'
              : yNumTicks > 40
                ? '8px'
                : yNumTicks > 20
                  ? '10px'
                  : '12px'

          return (
            <text
              key={layerIdx}
              x={SIDE_PADDING - 12}
              y={yPos + 4}
              textAnchor="end"
              fontSize={fontSize}
              fill={textColor}
              style={{
                fontFamily:
                  "'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', 'Consolas', 'Courier New', monospace",
                userSelect: 'none',
              }}
            >
              {label}
            </text>
          )
        })}
      </g>
    )
  },
)

YAxis.displayName = 'YAxis'
