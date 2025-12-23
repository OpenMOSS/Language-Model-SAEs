import React from 'react'

interface TokenLabelsProps {
  tokenData: { token: string; ctxIdx: number; x: number }[]
  dimensions: { width: number; height: number }
}

const BOTTOM_PADDING = 50

export const TokenLabels: React.FC<TokenLabelsProps> = React.memo(
  ({ tokenData, dimensions }) => {
    return (
      <g>
        {tokenData.map((d) => (
          <text
            key={d.ctxIdx}
            x={d.x}
            y={dimensions.height - BOTTOM_PADDING + 10}
            transform={`rotate(-20, ${d.x}, ${dimensions.height - BOTTOM_PADDING + 10})`}
            textAnchor="end"
            dominantBaseline="hanging"
            style={{
              fontFamily: "'Courier New', monospace",
              fontSize: '11px',
              fontWeight: '500',
              userSelect: 'none',
              fill: '#374151',
            }}
          >
            {d.token}
          </text>
        ))}
      </g>
    )
  },
)

TokenLabels.displayName = 'TokenLabels'
