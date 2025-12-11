import { formatCss, interpolate } from 'culori'

const orangeInterpolator = interpolate(['#fff7ed', '#fbbf24'], 'oklab')

const greenInterpolator = interpolate(
  ['#dcfce7', '#bbf7d0', '#86efac', '#4ade80', '#22c55e'],
  'oklab',
)

export const getAccentStyle = (
  featureAct: number,
  maxFeatureAct: number,
  variant: 'text' | 'bg' | 'border' | 'zpattern',
): { [key: string]: string } | undefined => {
  if (featureAct <= 0 || maxFeatureAct <= 0) {
    return undefined
  }

  const ratio = Math.max(0, Math.min(featureAct / maxFeatureAct, 1))

  let colorObject

  if (variant === 'zpattern') {
    colorObject = greenInterpolator(ratio)
  } else {
    // Use lighter/brighter scale for background to ensure contrast with black text
    colorObject = orangeInterpolator(ratio)
  }

  const colorStr = colorObject ? formatCss(colorObject) : undefined

  if (!colorStr) return undefined

  switch (variant) {
    case 'text':
      return { color: colorStr }
    case 'bg':
      return { backgroundColor: colorStr }
    case 'border':
      return { borderColor: colorStr }
    case 'zpattern':
      return { backgroundColor: colorStr }
    default:
      return undefined
  }
}
