import { formatCss, interpolate } from 'culori'

const orangeInterpolator = interpolate(['#fff7ed', '#fbbf24'], 'oklab')

const greenInterpolator = interpolate(
  ['#dcfce7', '#bbf7d0', '#86efac', '#4ade80', '#22c55e'],
  'oklab',
)

const positiveWeightInterpolator = interpolate(
  ['#f8fafc', '#f0fdf4', '#dcfce7', '#bbf7d0'],
  'oklab',
)

const negativeWeightInterpolator = interpolate(
  ['#f8fafc', '#fff1f2', '#ffe4e6', '#fecdd3'],
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

export const getWeightStyle = (
  weight: number,
): { backgroundColor: string; borderColor: string } | undefined => {
  const absWeight = Math.abs(weight)
  // Normalize weight to 0-1 range (clamp if larger)
  const normalized = Math.max(0, Math.min(Math.log10(absWeight) + 1, 2) / 2)

  const interpolator =
    weight > 0 ? positiveWeightInterpolator : negativeWeightInterpolator

  const bgColor = formatCss(interpolator(normalized * 0.6)) // More visible colors
  const borderColor = formatCss(interpolator(normalized)) // Full range for border

  return {
    backgroundColor: bgColor,
    borderColor: borderColor,
  }
}
