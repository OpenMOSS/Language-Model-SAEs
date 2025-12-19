import { memo, useEffect, useMemo, useRef, useState } from 'react'
import type {
  Feature,
  FeatureSampleCompact,
  ImageTokenOrigin,
  TextTokenOrigin,
} from '@/types/feature'
import { cn } from '@/lib/utils'
import { getAccentStyle } from '@/utils/style'
import { findHighestActivatingToken, getZPatternForToken } from '@/utils/token'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import { useSamples } from '@/hooks/useFeatures'

/**
 * Helper function to get activation value for a given index from COO format.
 */
const getActivationValue = (
  indices: number[],
  values: number[],
  targetIndex: number,
): number => {
  const indexPosition = indices.indexOf(targetIndex)
  return indexPosition !== -1 ? values[indexPosition] : 0
}

export type TokenInfoProps = {
  text: string
  textOffset: number
  featureAct: number
  maxFeatureAct: number
  origin: TextTokenOrigin | ImageTokenOrigin
}

/**
 * Displays token information including origin and activation.
 */
export const TokenInfo = ({
  text,
  textOffset,
  featureAct,
  maxFeatureAct,
  origin,
}: TokenInfoProps) => {
  return (
    <div className="grid grid-cols-2 gap-2">
      {origin.key === 'text' ? (
        <>
          <div className="text-sm font-bold self-center">Text:</div>
          <div className="text-sm flex items-center gap-1">
            <span
              className="font-mono whitespace-pre-wrap text-slate-600 px-1 py-0.5 rounded-sm"
              style={getAccentStyle(featureAct, maxFeatureAct, 'bg')}
            >
              {text
                .slice(
                  origin.range[0] - textOffset,
                  origin.range[1] - textOffset,
                )
                .replaceAll('\n', '↵')
                .replaceAll('\t', '→')
                .replaceAll(' ', '_')}
            </span>
          </div>
          <div className="text-sm font-bold">Text Range:</div>
          <div className="text-sm">{origin.range.join(' - ')}</div>
        </>
      ) : (
        <>
          <div className="text-sm font-bold">Image Region:</div>
          <div className="text-sm">
            {origin.rect.map((n) => n.toFixed(3)).join(', ')}
          </div>
        </>
      )}
      <div className="text-sm font-bold">Activation:</div>
      <div className="text-sm">{featureAct.toFixed(3)}</div>
    </div>
  )
}

export type FeatureSampleGroupProps = {
  feature: Feature
  samplingName: string
  totalLength: number
  className?: string
  defaultVisibleRange?: number
  initialSamples?: FeatureSampleCompact[]
}

export const FeatureSampleGroup = ({
  className,
  feature,
  samplingName,
  totalLength,
  initialSamples,
  defaultVisibleRange = 50,
}: FeatureSampleGroupProps) => {
  const rangeOptions = [
    { label: 'Stacked', value: 10 },
    { label: 'Snippet', value: 30 },
    { label: 'Detail', value: 50 },
    { label: 'Full', value: Infinity },
  ]
  const [visibleRange, setVisibleRange] = useState<number>(defaultVisibleRange)

  const { data, fetchNextPage, hasNextPage, isFetching } = useSamples({
    dictionary: feature.dictionaryName,
    featureIndex: feature.featureIndex,
    samplingName,
    totalLength,
    start: initialSamples?.length ?? 0,
    visibleRange: visibleRange === Infinity ? undefined : visibleRange,
  })

  const samples = useMemo(
    () =>
      (initialSamples ?? []).concat(data?.pages.flatMap((page) => page) ?? []),
    [data, initialSamples],
  )

  const samplingNameMap = (samplingName: string) => {
    if (samplingName === 'top_activations') {
      return 'TOP ACTIVATIONS'
    } else if (/^subsample-/.test(samplingName)) {
      const [, proportion] = samplingName.split('-')
      const percentage = parseFloat(proportion) * 100
      return `SUBSAMPLING ${percentage}%`
    } else if (samplingName === 'non_activating') {
      return 'NON ACTIVATING'
    } else {
      return samplingName
    }
  }

  const samplingNameDisplay = samplingNameMap(samplingName)

  return (
    <div className={cn('flex flex-col mt-4 -mx-6', className)}>
      <div className="flex items-center justify-between bg-slate-50 py-2 px-6 border-y border-slate-200">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-muted-foreground uppercase ml-2">
            {samplingNameDisplay.split(' ').slice(0, -1).join(' ')}
          </span>
          <span className="bg-orange-500 text-white text-xs font-bold px-2 py-0.5 rounded-full uppercase shadow-sm">
            {samplingNameDisplay.split(' ').slice(-1)[0]}
          </span>
        </div>

        <div className="flex items-center bg-slate-200/50 rounded-md p-1 gap-1">
          {rangeOptions.map((option) => (
            <Button
              key={option.label}
              variant="ghost"
              size="sm"
              className={cn(
                'h-7 px-3 text-xs hover:bg-white/50',
                visibleRange === option.value
                  ? 'bg-white shadow-sm text-foreground font-bold'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              onClick={() => setVisibleRange(option.value)}
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      {samples.map((sample, i) => (
        <FeatureActivationSample
          key={i}
          sample={sample}
          maxFeatureAct={feature.maxFeatureAct}
          visibleRange={visibleRange}
          className="border-b border-slate-200 last:border-0 py-2 px-6"
        />
      ))}
      {isFetching && (
        <div className="flex justify-center">
          <div className="flex justify-center px-6 py-4">
            <Spinner isAnimating={isFetching} />
          </div>
        </div>
      )}
      {hasNextPage && !isFetching && (
        <div className="flex justify-center px-6 py-4">
          <Button className="w-full" size="sm" onClick={() => fetchNextPage()}>
            Load more
          </Button>
        </div>
      )}
    </div>
  )
}

export type FeatureActivationSampleProps = {
  sample: FeatureSampleCompact
  maxFeatureAct: number
  visibleRange?: number
  className?: string
  sampleTextClassName?: string
  showHighestActivatingToken?: boolean
  showHoverCard?: boolean
}

/**
 * Displays a single feature activation sample with text and image highlights.
 */
export const FeatureActivationSample = memo(
  ({
    sample,
    maxFeatureAct,
    visibleRange,
    className,
    sampleTextClassName,
    showHighestActivatingToken = true,
    showHoverCard = true,
  }: FeatureActivationSampleProps) => {
    const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(
      null,
    )
    const [tooltipState, setTooltipState] = useState<{
      visible: boolean
      x: number
      y: number
      highlights: { origin: TextTokenOrigin; featureAct: number }[]
    } | null>(null)
    const segmentRefs = useRef<Map<number, HTMLSpanElement>>(new Map())
    const tooltipRef = useRef<HTMLDivElement>(null)

    const tokenOffset = sample.tokenOffset ?? 0
    const textOffset = sample.textOffset ?? 0

    // Memoize text highlights processing
    const textHighlights = useMemo(
      () =>
        sample.origins
          .map((origin, index) => ({
            origin,
            featureAct: getActivationValue(
              sample.featureActsIndices,
              sample.featureActsValues,
              index + tokenOffset,
            ),
          }))
          .filter(
            (item): item is { origin: TextTokenOrigin; featureAct: number } =>
              item.origin?.key === 'text',
          ),
      [sample],
    )

    // Memoize max activation highlight
    const maxActivationHighlight = useMemo(
      () =>
        textHighlights.length > 0
          ? textHighlights.reduce(
              (max, current) =>
                current.featureAct > max.featureAct ? current : max,
              textHighlights[0],
            )
          : null,
      [textHighlights],
    )

    // Find the highest activating token for z pattern highlighting
    const highestActivatingToken = useMemo(
      () =>
        findHighestActivatingToken(
          sample.featureActsIndices,
          sample.featureActsValues,
        ),
      [sample.featureActsIndices, sample.featureActsValues],
    )

    const highestActivatingTokenText = useMemo(() => {
      if (!highestActivatingToken || !sample.text) return null
      const origin =
        sample.origins[highestActivatingToken.tokenIndex - tokenOffset]
      if (origin && origin.key === 'text') {
        return sample.text.slice(
          origin.range[0] - textOffset,
          origin.range[1] - textOffset,
        )
      }
      return null
    }, [highestActivatingToken, sample.origins, sample.text, textOffset])

    // Get z pattern for hovered token
    const hoveredZPattern = useMemo(() => {
      if (hoveredTokenIndex === null) return null
      return getZPatternForToken(
        sample.zPatternIndices,
        sample.zPatternValues,
        hoveredTokenIndex + tokenOffset,
      )
    }, [
      hoveredTokenIndex,
      sample.zPatternIndices,
      sample.zPatternValues,
      tokenOffset,
    ])

    // Memoize segments calculation
    const segments = useMemo(() => {
      const segmentList: {
        start: number
        end: number
        highlights: typeof textHighlights
        maxSegmentAct: number
        index: number
      }[] = []

      if (sample.text && textHighlights.length > 0 && maxActivationHighlight) {
        // Get all unique positions
        const positions = new Set<number>()
        textHighlights.forEach((h) => {
          positions.add(h.origin.range[0])
          positions.add(h.origin.range[1])
        })
        const sortedPositions = Array.from(positions).sort((a, b) => a - b)

        // Create segments between each pair of positions
        for (let i = 0; i < sortedPositions.length - 1; i++) {
          const start = sortedPositions[i]
          const end = sortedPositions[i + 1]

          const activeHighlights = textHighlights.filter(
            (h) => h.origin.range[0] <= start && h.origin.range[1] >= end,
          )
          if (activeHighlights.length > 0) {
            const maxSegmentAct = Math.max(
              ...activeHighlights.map((h) => h.featureAct),
            )
            segmentList.push({
              start,
              end,
              highlights: activeHighlights,
              maxSegmentAct,
              index: i,
            })
          }
        }
      }
      return segmentList
    }, [sample.text, textHighlights, maxActivationHighlight])

    // Memoize max activation segment index
    const maxActivationSegmentIndex = useMemo(
      () =>
        segments.findIndex((segment) =>
          segment.highlights.some(
            (highlight) =>
              highlight.featureAct === maxActivationHighlight?.featureAct,
          ),
        ),
      [segments, maxActivationHighlight],
    )

    // Memoize visible segments
    const visibleSegments = useMemo(
      () =>
        segments.filter((segment) => {
          if (!visibleRange) return true
          if (maxActivationSegmentIndex === -1) return true
          return (
            Math.abs(segment.index - maxActivationSegmentIndex) <= visibleRange
          )
        }),
      [segments, visibleRange, maxActivationSegmentIndex],
    )

    // Helper function to determine if a segment should be highlighted
    const getSegmentStyle = (segment: (typeof visibleSegments)[0]) => {
      if (hoveredZPattern) {
        const contribution = segment.highlights.map((highlight) => {
          // Find the token index for this highlight's origin
          const tokenIndex =
            sample.origins.findIndex((origin) => origin === highlight.origin) +
            tokenOffset
          const isContributing =
            hoveredZPattern.contributingTokens.includes(tokenIndex)
          return isContributing
            ? hoveredZPattern.contributions[
                hoveredZPattern.contributingTokens.indexOf(tokenIndex)
              ]
            : 0
        })

        const containsHoveredToken =
          hoveredTokenIndex !== null &&
          segment.highlights.some((highlight) => {
            const hoveredOrigin = sample.origins[hoveredTokenIndex]
            return hoveredOrigin === highlight.origin
          })

        if (containsHoveredToken) {
          return segment.maxSegmentAct > 0
            ? {
                style: getAccentStyle(
                  segment.maxSegmentAct,
                  maxFeatureAct,
                  'bg',
                ),
              }
            : {}
        } else {
          return {
            style: getAccentStyle(
              contribution.reduce((a, b) => a + b, 0),
              maxFeatureAct,
              'zpattern',
            ),
          }
        }
      }

      const containsHoveredToken =
        hoveredTokenIndex !== null &&
        segment.highlights.some((highlight) => {
          const hoveredOrigin = sample.origins[hoveredTokenIndex]
          return hoveredOrigin === highlight.origin
        })

      return segment.maxSegmentAct > 0
        ? {
            style: getAccentStyle(segment.maxSegmentAct, maxFeatureAct, 'bg'),
          }
        : containsHoveredToken
          ? { className: 'bg-slate-200' }
          : {}
    }

    // Get first token text for display when there's no activation
    const firstTokenText = useMemo(() => {
      if (highestActivatingTokenText) return null
      if (!sample.text || sample.origins.length === 0) return null
      const firstOrigin = sample.origins[0]
      if (firstOrigin && firstOrigin.key === 'text') {
        return sample.text.slice(
          firstOrigin.range[0] - textOffset,
          firstOrigin.range[1] - textOffset,
        )
      }
      return null
    }, [highestActivatingTokenText, sample.text, sample.origins, textOffset])

    // Handle tooltip positioning
    useEffect(() => {
      if (!tooltipState?.visible || !tooltipRef.current) return

      const tooltip = tooltipRef.current
      const viewportWidth = window.innerWidth
      const viewportHeight = window.innerHeight
      const padding = 8

      // Get tooltip dimensions after render
      const rect = tooltip.getBoundingClientRect()
      const tooltipWidth = rect.width
      const tooltipHeight = rect.height

      let x = tooltipState.x
      let y = tooltipState.y
      let transformX = '-50%'

      // Adjust horizontal position if tooltip goes off screen (accounting for -50% transform)
      const halfWidth = tooltipWidth / 2
      if (x + halfWidth > viewportWidth - padding) {
        x = viewportWidth - padding
        transformX = '-100%'
      } else if (x - halfWidth < padding) {
        x = padding
        transformX = '0%'
      }

      // Adjust vertical position if tooltip goes off screen
      if (y + tooltipHeight > viewportHeight - padding) {
        y = tooltipState.y - tooltipHeight - 4
      }
      if (y < padding) {
        y = padding
      }

      tooltip.style.left = `${x}px`
      tooltip.style.top = `${y}px`
      tooltip.style.transform = `translateX(${transformX})`
    }, [tooltipState])

    return (
      <div className={cn('w-full flex gap-4 items-center', className)}>
        {showHighestActivatingToken && (
          <div className="flex flex-col items-center justify-center min-w-[80px] gap-1">
            {(highestActivatingTokenText || firstTokenText) && (
              <span className="bg-slate-200 text-slate-700 text-xs px-2 py-0.5 rounded font-medium whitespace-nowrap max-w-[80px] overflow-hidden text-ellipsis">
                {(highestActivatingTokenText || firstTokenText)
                  ?.replaceAll('\n', '↵')
                  .replaceAll('\t', '→')
                  .replaceAll(' ', '_')}
              </span>
            )}
            {highestActivatingToken ? (
              <span className="text-orange-600 font-bold text-sm">
                {highestActivatingToken.activationValue.toFixed(2)}
              </span>
            ) : (
              <span className="text-slate-400 font-bold text-sm">0.00</span>
            )}
          </div>
        )}

        <div className="flex gap-4 w-full justify-between cursor-default">
          {/* Text display with highlights */}
          {sample.text && (
            <div className="flex flex-col gap-2">
              <div
                className={cn(
                  'relative flex flex-wrap whitespace-pre-wrap text-sm leading-relaxed font-mono text-slate-600',
                  sampleTextClassName,
                )}
              >
                {visibleSegments.map((segment, index) => {
                  const segmentText = sample.text!.slice(
                    segment.start - textOffset,
                    segment.end - textOffset,
                  )
                  const { className: segmentClassName, style: segmentStyle } =
                    getSegmentStyle(segment)

                  const handleMouseEnter = (
                    e: React.MouseEvent<HTMLSpanElement>,
                  ) => {
                    const target = e.currentTarget
                    const rect = target.getBoundingClientRect()
                    const tokenIndex = sample.origins.indexOf(
                      segment.highlights.reduce((p, c) =>
                        c.featureAct > p.featureAct ? c : p,
                      ).origin,
                    )
                    setHoveredTokenIndex(tokenIndex)

                    if (showHoverCard) {
                      setTooltipState({
                        visible: true,
                        x: rect.left + rect.width / 2,
                        y: rect.bottom + 4,
                        highlights: segment.highlights,
                      })
                    }
                  }

                  const handleMouseLeave = () => {
                    setHoveredTokenIndex(null)
                    if (showHoverCard) {
                      setTooltipState(null)
                    }
                  }

                  return (
                    <span
                      key={index}
                      ref={(el) => {
                        if (el) {
                          segmentRefs.current.set(index, el)
                        } else {
                          segmentRefs.current.delete(index)
                        }
                      }}
                      className={cn(
                        'relative inline-flex items-center',
                        segmentClassName,
                      )}
                      style={segmentStyle}
                      onMouseEnter={handleMouseEnter}
                      onMouseLeave={handleMouseLeave}
                    >
                      {segmentText.replaceAll('\n', '↵').replaceAll('\t', '→')}
                    </span>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Manual tooltip */}
        {showHoverCard && tooltipState?.visible && (
          <div
            ref={tooltipRef}
            className="fixed z-50 w-64 rounded-md border bg-popover p-4 text-popover-foreground shadow-md pointer-events-none"
            style={{
              left: tooltipState.x,
              top: tooltipState.y,
            }}
          >
            <div className="flex flex-col gap-2">
              {tooltipState.highlights.map((highlight, i) => (
                <TokenInfo
                  key={i}
                  text={sample.text!}
                  textOffset={textOffset}
                  featureAct={highlight.featureAct}
                  maxFeatureAct={maxFeatureAct}
                  origin={highlight.origin}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    )
  },
)

FeatureActivationSample.displayName = 'FeatureActivationSample'
