import { memo, useMemo, useState } from 'react'
import type {
  Feature,
  FeatureSampleCompact,
  ImageTokenOrigin,
  TextTokenOrigin,
} from '@/types/feature'
import { cn } from '@/lib/utils'
import { getAccentClassname } from '@/utils/style'
import { findHighestActivatingToken, getZPatternForToken } from '@/utils/token'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/ui/spinner'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card'
import { useSamples } from '@/hooks/useFeatures'

/**
 * Helper function to get activation value for a given index from COO format.
 */
const getActivationValue = (
  indices: Array<number>,
  values: Array<number>,
  targetIndex: number,
): number => {
  const indexPosition = indices.indexOf(targetIndex)
  return indexPosition !== -1 ? values[indexPosition] : 0
}

export type TokenInfoProps = {
  featureAct: number
  maxFeatureAct: number
  origin: TextTokenOrigin | ImageTokenOrigin
}

/**
 * Displays token information including origin and activation.
 */
export const TokenInfo = ({
  featureAct,
  maxFeatureAct,
  origin,
}: TokenInfoProps) => {
  return (
    <div className="grid grid-cols-2 gap-2">
      {origin.key === 'text' ? (
        <>
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
      <div
        className={cn(
          'text-sm',
          getAccentClassname(featureAct, maxFeatureAct, 'text'),
        )}
      >
        {featureAct.toFixed(3)}
      </div>
    </div>
  )
}

export type FeatureSampleGroupProps = {
  feature: Feature
  samplingName: string
  totalLength: number
}

export const FeatureSampleGroup = ({
  feature,
  samplingName,
  totalLength,
}: FeatureSampleGroupProps) => {
  const { data, fetchNextPage, hasNextPage, isFetching } = useSamples({
    dictionary: feature.dictionaryName,
    featureIndex: feature.featureIndex,
    samplingName,
    totalLength,
  })

  const samples = useMemo(
    () => data?.pages.flatMap((page) => page) ?? [],
    [data],
  )

  const samplingNameMap = (samplingName: string) => {
    if (samplingName === 'top_activations') {
      return 'TOP ACTIVATIONS'
    } else if (/^subsample-/.test(samplingName)) {
      const [, proportion] = samplingName.split('-')
      const percentage = parseFloat(proportion) * 100
      return `SUBSAMPLING ${percentage}%`
    } else {
      return samplingName
    }
  }

  const [visibleRange, setVisibleRange] = useState<number>(50)

  const rangeOptions = [
    { label: 'Stacked', value: 10 },
    { label: 'Snippet', value: 30 },
    { label: 'Detail', value: 50 },
    { label: 'Full', value: Infinity },
  ]

  const samplingNameDisplay = samplingNameMap(samplingName)

  return (
    <div className="flex flex-col mt-4 -mx-6">
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
      <div className="flex justify-center">
        <Spinner isAnimating={isFetching} className="mt-4" />
      </div>
      {hasNextPage && !isFetching && (
        <Button size="sm" className="mx-6 mt-4" onClick={() => fetchNextPage()}>
          Load more
        </Button>
      )}
    </div>
  )
}

export type FeatureActivationSampleProps = {
  sample: FeatureSampleCompact
  maxFeatureAct: number
  visibleRange?: number
  className?: string
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
  }: FeatureActivationSampleProps) => {
    const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(
      null,
    )

    // Memoize text highlights processing
    const textHighlights = useMemo(
      () =>
        sample.origins
          .map((origin, index) => ({
            origin,
            featureAct: getActivationValue(
              sample.featureActsIndices,
              sample.featureActsValues,
              index,
            ),
          }))
          .filter(
            (item): item is { origin: TextTokenOrigin; featureAct: number } =>
              item.origin?.key === 'text',
          ),
      [sample.origins, sample.featureActsIndices, sample.featureActsValues],
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
      const origin = sample.origins[highestActivatingToken.tokenIndex]
      if (origin && origin.key === 'text') {
        return sample.text.slice(origin.range[0], origin.range[1])
      }
      return null
    }, [highestActivatingToken, sample.origins, sample.text])

    // Get z pattern for hovered token
    const hoveredZPattern = useMemo(() => {
      if (hoveredTokenIndex === null) return null
      return getZPatternForToken(
        sample.zPatternIndices,
        sample.zPatternValues,
        hoveredTokenIndex,
      )
    }, [hoveredTokenIndex, sample.zPatternIndices, sample.zPatternValues])

    // Memoize segments calculation
    const segments = useMemo(() => {
      const segmentList: Array<{
        start: number
        end: number
        highlights: typeof textHighlights
        maxSegmentAct: number
        index: number
      }> = []

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
    const getSegmentHighlightClass = (segment: (typeof visibleSegments)[0]) => {
      if (hoveredZPattern) {
        const contribution = segment.highlights.map((highlight) => {
          // Find the token index for this highlight's origin
          const tokenIndex = sample.origins.findIndex(
            (origin) => origin === highlight.origin,
          )
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
            ? getAccentClassname(segment.maxSegmentAct, maxFeatureAct, 'bg')
            : ''
        } else {
          return getAccentClassname(
            contribution.reduce((a, b) => a + b, 0),
            maxFeatureAct,
            'zpattern',
          )
        }
      }

      const containsHoveredToken =
        hoveredTokenIndex !== null &&
        segment.highlights.some((highlight) => {
          const hoveredOrigin = sample.origins[hoveredTokenIndex]
          return hoveredOrigin === highlight.origin
        })

      return segment.maxSegmentAct > 0
        ? getAccentClassname(segment.maxSegmentAct, maxFeatureAct, 'bg')
        : containsHoveredToken
          ? 'bg-slate-200'
          : ''
    }

    // Get first token text for display when there's no activation
    const firstTokenText = useMemo(() => {
      if (highestActivatingTokenText) return null
      if (!sample.text || sample.origins.length === 0) return null
      const firstOrigin = sample.origins[0]
      if (firstOrigin && firstOrigin.key === 'text') {
        return sample.text.slice(firstOrigin.range[0], firstOrigin.range[1])
      }
      return null
    }, [highestActivatingTokenText, sample.text, sample.origins])

    return (
      <div className={cn('w-full flex gap-4 items-center', className)}>
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

        <div className="flex gap-4 w-full justify-between cursor-default">
          {/* Text display with highlights */}
          {sample.text && (
            <div className="flex flex-col gap-2">
              <div className="relative flex flex-wrap whitespace-pre-wrap text-sm leading-relaxed font-mono text-slate-600">
                {visibleSegments.map((segment, index) => {
                  const segmentText = sample.text!.slice(
                    segment.start,
                    segment.end,
                  )

                  return (
                    <span key={index} className="inline-flex items-center">
                      <HoverCard openDelay={0} closeDelay={0}>
                        <HoverCardTrigger asChild>
                          <span
                            className={cn(
                              'relative inline-flex items-center',
                              getSegmentHighlightClass(segment),
                            )}
                            onMouseEnter={() => {
                              setHoveredTokenIndex(
                                sample.origins.indexOf(
                                  segment.highlights.reduce((p, c) =>
                                    c.featureAct > p.featureAct ? c : p,
                                  ).origin,
                                ),
                              )
                            }}
                            onMouseLeave={() => {
                              setHoveredTokenIndex(null)
                            }}
                          >
                            {segmentText
                              .replaceAll('\n', '↵')
                              .replaceAll('\t', '→')}
                          </span>
                        </HoverCardTrigger>
                        <HoverCardContent>
                          <div className="flex flex-col gap-2">
                            {segment.highlights.map((highlight, i) => (
                              <TokenInfo
                                key={i}
                                featureAct={highlight.featureAct}
                                maxFeatureAct={maxFeatureAct}
                                origin={highlight.origin}
                              />
                            ))}
                          </div>
                        </HoverCardContent>
                      </HoverCard>
                    </span>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  },
)
