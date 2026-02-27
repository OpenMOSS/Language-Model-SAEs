import { memo } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { Info } from '../ui/info'
import { FeatureSampleGroup } from './sample'
import { FeatureLogits, FeatureLogitsHorizontal } from './feature-logits'
import type { Feature, FeatureCompact } from '@/types/feature'
import { Card, CardContent } from '@/components/ui/card'
import { samplingsQueryOptions } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

type FeatureCardProps = {
  feature: Feature
  className?: string
}

export const FeatureCard = memo(({ feature, className }: FeatureCardProps) => {
  const samplings = useQuery(
    samplingsQueryOptions({
      dictionary: feature.dictionaryName,
      featureIndex: feature.featureIndex,
    }),
  )

  const samplingNames = samplings.data?.map((s) => s.name) ?? []
  const samplesQueries = useQueries({
    queries: samplingNames.map((name) => ({
      queryKey: ['samples', feature.dictionaryName, feature.featureIndex, name],
      enabled: false,
    })),
  })
  const isSamplesError = samplesQueries.some((q) => q.isError)

  return (
    <Card
      className={cn(
        'relative w-full overflow-hidden transition-all duration-200',
        className,
        (samplings.isError || isSamplesError) &&
          'border-red-500 hover:border-red-600',
      )}
    >
      <CardContent className="py-0">
        <div className="flex flex-col gap-2 pt-6">
          <div className="flex gap-6">
            {feature.logits && <FeatureLogits logits={feature.logits} />}
            <div className="flex flex-col basis-1/2 min-w-0 gap-4">
              <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default justify-center">
                ACTIVATION TIMES{' '}
                {((feature.actTimes / feature.nAnalyzedTokens!) * 100).toFixed(
                  4,
                )}
                %
                <Info iconSize={14}>
                  The percentage of tokens that the feature is activated on.
                  Activated {feature.actTimes.toLocaleString()} of{' '}
                  {feature.nAnalyzedTokens!.toLocaleString()} tokens.
                </Info>
              </div>
              <div className="flex flex-col items-center gap-4 grow">
                <div className="w-full basis-1/2 min-h-0">
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 mb-2 border border-dashed border-slate-200">
                    Activation histogram is not available.
                  </div>
                </div>
                <div className="w-full basis-1/2 min-h-0">
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 border border-dashed border-slate-200">
                    Logits histogram is not available.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col w-full gap-4">
            {samplings.data?.map(({ name, length }) => (
              <FeatureSampleGroup
                key={name}
                feature={feature}
                samplingName={name}
                totalLength={length}
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
})
FeatureCard.displayName = 'FeatureCard'

type FeatureCardWithSamplesProps = {
  feature: FeatureCompact
  className?: string
}
export const FeatureCardWithSamples = memo(
  ({ feature, className }: FeatureCardWithSamplesProps) => {
    return (
      <Card
        className={cn(
          'relative w-full overflow-hidden transition-all duration-200',
          className,
        )}
      >
        <CardContent className="py-0">
          <div className="flex flex-col gap-2 pt-6">
            <div className="flex gap-6">
              {feature.logits && <FeatureLogits logits={feature.logits} />}
              <div className="flex flex-col basis-1/2 min-w-0 gap-4">
                <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default justify-center">
                  ACTIVATION TIMES{' '}
                  {(
                    (feature.actTimes / feature.nAnalyzedTokens!) *
                    100
                  ).toFixed(4)}
                  %
                  <Info iconSize={14}>
                    The percentage of tokens that the feature is activated on.
                    Activated {feature.actTimes.toLocaleString()} of{' '}
                    {feature.nAnalyzedTokens!.toLocaleString()} tokens.
                  </Info>
                </div>
                <div className="flex flex-col items-center gap-4 grow">
                  <div className="w-full basis-1/2 min-h-0">
                    <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 mb-2 border border-dashed border-slate-200">
                      Activation histogram is not available.
                    </div>
                  </div>
                  <div className="w-full basis-1/2 min-h-0">
                    <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 border border-dashed border-slate-200">
                      Logits histogram is not available.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-col w-full gap-4">
              <FeatureSampleGroup
                feature={feature}
                samplingName="top_activations"
                totalLength={feature.samples.length}
                initialSamples={feature.samples}
                hideTitle={true}
              />
            </div>
          </div>
        </CardContent>
      </Card>
    )
  },
)

export const FeatureCardCompactForEmbed = memo(
  ({
    feature,
    className,
    plain = false,
    defaultVisibleRange = 20,
  }: {
    feature: FeatureCompact
    className?: string
    plain?: boolean
    defaultVisibleRange?: number
  }) => {
    const isLorsa = feature.samples.some(
      (s) => s.zPatternIndices && s.zPatternIndices.length > 0,
    )

    return (
      <div
        className={cn(
          'flex flex-col bg-white overflow-hidden',
          !plain && 'rounded-xl border border-slate-200/80 shadow-sm',
          className,
        )}
      >
        {/* Header */}
        <div className="px-5 py-4 bg-linear-to-r from-slate-50 to-white border-b border-slate-100">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              {feature.interpretation ? (
                <p className="text-sm text-slate-800 leading-relaxed">
                  {feature.interpretation.text}
                </p>
              ) : (
                <p className="text-sm text-slate-400 italic">
                  No interpretation available
                </p>
              )}
              {isLorsa && (
                <p className="text-xs text-slate-400 mt-1">
                  Hover on top activations to see z-pattern
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-xs text-slate-500 font-medium tracking-wide">
                {feature.dictionaryName}
              </span>
              <span
                className={cn(
                  'inline-flex items-center px-2 py-0.5 rounded-md text-white text-xs font-mono font-medium',
                  isLorsa ? 'bg-[#339af0]' : 'bg-[#f59f00]',
                )}
              >
                #{feature.featureIndex}
              </span>
            </div>
          </div>
        </div>

        {/* Logits */}
        {feature.logits && (
          <div className="px-5 py-4 border-b border-slate-100">
            <FeatureLogitsHorizontal logits={feature.logits} />
          </div>
        )}

        {/* Samples */}
        <div className="px-5">
          <FeatureSampleGroup
            feature={feature}
            samplingName="top_activations"
            totalLength={feature.samples.length}
            initialSamples={feature.samples}
            defaultVisibleRange={defaultVisibleRange}
            hideTitle={true}
            className="mt-0"
          />
        </div>
      </div>
    )
  },
)
