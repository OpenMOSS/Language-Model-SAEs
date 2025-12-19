import { memo } from 'react'
import { Info } from '../ui/info'
import { FeatureSampleGroup } from './sample'
import { FeatureLogits } from './feature-logits'
import type { Feature, FeatureSampleCompact } from '@/types/feature'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

type FeatureCardHorizontalProps = {
  feature: Feature
  sampleGroups: {
    name: string
    samples: FeatureSampleCompact[]
  }[]
  className?: string
}

export const FeatureCardHorizontal = memo(
  ({ feature, sampleGroups, className }: FeatureCardHorizontalProps) => {
    return (
      <Card
        className={cn(
          'relative w-full overflow-hidden transition-all duration-200',
          className,
        )}
      >
        <div className="flex gap-2 h-full">
          <div className="flex flex-col gap-4 min-w-0 w-2/5 shrink-0 p-4">
            {feature.logits && (
              <FeatureLogits
                logits={{
                  topNegative: feature.logits.topNegative.slice(0, 5),
                  topPositive: feature.logits.topPositive.slice(0, 5),
                }}
              />
            )}
            <div className="flex flex-col basis-1/2 min-h-0 gap-4">
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
              <div className="flex items-center gap-4 grow">
                <div className="h-full flex flex-1 items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 mb-2 border border-dashed border-slate-200">
                  Activation histogram is not available.
                </div>
                <div className="h-full flex flex-1 items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 border border-dashed border-slate-200">
                  Logits histogram is not available.
                </div>
              </div>
            </div>
          </div>

          <div className="w-full gap-4 border-l border-slate-200 overflow-y-auto no-scrollbar">
            {sampleGroups.map(({ name, samples }) => (
              <FeatureSampleGroup
                className="mx-0 mt-0"
                key={name}
                feature={feature}
                samplingName={name}
                totalLength={samples.length}
                defaultVisibleRange={10}
                initialSamples={samples}
              />
            ))}
          </div>
        </div>
      </Card>
    )
  },
)
FeatureCardHorizontal.displayName = 'FeatureCardHorizontal'
