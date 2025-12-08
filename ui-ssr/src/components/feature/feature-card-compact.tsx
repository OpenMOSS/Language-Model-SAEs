import { memo } from 'react'
import { FeatureActivationSample } from './sample'
import type { FeatureCompact } from '@/types/feature'
import { cn } from '@/lib/utils'

type FeatureCardCompactProps = {
  feature: FeatureCompact
  className?: string
}

export const FeatureCardCompact = memo(
  ({ feature, className }: FeatureCardCompactProps) => {
    return (
      <div className={cn('flex flex-col gap-2 p-2', className)}>
        {!feature.interpretation && (
          <p className="text-neutral-500">No interpretation available.</p>
        )}
        {feature.interpretation && (
          <div className="font-token font-medium text-sm rounded-md w-fit">
            {feature.interpretation.text}
          </div>
        )}
        <FeatureActivationSample
          sample={feature.samples[0]}
          maxFeatureAct={feature.maxFeatureAct}
          visibleRange={5}
          showHighestActivatingToken={false}
          showHoverCard={false}
          className="pl-2 pr-2"
          sampleTextClassName="text-xs line-clamp-1"
        />
      </div>
    )
  },
)

FeatureCardCompact.displayName = 'FeatureCardCompact'
