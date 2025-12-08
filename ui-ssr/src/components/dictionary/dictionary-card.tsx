import { useQuery } from '@tanstack/react-query'
import { memo, useState } from 'react'
import { Layers } from 'lucide-react'
import { ProgressBar } from '../ui/progress-bar'
import { FeatureCard } from '../feature/feature-card'
import { FeatureCardCompact } from '../feature/feature-card-compact'
import { Info } from '../ui/info'
import { Spinner } from '../ui/spinner'
import type { FeatureCompact } from '@/types/feature'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { featureQueryOptions, useFeatures } from '@/hooks/useFeatures'

type DictionaryCardProps = {
  dictionaryName: string
}

const FeatureListItem = memo(
  ({
    feature,
    isSelected,
    onClick,
  }: {
    feature: FeatureCompact
    isSelected: boolean
    onClick: () => void
  }) => {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn(
          'w-full text-left transition-all duration-150 cursor-pointer relative',
          'hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-700/50',
          'border-b border-slate-100 last:border-b-0',
          isSelected &&
            'bg-slate-100 hover:bg-slate-100 ring-2 ring-sky-600 z-10',
        )}
      >
        <FeatureCardCompact
          feature={feature}
          className={cn('pointer-events-none')}
        />
      </button>
    )
  },
)

FeatureListItem.displayName = 'FeatureListItem'

const FeatureList = memo(
  ({
    features,
    selectedIndex,
    onSelectFeature,
    className,
  }: {
    features: FeatureCompact[]
    selectedIndex: number | null
    onSelectFeature: (index: number) => void
    className?: string
  }) => {
    return (
      <div className={cn('flex flex-col', className)}>
        {features.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Layers className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">No features found</p>
          </div>
        ) : (
          features.map((feature) => (
            <FeatureListItem
              key={feature.featureIndex}
              feature={feature}
              isSelected={selectedIndex === feature.featureIndex}
              onClick={() => onSelectFeature(feature.featureIndex)}
            />
          ))
        )}
      </div>
    )
  },
)

FeatureList.displayName = 'FeatureList'

const FeatureCardSelfQueried = memo(
  ({
    dictionaryName,
    featureIndex,
    className,
  }: {
    dictionaryName: string
    featureIndex: number | null
    className?: string
  }) => {
    const { data } = useQuery({
      ...featureQueryOptions({
        dictionary: dictionaryName,
        featureIndex: featureIndex ?? 0,
      }),
      enabled: featureIndex !== null,
    })

    if (featureIndex === null) {
      return (
        <Card className={className}>
          <CardContent className="flex flex-col items-center justify-center h-full">
            <p className="text-slate-500 text-sm font-medium">
              Select a feature to view details
            </p>
            <p className="text-slate-400 text-xs mt-1">
              Click on any feature from the list
            </p>
          </CardContent>
        </Card>
      )
    }

    if (!data) {
      return (
        <Card className={className}>
          <CardContent className="flex flex-col items-center justify-center h-full">
            <Spinner isAnimating={true} />
          </CardContent>
        </Card>
      )
    }

    return <FeatureCard feature={data} className={className} />
  },
)

FeatureCardSelfQueried.displayName = 'FeatureCardSelfQueried'

export const DictionaryCard = memo(
  ({ dictionaryName }: DictionaryCardProps) => {
    const { data, isLoading, isError } = useFeatures({
      dictionary: dictionaryName,
    })

    const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<
      number | null
    >(null)

    const features = data?.pages.flatMap((page) => page) ?? []

    return (
      <div className="flex gap-4 h-[750px] w-[1600px]">
        <Card className="min-w-[380px] basis-[380px] shrink-0 flex flex-col rounded-xl border border-slate-200 bg-white overflow-hidden">
          <CardHeader className="px-4 py-2.5 border-b border-slate-100 bg-slate-50/50">
            <CardTitle className="text-xs font-semibold tracking-wide text-slate-500">
              Features from
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-y-auto p-0.5">
            <FeatureList
              features={features}
              selectedIndex={selectedFeatureIndex}
              onSelectFeature={setSelectedFeatureIndex}
            />
          </CardContent>
        </Card>
        <FeatureCardSelfQueried
          dictionaryName={dictionaryName}
          featureIndex={selectedFeatureIndex}
          className="grow overflow-y-auto"
        />
      </div>
    )
  },
)

DictionaryCard.displayName = 'DictionaryCard'
