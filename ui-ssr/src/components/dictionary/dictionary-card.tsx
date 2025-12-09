import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { Layers } from 'lucide-react'
import { memo, useState } from 'react'

import { FeatureCard } from '../feature/feature-card'
import { FeatureCardCompact } from '../feature/feature-card-compact'
import { Spinner } from '../ui/spinner'
import { IntersectionObserver } from '../ui/intersection-observer'

import type { FeatureCompact } from '@/types/feature'
import { Card } from '@/components/ui/card'
import { featureQueryOptions, useFeatures } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

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
          'border-b border-slate-3 last:border-b-0',
          isSelected &&
            'bg-slate-100 hover:bg-slate-100 inset-ring-2 inset-ring-sky-600 z-10',
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
    onLoadMore,
    hasNextPage,
    isFetchingNextPage,
  }: {
    features: FeatureCompact[]
    selectedIndex: number | null
    onSelectFeature: (index: number) => void
    className?: string
    onLoadMore: () => void
    hasNextPage: boolean
    isFetchingNextPage: boolean
  }) => {
    return (
      <div className={cn('flex flex-col', className)}>
        {features.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Layers className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">No features found</p>
          </div>
        ) : (
          <>
            {features.map((feature) => (
              <FeatureListItem
                key={feature.featureIndex}
                feature={feature}
                isSelected={selectedIndex === feature.featureIndex}
                onClick={() => onSelectFeature(feature.featureIndex)}
              />
            ))}
            <IntersectionObserver
              onIntersect={onLoadMore}
              enabled={hasNextPage}
              className="h-8 w-full flex items-center justify-center p-1"
            >
              {isFetchingNextPage && (
                <Spinner isAnimating={true} className="text-slate-400" />
              )}
            </IntersectionObserver>
          </>
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
    featureIndex: number
    className?: string
  }) => {
    const { data } = useQuery({
      ...featureQueryOptions({
        dictionary: dictionaryName,
        featureIndex: featureIndex,
      }),
    })

    if (!data) {
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <Spinner isAnimating={true} />
        </div>
      )
    }

    return (
      <FeatureCard
        feature={data}
        className={cn('rounded-none border-none', className)}
      />
    )
  },
)

FeatureCardSelfQueried.displayName = 'FeatureCardSelfQueried'

export const DictionaryCard = memo(
  ({ dictionaryName }: DictionaryCardProps) => {
    const {
      data,
      isLoading,
      isError,
      fetchNextPage,
      hasNextPage,
      isFetchingNextPage,
    } = useFeatures({
      dictionary: dictionaryName,
    })

    const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<
      number | null
    >(null)

    const features = data?.pages.flatMap((page) => page)

    return (
      <Card className="flex h-[750px] w-[1600px] overflow-hidden">
        <div className="min-w-[380px] basis-[380px] shrink-0 flex flex-col border-r border-slate-300 bg-white">
          <div className="w-full h-[50px] uppercase px-4 flex items-center justify-center gap-1 border-b border-b-slate-300 shrink-0 font-semibold tracking-tight text-sm text-slate-700 cursor-default">
            Features from
            <Link
              to={'/dictionaries/$dictionaryName'}
              params={{ dictionaryName }}
              className="text-sky-600 hover:text-sky-700"
            >
              {dictionaryName.replace('_', '-')}
            </Link>
          </div>
          {features && (
            <FeatureList
              features={features}
              selectedIndex={selectedFeatureIndex}
              onSelectFeature={setSelectedFeatureIndex}
              onLoadMore={() => fetchNextPage()}
              hasNextPage={hasNextPage}
              isFetchingNextPage={isFetchingNextPage}
              className="overflow-y-auto grow"
            />
          )}
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full">
              <Spinner isAnimating={true} />
            </div>
          )}
          {isError && (
            <div className="flex flex-col items-center justify-center h-full">
              <p className="text-slate-500 text-sm font-medium">
                Error loading features
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-col grow">
          {selectedFeatureIndex !== null && (
            <>
              <div className="relative w-full h-[50px] uppercase px-4 flex items-center justify-center text-sm gap-1 border-b border-b-slate-300 shrink-0 font-semibold tracking-tight text-slate-700 cursor-default">
                Feature{' '}
                <Link
                  to={'/dictionaries/$dictionaryName/features/$featureIndex'}
                  params={{
                    dictionaryName,
                    featureIndex: selectedFeatureIndex.toString(),
                  }}
                  className="text-sky-600 hover:text-sky-700"
                >
                  #{selectedFeatureIndex}
                </Link>{' '}
                from{' '}
                <Link
                  to={'/dictionaries/$dictionaryName'}
                  params={{ dictionaryName }}
                  className="text-sky-600 hover:text-sky-700"
                >
                  {dictionaryName.replace('_', '-')}
                </Link>
                <Link
                  to={'/dictionaries/$dictionaryName/features/$featureIndex'}
                  params={{
                    dictionaryName,
                    featureIndex: selectedFeatureIndex.toString(),
                  }}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-sky-600 hover:text-sky-700"
                >
                  Show Detail
                </Link>
              </div>
              <FeatureCardSelfQueried
                dictionaryName={dictionaryName}
                featureIndex={selectedFeatureIndex}
                className="overflow-y-auto grow [scrollbar-gutter:stable]"
              />
            </>
          )}
          {selectedFeatureIndex === null && (
            <div className="flex flex-col items-center justify-center h-full self-center">
              <p className="text-slate-500 text-sm font-medium">
                Select a feature to view details
              </p>
              <p className="text-slate-400 text-xs mt-1">
                Click on any feature from the list
              </p>
            </div>
          )}
        </div>
      </Card>
    )
  },
)

DictionaryCard.displayName = 'DictionaryCard'
