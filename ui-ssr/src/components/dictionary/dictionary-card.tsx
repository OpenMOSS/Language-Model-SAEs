import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { memo, useState } from 'react'

import { FeatureCard } from '../feature/feature-card'
import { Spinner } from '../ui/spinner'
import { FeatureList } from '@/components/feature/feature-list'

import { Card } from '@/components/ui/card'
import { featureQueryOptions, useFeatures } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

type DictionaryCardProps = {
  dictionaryName: string
}

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
    const { data, isLoading, isError, fetchNextPage, hasNextPage } =
      useFeatures({
        dictionary: dictionaryName,
        concernedFeatureIndex: 0,
      })

    const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<
      number | null
    >(null)

    const features = data?.pages.flatMap((page) => page)

    return (
      <Card className="flex h-[750px] w-[1400px] overflow-hidden">
        <div className="min-w-[350px] basis-[350px] shrink-0 flex flex-col border-r border-slate-300 bg-white">
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
              className="overflow-y-auto grow no-scrollbar"
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

        <div className="flex flex-col grow min-w-0 basis-0">
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
