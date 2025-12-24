import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { z } from 'zod'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { FeatureBookmarkButton } from '@/components/feature/bookmark-button'
import { FeatureCard } from '@/components/feature/feature-card'
import { FeatureInterpretation } from '@/components/feature/feature-interpretation'
import { LabeledInput } from '@/components/ui/labeled-input'
import { LabeledSelect } from '@/components/ui/labeled-select'
import { InferenceCard } from '@/components/feature/inference-card'

import {
  dictionariesQueryOptions,
  featureQueryOptions,
  samplingsQueryOptions,
  useFeatures,
} from '@/hooks/useFeatures'
import { FeatureList } from '@/components/feature/feature-list'

const searchParamsSchema = z.object({
  dictionary: z.string().optional(),
  featureIndex: z.coerce.number().optional(),
  analysis: z.string().optional(),
})

export const Route = createFileRoute(
  '/dictionaries/$dictionaryName/features/$featureIndex',
)({
  validateSearch: searchParamsSchema,
  component: FeaturesPage,
  loader: async ({ context, params }) => {
    const dictionaries = await context.queryClient.ensureQueryData(
      dictionariesQueryOptions(),
    )
    context.queryClient.prefetchQuery(
      samplingsQueryOptions({
        dictionary: params.dictionaryName,
        featureIndex: Number(params.featureIndex),
      }),
    )
    const feature = await context.queryClient.fetchQuery(
      featureQueryOptions({
        dictionary: params.dictionaryName,
        featureIndex: Number(params.featureIndex),
      }),
    )
    return {
      dictionaries,
      feature,
      dictionaryName: params.dictionaryName,
      featureIndex: Number(params.featureIndex),
    }
  },
})

function FeaturesPage() {
  const navigate = useNavigate()

  const { dictionaries, feature, dictionaryName, featureIndex } =
    Route.useLoaderData()

  const [selectedDictionary, setSelectedDictionary] =
    useState<string>(dictionaryName)

  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<string>(
    featureIndex.toString(),
  )

  const navigateToFeature = async (
    dictionaryName: string,
    featureIndex: number,
  ) => {
    await navigate({
      to: '/dictionaries/$dictionaryName/features/$featureIndex',
      params: {
        dictionaryName,
        featureIndex: featureIndex.toString(),
      },
    })
    setSelectedFeatureIndex(featureIndex.toString())
  }

  const {
    data: featuresData,
    fetchNextPage,
    hasNextPage,
    fetchPreviousPage,
    hasPreviousPage,
    isLoading: isFeaturesLoading,
  } = useFeatures({
    dictionary: dictionaryName,
    concernedFeatureIndex: featureIndex,
  })

  const features = featuresData?.pages.flatMap((page) => page) ?? []

  return (
    <div className="flex h-full w-full overflow-hidden bg-slate-50/50">
      <div className="w-[350px] shrink-0 border-r border-slate-200 bg-white flex flex-col h-full overflow-hidden">
        <FeatureList
          features={features}
          selectedIndex={featureIndex}
          onSelectFeature={(idx) => navigateToFeature(dictionaryName, idx)}
          onLoadMore={() => fetchNextPage()}
          hasNextPage={hasNextPage}
          onLoadPrevious={() => fetchPreviousPage()}
          hasPreviousPage={hasPreviousPage}
          isLoading={isFeaturesLoading}
          className="overflow-y-auto grow no-scrollbar"
        />
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-6">
          <div className="w-full flex justify-center items-center relative h-12">
            <div className="flex justify-center items-center gap-3">
              <div className="flex items-center -space-x-px">
                <Button
                  className="h-12 rounded-r-none px-3 relative focus:z-10"
                  onClick={() =>
                    navigateToFeature(selectedDictionary, featureIndex - 1)
                  }
                  disabled={featureIndex <= 0}
                >
                  <div className="flex flex-col items-center gap-0.5">
                    <ChevronLeft className="h-4 w-4" />
                    <span className="text-[10px] font-bold">PREV</span>
                  </div>
                </Button>
                <Button
                  className="h-12 rounded-l-none px-3 relative focus:z-10"
                  onClick={() =>
                    navigateToFeature(selectedDictionary, featureIndex + 1)
                  }
                >
                  <div className="flex flex-col items-center gap-0.5">
                    <ChevronRight className="h-4 w-4" />
                    <span className="text-[10px] font-bold">NEXT</span>
                  </div>
                </Button>
              </div>
              <div className="w-[300px]">
                <LabeledSelect
                  label="Dictionary"
                  placeholder="Select a dictionary"
                  value={selectedDictionary}
                  onValueChange={setSelectedDictionary}
                  options={dictionaries.map((d) => ({ value: d, label: d }))}
                  triggerClassName="bg-white w-full"
                />
              </div>
              <div className="w-[100px]">
                <LabeledInput
                  label="Index"
                  id="feature-input"
                  value={selectedFeatureIndex}
                  onChange={(e) => setSelectedFeatureIndex(e.target.value)}
                />
              </div>
              <Button
                onClick={() =>
                  navigate({
                    to: '/dictionaries/$dictionaryName/features/$featureIndex',
                    params: {
                      dictionaryName: selectedDictionary,
                      featureIndex: selectedFeatureIndex,
                    },
                  })
                }
                className="h-12 px-4"
                disabled={isNaN(Number(selectedFeatureIndex))}
              >
                Go
              </Button>
            </div>
            <FeatureBookmarkButton
              feature={feature}
              className="rounded-full h-12 w-12 absolute right-0 flex items-center justify-center"
            />
          </div>

          <div className="flex gap-12 w-full justify-center">
            <div className="flex flex-col sm:mb-0 sm:basis-1/2 lg:basis-1/3 min-w-0 gap-4">
              <FeatureInterpretation
                feature={feature}
                dictionaryName={dictionaryName}
                featureIndex={featureIndex}
              />
              <InferenceCard
                key={`inference-card-${dictionaryName}-${featureIndex}`}
                dictionaryName={dictionaryName}
                featureIndex={featureIndex}
                maxFeatureAct={feature.maxFeatureAct}
              />
            </div>
            <div className="flex flex-col px-0 sm:basis-1/2 lg:basis-2/3 min-w-0">
              <FeatureCard feature={feature} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
