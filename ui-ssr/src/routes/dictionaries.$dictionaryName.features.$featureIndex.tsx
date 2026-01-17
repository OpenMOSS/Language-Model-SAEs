import { Link, createFileRoute, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { z } from 'zod'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { parseWithPrettify } from '@/utils/zod'
import { Button } from '@/components/ui/button'
import { FeatureBookmarkButton } from '@/components/feature/bookmark-button'
import { FeatureCard } from '@/components/feature/feature-card'
import { FeatureInterpretation } from '@/components/feature/feature-interpretation'
import { LabeledInput } from '@/components/ui/labeled-input'
import { DictionarySelect } from '@/components/dictionary/dictionary-select'
import { InferenceCard } from '@/components/feature/inference-card'
import { IframeIntegrationCard } from '@/components/feature/iframe-integration-card'
import { fetchAdminSaes } from '@/api/admin'

import {
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
  validateSearch: (search) => parseWithPrettify(searchParamsSchema, search),
  staticData: {
    fullScreen: true,
  },
  component: FeaturesPage,
  loader: async ({ context, params }) => {
    const [{ saes }, samplings, feature] = await Promise.all([
      await context.queryClient.ensureQueryData({
        queryKey: ['admin', 'saes', { limit: 1000 }],
        queryFn: () => fetchAdminSaes({ data: { limit: 1000 } }),
      }),
      await context.queryClient.ensureQueryData(
        samplingsQueryOptions({
          dictionary: params.dictionaryName,
          featureIndex: Number(params.featureIndex),
        }),
      ),
      await context.queryClient.ensureQueryData(
        featureQueryOptions({
          dictionary: params.dictionaryName,
          featureIndex: Number(params.featureIndex),
        }),
      ),
    ])
    return {
      saes,
      samplings,
      feature,
      dictionaryName: params.dictionaryName,
      featureIndex: Number(params.featureIndex),
    }
  },
})

function FeaturesPage() {
  const navigate = useNavigate()

  const { saes, feature, dictionaryName, featureIndex } = Route.useLoaderData()

  const [selectedDictionary, setSelectedDictionary] =
    useState<string>(dictionaryName)

  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState<string>(
    featureIndex.toString(),
  )

  useEffect(() => {
    setSelectedFeatureIndex(featureIndex.toString())
  }, [featureIndex])

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
      <div
        className={`shrink-0 border-r border-slate-200 bg-white flex flex-col h-full overflow-hidden transition-all duration-300 ease-in-out ${
          isFeaturesLoading
            ? '-translate-x-full opacity-0 w-0 border-r-0'
            : 'w-[350px] translate-x-0 opacity-100'
        }`}
      >
        <FeatureList
          features={features}
          selectedIndex={featureIndex}
          onSelectFeature={async (idx) =>
            await navigate({
              to: '/dictionaries/$dictionaryName/features/$featureIndex',
              params: {
                dictionaryName: selectedDictionary,
                featureIndex: idx.toString(),
              },
            })
          }
          onLoadMore={() => fetchNextPage()}
          hasNextPage={hasNextPage}
          onLoadPrevious={() => fetchPreviousPage()}
          hasPreviousPage={hasPreviousPage}
          isLoading={isFeaturesLoading}
          className="overflow-y-auto grow no-scrollbar"
        />
      </div>

      <div className="flex-1 overflow-y-auto transition-all duration-300 ease-in-out">
        <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-6">
          <div className="w-full flex justify-center items-center relative h-12">
            <div className="flex justify-center items-center gap-3">
              <div className="flex items-center -space-x-px">
                {featureIndex > 0 ? (
                  <Link
                    to="/dictionaries/$dictionaryName/features/$featureIndex"
                    params={{
                      dictionaryName: selectedDictionary,
                      featureIndex: (featureIndex - 1).toString(),
                    }}
                  >
                    <Button className="h-12 rounded-r-none px-3 relative focus:z-10">
                      <div className="flex flex-col items-center gap-0.5">
                        <ChevronLeft className="h-4 w-4" />
                        <span className="text-[10px] font-bold">PREV</span>
                      </div>
                    </Button>
                  </Link>
                ) : (
                  <Button
                    className="h-12 rounded-r-none px-3 relative focus:z-10"
                    disabled
                  >
                    <div className="flex flex-col items-center gap-0.5">
                      <ChevronLeft className="h-4 w-4" />
                      <span className="text-[10px] font-bold">PREV</span>
                    </div>
                  </Button>
                )}
                <Link
                  to="/dictionaries/$dictionaryName/features/$featureIndex"
                  params={{
                    dictionaryName: selectedDictionary,
                    featureIndex: (featureIndex + 1).toString(),
                  }}
                >
                  <Button className="h-12 rounded-l-none px-3 relative focus:z-10">
                    <div className="flex flex-col items-center gap-0.5">
                      <ChevronRight className="h-4 w-4" />
                      <span className="text-[10px] font-bold">NEXT</span>
                    </div>
                  </Button>
                </Link>
              </div>
              <div>
                <DictionarySelect
                  saes={saes}
                  selectedDictionary={selectedDictionary}
                  onSelect={setSelectedDictionary}
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
              {!isNaN(Number(selectedFeatureIndex)) ? (
                <Link
                  to="/dictionaries/$dictionaryName/features/$featureIndex"
                  params={{
                    dictionaryName: selectedDictionary,
                    featureIndex: selectedFeatureIndex,
                  }}
                >
                  <Button className="h-12 px-4">Go</Button>
                </Link>
              ) : (
                <Button className="h-12 px-4" disabled>
                  Go
                </Button>
              )}
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
              <IframeIntegrationCard
                dictionaryName={dictionaryName}
                featureIndex={featureIndex}
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
