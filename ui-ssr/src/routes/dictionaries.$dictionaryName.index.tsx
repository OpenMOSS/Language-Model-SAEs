import { Link, createFileRoute } from '@tanstack/react-router'
import { useState } from 'react'
import { DictionaryCard } from '@/components/dictionary/dictionary-card'
import { Button } from '@/components/ui/button'
import { fetchAdminSaes } from '@/api/admin'
import { DictionarySelect } from '@/components/dictionary/dictionary-select'
import { LabeledInput } from '@/components/ui/labeled-input'

export const Route = createFileRoute('/dictionaries/$dictionaryName/')({
  component: DictionaryIndexPage,
  staticData: {
    fullScreen: false,
  },
  loader: async ({ context, params }) => {
    const { saes } = await context.queryClient.ensureQueryData({
      queryKey: ['admin', 'saes', { limit: 1000 }],
      queryFn: () => fetchAdminSaes({ data: { limit: 1000 } }),
    })
    return { saes, dictionaryName: params.dictionaryName }
  },
})

function DictionaryIndexPage() {
  const { saes, dictionaryName } = Route.useLoaderData()

  const [selectedDictionary, setSelectedDictionary] = useState(dictionaryName)
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState('0')

  return (
    <div className="h-full overflow-y-auto pt-4 pb-20 px-20 flex flex-col items-center gap-6">
      <div className="w-full flex justify-center items-center relative h-12">
        <div className="flex flex-wrap justify-center items-center gap-3">
          <DictionarySelect
            saes={saes}
            selectedDictionary={selectedDictionary}
            onSelect={setSelectedDictionary}
          />

          {selectedDictionary ? (
            <Link
              to="/dictionaries/$dictionaryName"
              params={{ dictionaryName: selectedDictionary }}
            >
              <Button className="h-12 px-4">Go</Button>
            </Link>
          ) : (
            <Button className="h-12 px-4" disabled>
              Go
            </Button>
          )}

          <div className="w-[100px] ml-4">
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
              <Button className="h-12 px-4">View Feature</Button>
            </Link>
          ) : (
            <Button className="h-12 px-4" disabled>
              View Feature
            </Button>
          )}
        </div>
      </div>
      <DictionaryCard dictionaryName={dictionaryName} />
    </div>
  )
}
