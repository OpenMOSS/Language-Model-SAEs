import { Link, createFileRoute } from '@tanstack/react-router'
import { useState } from 'react'
import { dictionariesQueryOptions } from '@/hooks/useFeatures'
import { DictionaryCard } from '@/components/dictionary/dictionary-card'
import { LabeledSelect } from '@/components/ui/labeled-select'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/dictionaries/$dictionaryName/')({
  component: DictionaryIndexPage,
  loader: async ({ context, params }) => {
    const dictionaries = await context.queryClient.ensureQueryData(
      dictionariesQueryOptions(),
    )
    return { dictionaries, dictionaryName: params.dictionaryName }
  },
})

function DictionaryIndexPage() {
  const { dictionaries, dictionaryName } = Route.useLoaderData()

  const [selectedDictionary, setSelectedDictionary] = useState(dictionaryName)

  return (
    <div className="h-full overflow-y-auto pt-4 pb-20 px-20 flex flex-col items-center gap-6">
      <div className="w-full flex justify-center items-center relative h-12">
        <div className="flex justify-center items-center gap-3">
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
        </div>
      </div>
      <DictionaryCard dictionaryName={dictionaryName} />
    </div>
  )
}
