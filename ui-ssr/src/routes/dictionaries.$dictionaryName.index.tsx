import { createFileRoute } from '@tanstack/react-router'
import { dictionariesQueryOptions } from '@/hooks/useFeatures'
import { DictionaryCard } from '@/components/dictionary/dictionary-card'

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
  const { dictionaryName } = Route.useLoaderData()
  return (
    <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-6">
      <DictionaryCard dictionaryName={dictionaryName} />
    </div>
  )
}
