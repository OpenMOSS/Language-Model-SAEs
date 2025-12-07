import { createFileRoute, redirect } from '@tanstack/react-router'
import { dictionariesQueryOptions } from '@/hooks/useFeatures'

export const Route = createFileRoute('/dictionaries/')({
  beforeLoad: async ({ context }) => {
    const dictionaries = await context.queryClient.ensureQueryData(
      dictionariesQueryOptions(),
    )
    throw redirect({
      to: '/dictionaries/$dictionaryName/features/$featureIndex',
      params: {
        dictionaryName: dictionaries[0],
        featureIndex: '0',
      },
    })
  },
})
