import { createFileRoute, redirect } from '@tanstack/react-router'
import { dictionariesQueryOptions } from '@/hooks/useFeatures'

export const Route = createFileRoute('/dictionaries/')({
  beforeLoad: async ({ context }) => {
    const dictionaries = await context.queryClient.ensureQueryData(
      dictionariesQueryOptions(),
    )
    throw redirect({
      to: '/dictionaries/$dictionaryName',
      params: {
        dictionaryName: dictionaries[0],
      },
    })
  },
})
