import {
  queryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import type { Feature } from '@/types/feature'
import { fetchDictionaries, fetchFeature, toggleBookmark } from '@/api/features'

export const dictionariesQueryOptions = () =>
  queryOptions({
    queryKey: ['dictionaries'],
    queryFn: fetchDictionaries,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

export const featureQueryOptions = (params: {
  dictionary: string
  featureIndex: number
}) =>
  queryOptions({
    queryKey: ['feature', params.dictionary, params.featureIndex],
    queryFn: () =>
      fetchFeature({
        data: params,
      }),
  })

export function useToggleBookmark() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: toggleBookmark,
    onSuccess: (
      newBookmarkStatus,
      { data: { dictionaryName, featureIndex } },
    ) => {
      // Optimistically update the feature cache
      queryClient.setQueriesData(
        { queryKey: ['feature', dictionaryName, featureIndex] },
        (oldData: Feature) => {
          return { ...oldData, isBookmarked: newBookmarkStatus }
        },
      )
    },
  })
}
