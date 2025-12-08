import {
  queryOptions,
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query'
import type { Feature } from '@/types/feature'
import {
  fetchDictionaries,
  fetchFeature,
  fetchSamples,
  fetchSamplings,
  toggleBookmark,
} from '@/api/features'

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

export const samplingsQueryOptions = (params: {
  dictionary: string
  featureIndex: number
}) =>
  queryOptions({
    queryKey: ['samplings', params.dictionary, params.featureIndex],
    queryFn: () => fetchSamplings({ data: params }),
  })

export const useSamples = (params: {
  dictionary: string
  featureIndex: number
  samplingName: string
  totalLength: number
}) =>
  useInfiniteQuery({
    queryKey: [
      'samples',
      params.dictionary,
      params.featureIndex,
      params.samplingName,
    ],
    queryFn: ({ pageParam = 0 }) =>
      fetchSamples({ data: { ...params, start: pageParam, length: 5 } }),
    getNextPageParam: (_, allPages) =>
      allPages.reduce((acc, page) => acc + page.length, 0) < params.totalLength
        ? allPages.reduce((acc, page) => acc + page.length, 0)
        : undefined,
    initialPageParam: 0,
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
