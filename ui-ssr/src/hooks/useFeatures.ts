import {
  queryOptions,
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import type { Feature } from '@/types/feature'
import {
  fetchDictionaries,
  fetchFeature,
  fetchFeatures,
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

export const useFeatures = (params: {
  dictionary: string
  concernedFeatureIndex: number
}) => {
  const [concernedFeatureIndex, setConcernedFeatureIndex] = useState(
    params.concernedFeatureIndex,
  )
  const queryStart = Math.max(0, concernedFeatureIndex - 12)
  const queryEnd = queryStart + 25
  const query = useInfiniteQuery({
    queryKey: ['features', params.dictionary, concernedFeatureIndex],
    queryFn: ({
      pageParam: { start, end },
    }: {
      pageParam: { start: number; end: number }
    }) => {
      return fetchFeatures({
        data: {
          dictionary: params.dictionary,
          start,
          end,
        },
      })
    },
    getNextPageParam: (lastPage) => {
      if (lastPage.length === 0) return undefined
      const lastFeature = lastPage[lastPage.length - 1]
      const nextStart = lastFeature.featureIndex + 1
      return { start: nextStart, end: nextStart + 25 }
    },
    getPreviousPageParam: (firstPage) => {
      if (firstPage.length === 0) return undefined
      const firstFeature = firstPage[0]
      const currentStart = firstFeature.featureIndex
      if (currentStart <= 0) return undefined

      const prevEnd = currentStart
      const prevStart = Math.max(0, prevEnd - 25)
      return { start: prevStart, end: prevEnd }
    },
    initialPageParam: {
      start: queryStart,
      end: queryEnd,
    },
  })

  const features = query.data?.pages.flatMap((page) => page) ?? []

  useEffect(() => {
    const queryHasConcernedFeature = features.some(
      (feature) => feature.featureIndex == params.concernedFeatureIndex,
    )
    const matchState = concernedFeatureIndex === params.concernedFeatureIndex
    if (!queryHasConcernedFeature && !matchState) {
      setConcernedFeatureIndex(params.concernedFeatureIndex)
    }
  }, [params.concernedFeatureIndex])

  return query
}

export const samplesQueryOptions = (params: {
  dictionary: string
  featureIndex: number
  samplingName: string
  length: number
}) =>
  queryOptions({
    queryKey: [
      'samples',
      params.dictionary,
      params.featureIndex,
      params.samplingName,
      params.length,
    ],
    queryFn: () => fetchSamples({ data: { ...params, start: 0 } }),
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
