import { decode } from '@msgpack/msgpack'
import camelcaseKeys from 'camelcase-keys'
import { z } from 'zod'
import { createServerFn } from '@tanstack/react-start'
import { FeatureSampleCompactSchema, FeatureSchema } from '@/types/feature'

export const fetchDictionaries = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/dictionaries`)
    const data = await response.json()
    return z.array(z.string()).parse(data)
  },
)

export const fetchMetrics = createServerFn({ method: 'GET' })
  .inputValidator((data: { dictionary: string }) => data)
  .handler(async ({ data: { dictionary } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/dictionaries/${dictionary}/metrics`,
    )
    const data = await response.json()
    return z.object({ metrics: z.array(z.string()) }).parse(data).metrics
  })

export type MetricFilters = Record<string, { min?: number; max?: number }>

const buildMetricFiltersParam = (
  metricFilters?: MetricFilters,
): string | null => {
  if (!metricFilters) return null

  const mongoFilters: Record<string, Record<string, number>> = {}

  for (const [metricName, filter] of Object.entries(metricFilters)) {
    const mongoFilter: Record<string, number> = {}
    if (filter.min !== undefined) {
      mongoFilter['$gte'] = filter.min
    }
    if (filter.max !== undefined) {
      mongoFilter['$lte'] = filter.max
    }

    if (Object.keys(mongoFilter).length > 0) {
      mongoFilters[metricName] = mongoFilter
    }
  }

  if (Object.keys(mongoFilters).length === 0) return null
  return JSON.stringify(mongoFilters)
}

export const fetchFeature = createServerFn({ method: 'GET' })
  .inputValidator((data: { dictionary: string; featureIndex: number }) => data)
  .handler(async ({ data: { dictionary, featureIndex } }) => {
    const url = `${process.env.BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/x-msgpack',
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch feature: ${await response.text()}`)
    }

    const arrayBuffer = await response.arrayBuffer()
    const decoded = decode(new Uint8Array(arrayBuffer)) as any
    const camelCased = camelcaseKeys(decoded, {
      deep: true,
      stopPaths: ['sample_groups.samples.context'],
    })

    return FeatureSchema.parse(camelCased)
  })

export const countFeatures = createServerFn({ method: 'GET' })
  .inputValidator(
    (data: {
      dictionary: string
      analysisName?: string | null
      metricFilters?: MetricFilters
    }) => data,
  )
  .handler(async ({ data: { dictionary, analysisName, metricFilters } }) => {
    const queryParams = new URLSearchParams()
    if (analysisName) {
      queryParams.append('feature_analysis_name', analysisName)
    }

    const filtersParam = buildMetricFiltersParam(metricFilters)
    if (filtersParam) {
      queryParams.append('metric_filters', filtersParam)
    }

    const queryString = queryParams.toString()
    const url = `${process.env.BACKEND_URL}/dictionaries/${dictionary}/features/count${queryString ? `?${queryString}` : ''}`

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Failed to count features: ${await response.text()}`)
    }

    const data = await response.json()
    return z.object({ count: z.number() }).parse(data).count
  })

export const submitCustomInput = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: {
      dictionaryName: string
      featureIndex: number
      inputText: string
    }) => data,
  )
  .handler(async ({ data: { dictionaryName, featureIndex, inputText } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/dictionaries/${dictionaryName}/features/${featureIndex}/infer?text=${encodeURIComponent(inputText)}`,
      {
        method: 'POST',
        headers: {
          Accept: 'application/x-msgpack',
        },
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to submit custom input: ${await response.text()}`)
    }

    const arrayBuffer = await response.arrayBuffer()
    const decoded = decode(new Uint8Array(arrayBuffer)) as any
    const camelCased = camelcaseKeys(decoded, {
      deep: true,
      stopPaths: ['context'],
    })

    return FeatureSampleCompactSchema.parse(camelCased)
  })

export const toggleBookmark = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: {
      dictionaryName: string
      featureIndex: number
      isBookmarked: boolean
    }) => data,
  )
  .handler(async ({ data: { dictionaryName, featureIndex, isBookmarked } }) => {
    const method = isBookmarked ? 'DELETE' : 'POST'

    const response = await fetch(
      `${process.env.BACKEND_URL}/dictionaries/${dictionaryName}/features/${featureIndex}/bookmark`,
      { method },
    )

    if (!response.ok) {
      throw new Error(`Failed to toggle bookmark: ${await response.text()}`)
    }

    return !isBookmarked
  })

export const getImageUrl = createServerFn({ method: 'GET' })
  .inputValidator((data: { imagePath: string }) => data)
  .handler(({ data: { imagePath } }) => {
    return `${process.env.BACKEND_URL}${imagePath}`
  })
