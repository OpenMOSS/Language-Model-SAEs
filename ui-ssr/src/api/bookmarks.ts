import { queryOptions } from '@tanstack/react-query'
import { createServerFn } from '@tanstack/react-start'
import camelcaseKeys from 'camelcase-keys'
import type { z } from 'zod'
import { FeatureCompactSchema } from '@/types/feature'
import { parseWithPrettify } from '@/utils/zod'

export interface Bookmark {
  saeName: string
  featureIndex: number
  createdAt: string
  tags: string[]
  notes: string | null
}

export interface BookmarkWithFeature extends Bookmark {
  feature: z.infer<typeof FeatureCompactSchema>
}

export const fetchBookmarks = createServerFn({ method: 'GET' })
  .inputValidator(
    (data: { saeName?: string; limit?: number; skip?: number }) => data,
  )
  .handler(async ({ data: { saeName, limit = 25, skip = 0 } }) => {
    const params = new URLSearchParams({
      limit: String(limit),
      skip: String(skip),
      include_features: 'true',
    })
    if (saeName) {
      params.append('sae_name', saeName)
    }
    const response = await fetch(
      `${process.env.BACKEND_URL}/bookmarks?${params}`,
    )
    if (!response.ok) {
      throw new Error(`Failed to fetch bookmarks: ${await response.text()}`)
    }
    const data = await response.json()
    const camelCased = camelcaseKeys(data, {
      deep: true,
      stopPaths: [
        'bookmarks.feature.samples.feature_acts_indices',
        'bookmarks.feature.samples.feature_acts_values',
        'bookmarks.feature.samples.z_pattern_indices',
        'bookmarks.feature.samples.z_pattern_values',
      ],
    })

    // Validate and parse feature data if present
    const bookmarks = camelCased.bookmarks.map((bookmark: any) => {
      if (bookmark.feature) {
        bookmark.feature = parseWithPrettify(
          FeatureCompactSchema,
          bookmark.feature,
        )
      }
      return bookmark
    })

    return {
      bookmarks: bookmarks as BookmarkWithFeature[],
      totalCount: camelCased.totalCount as number,
    }
  })

export const bookmarksQueryOptions = (limit = 25, skip = 0, saeName?: string) =>
  queryOptions({
    queryKey: ['bookmarks', { limit, skip, saeName }],
    queryFn: () => fetchBookmarks({ data: { limit, skip, saeName } }),
  })
