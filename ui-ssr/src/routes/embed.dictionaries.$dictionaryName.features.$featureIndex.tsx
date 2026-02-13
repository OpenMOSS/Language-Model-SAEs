import { createFileRoute } from '@tanstack/react-router'
import { z } from 'zod'
import type { FeatureCompact } from '@/types/feature'
import { FeatureCardCompactForEmbed } from '@/components/feature/feature-card'
import { featureQueryOptions, samplesQueryOptions } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

export const Route = createFileRoute(
  '/embed/dictionaries/$dictionaryName/features/$featureIndex',
)({
  staticData: {
    embed: true,
  },
  validateSearch: (search) =>
    z
      .object({
        plain: z.boolean().optional().catch(false),
        visibleRange: z.coerce.number().optional(),
      })
      .parse(search),
  component: EmbedFeaturePage,
  loader: async ({ context, params }) => {
    const [feature, samples] = await Promise.all([
      await context.queryClient.ensureQueryData(
        featureQueryOptions({
          dictionary: params.dictionaryName,
          featureIndex: Number(params.featureIndex),
        }),
      ),
      await context.queryClient.ensureQueryData(
        samplesQueryOptions({
          dictionary: params.dictionaryName,
          featureIndex: Number(params.featureIndex),
          samplingName: 'top_activations',
          length: 5,
        }),
      ),
    ])
    return {
      feature: { ...feature, samples } as FeatureCompact,
    }
  },
})

function EmbedFeaturePage() {
  const { feature } = Route.useLoaderData()
  const { plain, visibleRange } = Route.useSearch()

  return (
    <div
      className={cn(
        'min-h-screen',
        !plain && 'bg-linear-to-b from-slate-50 to-slate-100/50 p-4 sm:p-6',
      )}
    >
      <div className={cn('max-w-3xl mx-auto', plain && 'max-w-none')}>
        <FeatureCardCompactForEmbed
          feature={feature}
          plain={plain}
          defaultVisibleRange={visibleRange}
        />
      </div>
    </div>
  )
}
