import { createFileRoute } from '@tanstack/react-router'
import type { FeatureCompact } from '@/types/feature'
import { FeatureCardCompactForEmbed } from '@/components/feature/feature-card'
import { featureQueryOptions, samplesQueryOptions } from '@/hooks/useFeatures'

export const Route = createFileRoute(
  '/embed/dictionaries/$dictionaryName/features/$featureIndex',
)({
  staticData: {
    embed: true,
  },
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

  return (
    <div className="min-h-screen bg-linear-to-b from-slate-50 to-slate-100/50 p-4 sm:p-6">
      <div className="max-w-3xl mx-auto">
        <FeatureCardCompactForEmbed feature={feature} />
      </div>
    </div>
  )
}
