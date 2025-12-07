import { memo } from 'react'
import { Info } from '../ui/info'
import { FeatureSampleGroup } from './sample'
import type { Feature } from '@/types/feature'
import { Card, CardContent } from '@/components/ui/card'

type FeatureCardProps = {
  feature: Feature
}

export const FeatureCard = memo(({ feature }: FeatureCardProps) => {
  return (
    <Card className="w-full">
      <CardContent>
        <div className="flex flex-col gap-2 pt-6">
          <div className="flex gap-6">
            <div className="flex flex-col basis-1/2 min-w-0 gap-4">
              {feature.logits && (
                <div className="flex flex-col w-full gap-4">
                  <div className="flex gap-4">
                    <div className="flex flex-col w-1/2 gap-2">
                      <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default">
                        <span>NEGATIVE LOGITS</span>
                        <Info iconSize={14}>
                          The top negative logits of the feature.
                        </Info>
                      </div>
                      {feature.logits.topNegative.map((token) => (
                        <div
                          key={token.token}
                          className="flex gap-2 items-center justify-between"
                        >
                          <span className="bg-blue-200 text-slate-700 text-xs px-2 py-0.5 rounded font-medium whitespace-nowrap overflow-hidden text-ellipsis">
                            {token.token
                              .replaceAll('\n', '↵')
                              .replaceAll('\t', '→')
                              .replaceAll(' ', '_')}
                          </span>
                          <span className="text-slate-500 text-sm">
                            {token.logit.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="flex flex-col w-1/2 gap-2">
                      <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default">
                        <span>POSITIVE LOGITS</span>
                        <Info iconSize={14}>
                          The top positive logits of the feature.
                        </Info>
                      </div>
                      {feature.logits.topPositive.map((token) => (
                        <div
                          key={token.token}
                          className="flex gap-2 items-center justify-between"
                        >
                          <span className="bg-red-200 text-slate-700 text-xs px-2 py-0.5 rounded font-medium whitespace-nowrap overflow-hidden text-ellipsis">
                            {token.token
                              .replaceAll('\n', '↵')
                              .replaceAll('\t', '→')
                              .replaceAll(' ', '_')}
                          </span>
                          <span className="text-slate-500 text-sm">
                            {token.logit.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="flex flex-col basis-1/2 min-w-0 gap-4">
              <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default justify-center">
                ACTIVATION TIMES{' '}
                {((feature.actTimes / feature.nAnalyzedTokens!) * 100).toFixed(
                  4,
                )}
                %
                <Info iconSize={14}>
                  The percentage of tokens that the feature is activated on.
                  Activated {feature.actTimes.toLocaleString()} of{' '}
                  {feature.nAnalyzedTokens!.toLocaleString()} tokens.
                </Info>
              </div>
              <div className="flex flex-col items-center gap-4 grow">
                <div className="w-full basis-1/2 min-h-0">
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 mb-2 border border-dashed border-slate-200">
                    Activation histogram is not available.
                  </div>
                </div>
                <div className="w-full basis-1/2 min-h-0">
                  <div className="h-full flex items-center justify-center text-slate-500 text-sm bg-slate-100 rounded-2xl p-3 border border-dashed border-slate-200">
                    Logits histogram is not available.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col w-full gap-4">
            <FeatureSampleGroup
              feature={feature}
              sampleGroup={feature.sampleGroups[0]}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
})
FeatureCard.displayName = 'FeatureCard'
