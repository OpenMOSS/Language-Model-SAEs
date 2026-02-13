import { memo } from 'react'
import { Info } from '../ui/info'
import { cn } from '@/lib/utils'

type FeatureLogitsProps = {
  logits: {
    topPositive: {
      logit: number
      token: string
    }[]
    topNegative: {
      logit: number
      token: string
    }[]
  }
  className?: string
}

export const FeatureLogits = memo(
  ({ logits, className }: FeatureLogitsProps) => {
    return (
      <div className={cn('flex flex-col basis-1/2 min-w-0 gap-4', className)}>
        <div className="flex flex-col w-full gap-4">
          <div className="flex gap-4">
            <div className="flex flex-col w-1/2 gap-2">
              <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default">
                <span>NEGATIVE LOGITS</span>
                <Info iconSize={14}>
                  The top negative logits of the feature.
                </Info>
              </div>
              {logits.topNegative.map((token, index) => (
                <div
                  key={`negative-${index}`}
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
              {logits.topPositive.map((token, index) => (
                <div
                  key={`positive-${index}`}
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
      </div>
    )
  },
)
FeatureLogits.displayName = 'FeatureLogits'

export const FeatureLogitsHorizontal = memo(
  ({ logits }: FeatureLogitsProps) => {
    return (
      <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center">
        <span className="font-semibold tracking-tight text-sm text-slate-700 uppercase">
          Positive
        </span>
        <div className="flex flex-wrap gap-2">
          {logits.topPositive.map((token, index) => (
            <span
              key={`positive-${index}`}
              className="bg-red-200 text-slate-700 text-xs px-2 py-0.5 rounded font-medium whitespace-nowrap overflow-hidden text-ellipsis"
            >
              {token.token
                .replaceAll('\n', '↵')
                .replaceAll('\t', '→')
                .replaceAll(' ', '_')}
            </span>
          ))}
        </div>

        <span className="font-semibold tracking-tight text-sm text-slate-700 uppercase">
          Negative
        </span>
        <div className="flex flex-wrap gap-2">
          {logits.topNegative.map((token, index) => (
            <span
              key={`negative-${index}`}
              className="bg-blue-200 text-slate-700 text-xs px-2 py-0.5 rounded font-medium whitespace-nowrap overflow-hidden text-ellipsis"
            >
              {token.token
                .replaceAll('\n', '↵')
                .replaceAll('\t', '→')
                .replaceAll(' ', '_')}
            </span>
          ))}
        </div>
      </div>
    )
  },
)
FeatureLogitsHorizontal.displayName = 'FeatureLogitsHorizontal'
