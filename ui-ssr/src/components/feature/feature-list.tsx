import { memo, useEffect, useRef } from 'react'
import { Layers } from 'lucide-react'
import { Spinner } from '../ui/spinner'
import { IntersectionObserver } from '../ui/intersection-observer'
import { FeatureCardCompact } from './feature-card-compact'
import type { FeatureCompact } from '@/types/feature'
import { cn } from '@/lib/utils'

export const FeatureListItem = memo(
  ({
    feature,
    isSelected,
    onClick,
  }: {
    feature: FeatureCompact
    isSelected: boolean
    onClick: () => void
  }) => {
    const ref = useRef<HTMLButtonElement>(null)
    const isFirstRender = useRef(true)

    useEffect(() => {
      if (isSelected && ref.current) {
        if (isFirstRender.current) {
          ref.current.scrollIntoView({ block: 'center', behavior: 'auto' })
        } else {
          ref.current.scrollIntoView({ block: 'center', behavior: 'smooth' })
        }
      }
      isFirstRender.current = false
    }, [isSelected])

    return (
      <button
        ref={ref}
        type="button"
        onClick={onClick}
        className={cn(
          'w-full text-left transition-all duration-150 cursor-pointer relative',
          'hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-700/50',
          'border-b border-slate-3 last:border-b-0',
          isSelected &&
            'bg-slate-100 hover:bg-slate-100 inset-ring-2 inset-ring-sky-600 z-10',
        )}
      >
        <FeatureCardCompact
          feature={feature}
          className={cn('pointer-events-none')}
        />
      </button>
    )
  },
)

FeatureListItem.displayName = 'FeatureListItem'

export const FeatureList = memo(
  ({
    features,
    selectedIndex,
    onSelectFeature,
    className,
    onLoadMore,
    hasNextPage,
    onLoadPrevious,
    hasPreviousPage,
    isLoading = false,
  }: {
    features: FeatureCompact[]
    selectedIndex: number | null
    onSelectFeature: (index: number) => void
    className?: string
    onLoadMore: () => void
    hasNextPage: boolean
    onLoadPrevious?: () => void
    hasPreviousPage?: boolean
    isLoading?: boolean
  }) => {
    return (
      <div className={cn('flex flex-col', className)}>
        {hasPreviousPage && onLoadPrevious && (
          <IntersectionObserver
            onIntersect={onLoadPrevious}
            enabled={hasPreviousPage}
            className="h-8 w-full flex items-center justify-center p-1"
          >
            <Spinner isAnimating={true} className="text-slate-400" />
          </IntersectionObserver>
        )}
        {features.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Layers className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">No features found</p>
          </div>
        ) : (
          <>
            {features.map((feature) => (
              <FeatureListItem
                key={feature.featureIndex}
                feature={feature}
                isSelected={selectedIndex === feature.featureIndex}
                onClick={() => onSelectFeature(feature.featureIndex)}
              />
            ))}
            <IntersectionObserver
              onIntersect={onLoadMore}
              enabled={hasNextPage}
              className="h-8 w-full flex items-center justify-center p-1"
            >
              <Spinner isAnimating={true} className="text-slate-400" />
            </IntersectionObserver>
          </>
        )}
      </div>
    )
  },
)

FeatureList.displayName = 'FeatureList'
