import { useState } from 'react'
import { Bookmark } from 'lucide-react'
import type { Feature } from '@/types/feature'
import { useToggleBookmark } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

type FeatureBookmarkButtonProps = {
  feature: Feature
  className?: string
}

export const FeatureBookmarkButton = ({
  feature,
  className,
}: FeatureBookmarkButtonProps) => {
  const [isBookmarked, setIsBookmarked] = useState<boolean>(
    feature.isBookmarked || false,
  )
  const toggleMutation = useToggleBookmark()

  const handleToggle = () => {
    toggleMutation.mutate(
      {
        data: {
          dictionaryName: feature.dictionaryName,
          featureIndex: feature.featureIndex,
          isBookmarked: isBookmarked,
        },
      },
      {
        onSuccess: (newStatus) => {
          setIsBookmarked(newStatus)
        },
      },
    )
  }

  return (
    <button
      type="button"
      onClick={handleToggle}
      disabled={toggleMutation.isPending}
      className={cn(
        'p-1.5 rounded transition-colors cursor-pointer',
        'hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-1',
        toggleMutation.isPending && 'opacity-50 cursor-not-allowed',
        toggleMutation.isError && 'text-red-500',
        isBookmarked ? 'text-amber-500' : 'text-slate-400',
        className,
      )}
      aria-label={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
    >
      <Bookmark
        className={cn(
          'h-3.5 w-3.5 transition-all',
          isBookmarked && 'fill-current',
        )}
      />
    </button>
  )
}
