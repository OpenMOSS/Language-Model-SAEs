import { useState } from 'react'
import { Bookmark } from 'lucide-react'
import type { Feature } from '@/types/feature'
import { ToggleButton } from '@/components/ui/toggle-button'
import { useToggleBookmark } from '@/hooks/useFeatures'
import { cn } from '@/lib/utils'

type FeatureBookmarkButtonProps = {
  feature: Feature
  size?: 'default' | 'sm' | 'lg' | 'icon'
  className?: string
}

export const FeatureBookmarkButton = ({
  feature,
  size = 'default',
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
    <ToggleButton
      pressed={isBookmarked}
      onPressedChange={handleToggle}
      disabled={toggleMutation.isPending}
      size={size}
      className={cn(
        toggleMutation.isPending && 'opacity-50',
        toggleMutation.isError && 'border-red-500 hover:border-red-600',
        className,
      )}
    >
      <Bookmark className={isBookmarked ? 'fill-current' : ''} size={16} />
    </ToggleButton>
  )
}
