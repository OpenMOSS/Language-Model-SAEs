import { useNProgress } from '@tanem/react-nprogress'
import { Loader2 } from 'lucide-react'

import { cn } from '@/lib/utils'

interface SpinnerProps {
  isAnimating: boolean
  className?: string
}

export function Spinner({ isAnimating, className }: SpinnerProps) {
  const { animationDuration, isFinished } = useNProgress({
    isAnimating,
  })

  if (isFinished) return null

  return (
    <div
      className={cn(
        'pointer-events-none transition-opacity ease-linear',
        className,
      )}
      style={{
        opacity: isFinished ? 0 : 1,
        transitionDuration: `${animationDuration}ms`,
      }}
    >
      <Loader2 className="animate-spin" />
    </div>
  )
}

