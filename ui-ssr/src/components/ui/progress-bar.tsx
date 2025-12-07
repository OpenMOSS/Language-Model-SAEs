import { useNProgress } from '@tanem/react-nprogress'

import { cn } from '@/lib/utils'

interface ProgressBarProps {
  isAnimating: boolean
  className?: string
  barClassName?: string
}

export function ProgressBar({
  isAnimating,
  className,
  barClassName,
}: ProgressBarProps) {
  const { animationDuration, isFinished, progress } = useNProgress({
    isAnimating,
  })

  return (
    <div
      className={cn(
        'pointer-events-none absolute inset-x-0 top-0 z-50 h-1 rounded-t-xl',
        className,
      )}
      style={{
        opacity: isFinished ? 0 : 1,
        transition: `opacity ${animationDuration}ms linear`,
      }}
    >
      <div
        className={cn('h-full w-full bg-primary origin-left', barClassName)}
        style={{
          marginLeft: `${(-1 + progress) * 100}%`,
          transition: `margin-left ${animationDuration}ms linear`,
        }}
      />
    </div>
  )
}
