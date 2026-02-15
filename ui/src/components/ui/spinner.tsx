import { Loader2 } from 'lucide-react'

import { cn } from '@/lib/utils'

interface SpinnerProps {
  isAnimating: boolean
  className?: string
}

export function Spinner({ isAnimating, className }: SpinnerProps) {
  if (!isAnimating) return null

  return (
    <div
      className={cn(
        'pointer-events-none transition-opacity ease-linear',
        className,
      )}
    >
      <Loader2 className="animate-spin" />
    </div>
  )
}
