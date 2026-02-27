import * as React from 'react'
import { Info as InfoIcon } from 'lucide-react'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card'
import { cn } from '@/lib/utils'

interface InfoProps extends React.ComponentPropsWithoutRef<typeof HoverCard> {
  children: React.ReactNode
  triggerClassName?: string
  contentClassName?: string
  iconSize?: number
}

export function Info({
  children,
  triggerClassName,
  contentClassName,
  iconSize = 16,
  ...props
}: InfoProps) {
  return (
    <HoverCard openDelay={200} closeDelay={100} {...props}>
      <HoverCardTrigger asChild>
        <InfoIcon
          size={iconSize}
          className={cn(
            'text-slate-700 hover:text-foreground transition-colors',
            triggerClassName,
          )}
        />
      </HoverCardTrigger>
      <HoverCardContent
        className={cn('w-80 text-sm font-normal', contentClassName)}
      >
        {children}
      </HoverCardContent>
    </HoverCard>
  )
}
