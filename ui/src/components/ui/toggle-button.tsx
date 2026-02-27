import * as React from 'react'
import { cva } from 'class-variance-authority'
import type { VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

const toggleVariants = cva('transition-all duration-200 ease-in-out', {
  variants: {
    pressed: {
      true: 'bg-primary text-primary-foreground hover:bg-primary/90',
      false:
        'bg-slate-200 text-slate-500 hover:bg-slate-300 hover:text-slate-600',
    },
    size: {
      default: 'h-10 px-4 py-2',
      sm: 'h-9 rounded-md px-3',
      lg: 'h-11 rounded-md px-8',
      icon: 'h-10 w-10',
    },
  },
  defaultVariants: {
    pressed: false,
    size: 'default',
  },
})

export interface ToggleButtonProps
  extends
    React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof toggleVariants> {
  pressed?: boolean
  onPressedChange?: (pressed: boolean) => void
}

const ToggleButton = React.forwardRef<HTMLButtonElement, ToggleButtonProps>(
  (
    {
      className,
      pressed = false,
      onPressedChange,
      size,
      children,
      onClick,
      ...props
    },
    ref,
  ) => {
    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (onPressedChange) {
        onPressedChange(!pressed)
      }
      onClick?.(e)
    }

    return (
      <Button
        ref={ref}
        type="button"
        variant="ghost" // Use ghost as base since we handle colors manually
        size={size}
        className={cn(toggleVariants({ pressed, size, className }))}
        onClick={handleClick}
        {...props}
      >
        {children}
      </Button>
    )
  },
)
ToggleButton.displayName = 'ToggleButton'

export { ToggleButton, toggleVariants }
