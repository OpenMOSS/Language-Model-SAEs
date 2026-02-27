import * as React from 'react'
import { cn } from '@/lib/utils'

export interface LabeledInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string
}

const LabeledInput = React.forwardRef<HTMLInputElement, LabeledInputProps>(
  ({ className, label, ...props }, ref) => {
    return (
      <div className="group relative flex h-12 w-full items-center rounded-md border border-input bg-white px-3 ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2">
        <div className="flex w-full flex-col items-start text-left gap-0.5">
          <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none">
            {label}
          </label>
          <input
            className={cn(
              'flex h-6 w-full bg-transparent text-base font-medium placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 -my-1',
              className,
            )}
            ref={ref}
            {...props}
          />
        </div>
      </div>
    )
  },
)
LabeledInput.displayName = 'LabeledInput'

export { LabeledInput }
