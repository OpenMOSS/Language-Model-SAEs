import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SliderProps extends Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  'onChange'
> {
  label?: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  showValue?: boolean
  valueFormatter?: (value: number) => string
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      className,
      label,
      value,
      onChange,
      min = 0,
      max = 100,
      step = 1,
      showValue = true,
      valueFormatter = (v) => String(v),
      ...props
    },
    ref,
  ) => {
    return (
      <div className={cn('flex flex-col gap-2', className)}>
        {(label || showValue) && (
          <div className="flex items-center justify-between">
            {label && (
              <label className="text-sm font-medium text-slate-700">
                {label}
              </label>
            )}
            {showValue && (
              <span className="text-sm text-slate-500">
                {valueFormatter(value)}
              </span>
            )}
          </div>
        )}
        <div className="flex items-center gap-3">
          <input
            type="number"
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            min={min}
            max={max}
            step={step}
            className="w-16 h-8 px-2 text-sm border border-input rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-ring"
          />
          <input
            type="range"
            ref={ref}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            min={min}
            max={max}
            step={step}
            className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary"
            {...props}
          />
        </div>
      </div>
    )
  },
)
Slider.displayName = 'Slider'

export { Slider }
