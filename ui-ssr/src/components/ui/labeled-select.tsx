import * as React from 'react'
import { cn } from '@/lib/utils'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface LabeledSelectProps extends React.ComponentProps<typeof Select> {
  label: string
  placeholder?: string
  options: Array<{ value: string; label: string }>
  className?: string
  triggerClassName?: string
}

export function LabeledSelect({
  label,
  placeholder,
  options,
  className,
  triggerClassName,
  ...props
}: LabeledSelectProps) {
  return (
    <Select {...props}>
      <SelectTrigger className={cn('h-12 px-3', triggerClassName)}>
        <div className="flex flex-col items-start text-left gap-0.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none">
            {label}
          </span>
          <span className="font-medium leading-none">
            <SelectValue placeholder={placeholder} />
          </span>
        </div>
      </SelectTrigger>
      <SelectContent className={className}>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
