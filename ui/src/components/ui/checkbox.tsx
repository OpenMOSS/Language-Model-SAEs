import * as React from "react";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

export interface CheckboxProps {
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
  id?: string;
}

const Checkbox = React.forwardRef<HTMLButtonElement, CheckboxProps>(
  ({ checked = false, onCheckedChange, disabled = false, className, id }, ref) => {
    const handleClick = () => {
      if (!disabled) {
        onCheckedChange?.(!checked);
      }
    };

    return (
      <button
        ref={ref}
        id={id}
        type="button"
        role="checkbox"
        aria-checked={checked}
        disabled={disabled}
        onClick={handleClick}
        className={cn(
          "h-4 w-4 rounded-sm border border-input bg-background",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          checked && "bg-primary border-primary text-primary-foreground",
          disabled && "cursor-not-allowed opacity-50",
          "inline-flex items-center justify-center cursor-pointer transition-colors",
          className
        )}
      >
        {checked && <Check className="h-3 w-3" />}
      </button>
    );
  }
);

Checkbox.displayName = "Checkbox";

export { Checkbox };
