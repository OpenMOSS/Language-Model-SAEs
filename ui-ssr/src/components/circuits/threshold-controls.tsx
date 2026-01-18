import { Loader2, Settings2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Info } from '@/components/ui/info'
import { Slider } from '@/components/ui/slider'
import { cn } from '@/lib/utils'

interface ThresholdControlsProps {
  nodeThreshold: number
  edgeThreshold: number
  onThresholdsChange: (nodeThreshold: number, edgeThreshold: number) => void
  isLoading?: boolean
  className?: string
}

export function ThresholdControls({
  nodeThreshold,
  edgeThreshold,
  onThresholdsChange,
  isLoading = false,
  className,
}: ThresholdControlsProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [localNodeThreshold, setLocalNodeThreshold] = useState(nodeThreshold)
  const [localEdgeThreshold, setLocalEdgeThreshold] = useState(edgeThreshold)

  // Sync local state when props change (e.g., when switching circuits)
  useEffect(() => {
    setLocalNodeThreshold(nodeThreshold)
    setLocalEdgeThreshold(edgeThreshold)
  }, [nodeThreshold, edgeThreshold])

  const hasChanges =
    localNodeThreshold !== nodeThreshold || localEdgeThreshold !== edgeThreshold

  const handleApply = () => {
    onThresholdsChange(localNodeThreshold, localEdgeThreshold)
    setIsExpanded(false)
  }

  const handleReset = () => {
    setLocalNodeThreshold(nodeThreshold)
    setLocalEdgeThreshold(edgeThreshold)
  }

  return (
    <div className={cn('relative', className)}>
      {/* Collapsed view / Trigger button */}
      <button
        type="button"
        onClick={() => setIsExpanded(true)}
        className={cn(
          'flex h-10 items-center gap-2.5 rounded-md border border-input bg-white px-3 py-2 text-sm font-medium shadow-sm transition-all hover:bg-slate-50 hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          isLoading && 'ring-2 ring-blue-100 border-blue-200',
          isExpanded && 'opacity-0 pointer-events-none',
        )}
      >
        <Settings2 className="h-4 w-4 text-slate-500" />
        <div className="flex items-center gap-1.5 text-xs text-slate-600">
          <span>Thresholds:</span>
          <span className="font-mono text-slate-900 bg-slate-100/80 px-1.5 py-0.5 rounded-sm border border-slate-200/50">
            {nodeThreshold.toFixed(2)}
          </span>
          <span className="text-slate-300">/</span>
          <span className="font-mono text-slate-900 bg-slate-100/80 px-1.5 py-0.5 rounded-sm border border-slate-200/50">
            {edgeThreshold.toFixed(2)}
          </span>
        </div>
        {isLoading && (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
        )}
      </button>

      {/* Expanded panel */}
      {isExpanded && (
        <div className="absolute top-0 left-0 z-50 bg-white rounded-md border shadow-lg p-4 w-72 animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <Settings2 className="h-4 w-4 text-slate-500" />
              <h4 className="text-sm font-semibold text-slate-900">
                Pruning Thresholds
              </h4>
            </div>
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              className="text-slate-400 hover:text-slate-600 text-xs font-medium transition-colors"
            >
              Collapse
            </button>
          </div>

          <div className="space-y-6">
            <div className="space-y-3">
              <div className="flex items-center gap-1.5">
                <label className="text-xs font-semibold uppercase text-slate-500">
                  Node Threshold
                </label>
                <Info iconSize={14}>
                  Keep nodes contributing to this fraction of total influence.
                </Info>
              </div>
              <div className="px-1">
                <Slider
                  value={localNodeThreshold}
                  onChange={setLocalNodeThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                  showValue={false}
                />
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-1.5">
                <label className="text-xs font-semibold uppercase text-slate-500">
                  Edge Threshold
                </label>
                <Info iconSize={14}>
                  Keep edges contributing to this fraction of total influence.
                </Info>
              </div>
              <div className="px-1">
                <Slider
                  value={localEdgeThreshold}
                  onChange={setLocalEdgeThreshold}
                  min={0}
                  max={1}
                  step={0.01}
                  showValue={false}
                />
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleReset}
                disabled={!hasChanges || isLoading}
                className="flex-1 text-xs text-slate-500 hover:text-slate-900"
              >
                Reset
              </Button>
              <Button
                size="sm"
                onClick={handleApply}
                disabled={!hasChanges || isLoading}
                className="flex-1 text-xs"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin mr-1.5" />
                    Applying...
                  </>
                ) : (
                  'Apply Changes'
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
