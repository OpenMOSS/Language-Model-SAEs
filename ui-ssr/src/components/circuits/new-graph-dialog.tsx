import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronUp, Plus, Search, Settings2 } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import type { GenerateCircuitParams } from '@/api/circuits'
import {
  circuitQueryOptions,
  createSaeSet,
  generateCircuit,
} from '@/api/circuits'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Spinner } from '@/components/ui/spinner'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface NewGraphDialogProps {
  saeSets: string[]
  dictionaries: string[]
  onGraphCreated: (circuitId: string) => void
}

export function NewGraphDialog({
  saeSets,
  dictionaries,
  onGraphCreated,
}: NewGraphDialogProps) {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showCreateSaeSet, setShowCreateSaeSet] = useState(false)

  const [selectedSaeSet, setSelectedSaeSet] = useState<string>('')
  const [customGraphId, setCustomGraphId] = useState('')
  const [prompt, setPrompt] = useState('')

  const [desiredLogitProb, setDesiredLogitProb] = useState(0.98)
  const [maxNodes, setMaxNodes] = useState(256)
  const [maxLogits, setMaxLogits] = useState(1)
  const [qkTracingTopk, setQkTracingTopk] = useState(10)

  const [nodeThreshold, setNodeThreshold] = useState(0.8)
  const [edgeThreshold, setEdgeThreshold] = useState(0.98)

  const [newSetName, setNewSetName] = useState('')
  const [selectedSaeNames, setSelectedSaeNames] = useState<string[]>([])
  const [filterText, setFilterText] = useState('')
  const lastClickedIndexRef = useRef<number | null>(null)

  const filteredSaes = useMemo(() => {
    if (!filterText.trim()) return dictionaries
    const lowerFilter = filterText.toLowerCase()
    return dictionaries.filter((sae) => sae.toLowerCase().includes(lowerFilter))
  }, [dictionaries, filterText])

  const filteredSaesSet = useMemo(() => new Set(filteredSaes), [filteredSaes])

  const allFilteredSelected = useMemo(() => {
    if (filteredSaes.length === 0) return false
    return filteredSaes.every((sae) => selectedSaeNames.includes(sae))
  }, [filteredSaes, selectedSaeNames])

  const { mutate: mutateCreateSaeSet, isPending: isCreatingSaeSet } =
    useMutation({
      mutationFn: createSaeSet,
      onSuccess: async () => {
        await queryClient.invalidateQueries({ queryKey: ['sae-sets'] })
        setSelectedSaeSet(newSetName)
        setShowCreateSaeSet(false)
        setNewSetName('')
        setSelectedSaeNames([])
        setFilterText('')
        lastClickedIndexRef.current = null
      },
    })

  const {
    mutate: mutateGenerateCircuit,
    isPending: isGenerating,
    error: generateError,
  } = useMutation({
    mutationFn: generateCircuit,
    onSuccess: async (data) => {
      await queryClient.invalidateQueries({ queryKey: ['circuits'] })
      queryClient.setQueryData(
        circuitQueryOptions(data.circuitId).queryKey,
        data,
      )
      onGraphCreated(data.circuitId)
      handleDialogClose()
    },
  })

  const handleCreateSaeSet = () => {
    if (!newSetName || selectedSaeNames.length === 0) return
    mutateCreateSaeSet({
      data: {
        name: newSetName,
        saeNames: selectedSaeNames,
      },
    })
  }

  const handleSaeClick = (
    saeName: string,
    filteredIndex: number,
    event: React.MouseEvent,
  ) => {
    const originalIndex = dictionaries.indexOf(saeName)

    if (event.shiftKey && lastClickedIndexRef.current !== null) {
      const lastClickedSae = dictionaries[lastClickedIndexRef.current]
      const lastClickedFilteredIndex = filteredSaes.indexOf(lastClickedSae)

      if (lastClickedFilteredIndex !== -1) {
        const start = Math.min(lastClickedFilteredIndex, filteredIndex)
        const end = Math.max(lastClickedFilteredIndex, filteredIndex)
        const range = filteredSaes.slice(start, end + 1)

        setSelectedSaeNames((prev) => {
          const newSelection = new Set(prev)
          const isAdding = !prev.includes(lastClickedSae)

          if (isAdding) {
            range.forEach((sae) => newSelection.add(sae))
          } else {
            range.forEach((sae) => newSelection.delete(sae))
          }

          return Array.from(newSelection)
        })
      }
    } else {
      setSelectedSaeNames((prev) =>
        prev.includes(saeName)
          ? prev.filter((n) => n !== saeName)
          : [...prev, saeName],
      )
      lastClickedIndexRef.current = originalIndex
    }
  }

  const handleSelectAll = () => {
    if (allFilteredSelected) {
      setSelectedSaeNames((prev) =>
        prev.filter((sae) => !filteredSaesSet.has(sae)),
      )
    } else {
      setSelectedSaeNames((prev) => {
        const newSelection = new Set(prev)
        filteredSaes.forEach((sae) => newSelection.add(sae))
        return Array.from(newSelection)
      })
    }
  }

  const handleStartGeneration = () => {
    if (!selectedSaeSet || !prompt.trim()) return

    const params: GenerateCircuitParams = {
      saeSetName: selectedSaeSet,
      text: prompt,
      name: customGraphId.trim() || undefined,
      desiredLogitProb,
      maxFeatureNodes: maxNodes,
      maxNLogits: maxLogits,
      qkTracingTopk,
      nodeThreshold,
      edgeThreshold,
    }

    mutateGenerateCircuit({ data: params })
  }

  const handleReset = () => {
    setCustomGraphId('')
    setPrompt('')
    setDesiredLogitProb(0.98)
    setMaxNodes(256)
    setMaxLogits(1)
    setQkTracingTopk(10)
    setNodeThreshold(0.8)
    setEdgeThreshold(0.98)
  }

  const handleDialogClose = () => {
    setDialogOpen(false)
    setShowAdvanced(false)
    setShowCreateSaeSet(false)
    handleReset()
  }

  const handleDialogOpenChange = (open: boolean) => {
    if (open) {
      setDialogOpen(true)
    } else {
      handleDialogClose()
    }
  }

  const canGenerate = selectedSaeSet && prompt.trim() && !isGenerating

  return (
    <Dialog open={dialogOpen} onOpenChange={handleDialogOpenChange}>
      <DialogTrigger asChild>
        <Button className="h-12 px-4 gap-2">
          <Plus className="h-4 w-4" />
          New Graph
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">Generate New Graph</DialogTitle>
          <DialogDescription>
            Generate a new attribution graph for a custom prompt using{' '}
            <a
              href="https://github.com/TransformerLensOrg/circuit-tracer"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              circuit-tracer
            </a>
            .
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-6 py-4">
          {/* Model and Source Set Selection */}
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5 block">
                Source Set
              </label>
              <div className="flex gap-2">
                <Select
                  value={selectedSaeSet}
                  onValueChange={setSelectedSaeSet}
                >
                  <SelectTrigger className="h-10 bg-white flex-1">
                    <SelectValue placeholder="Select a source set" />
                  </SelectTrigger>
                  <SelectContent>
                    {saeSets.map((saeSet) => (
                      <SelectItem key={saeSet} value={saeSet}>
                        {saeSet}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-10 w-10 shrink-0"
                  onClick={() => setShowCreateSaeSet(!showCreateSaeSet)}
                  title="Create new SAE set"
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="flex-1">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5 block">
                Custom Graph ID (Optional)
              </label>
              <Input
                value={customGraphId}
                onChange={(e) => setCustomGraphId(e.target.value)}
                placeholder="my-graph"
                className="h-10"
              />
            </div>
          </div>

          {/* Create SAE Set Section */}
          {showCreateSaeSet && (
            <div className="border rounded-lg p-4 bg-slate-50 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-slate-700">
                  Create New SAE Set
                </h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCreateSaeSet(false)}
                >
                  Cancel
                </Button>
              </div>

              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5 block">
                  Set Name
                </label>
                <Input
                  value={newSetName}
                  onChange={(e) => setNewSetName(e.target.value)}
                  placeholder="Enter a name for the SAE set"
                  className="h-10"
                />
              </div>

              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Select SAEs ({selectedSaeNames.length} selected)
                  </label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleSelectAll}
                    className="h-7 px-2 text-xs"
                  >
                    {allFilteredSelected ? 'Deselect All' : 'Select All'}
                  </Button>
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <Input
                    type="text"
                    value={filterText}
                    onChange={(e) => setFilterText(e.target.value)}
                    placeholder="Filter SAEs..."
                    className="pl-9 h-9"
                  />
                </div>
                <div className="border rounded-md p-3 overflow-y-auto max-h-[200px] bg-white">
                  {filteredSaes.length === 0 ? (
                    <div className="flex items-center justify-center h-20 text-slate-500 text-sm">
                      {filterText.trim()
                        ? 'No SAEs match the filter'
                        : 'No SAEs available'}
                    </div>
                  ) : (
                    <div className="flex flex-col gap-1">
                      {filteredSaes.map((sae, filteredIndex) => {
                        const isSelected = selectedSaeNames.includes(sae)
                        return (
                          <label
                            key={sae}
                            className="flex items-center gap-2 p-1.5 rounded hover:bg-slate-100 cursor-pointer"
                            onClick={(e) => {
                              if (
                                e.target === e.currentTarget ||
                                (e.target as HTMLElement).tagName === 'SPAN'
                              ) {
                                handleSaeClick(sae, filteredIndex, e)
                              }
                            }}
                          >
                            <Checkbox
                              checked={isSelected}
                              onClick={(e) => {
                                e.stopPropagation()
                                handleSaeClick(sae, filteredIndex, e)
                              }}
                            />
                            <span className="text-sm text-slate-700">
                              {sae}
                            </span>
                          </label>
                        )
                      })}
                    </div>
                  )}
                </div>
                <Button
                  onClick={handleCreateSaeSet}
                  disabled={
                    !newSetName ||
                    selectedSaeNames.length === 0 ||
                    isCreatingSaeSet
                  }
                  className="mt-2"
                >
                  {isCreatingSaeSet ? 'Creating...' : 'Create SAE Set'}
                </Button>
              </div>
            </div>
          )}

          {/* Advanced Settings Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900 w-fit"
          >
            <Settings2 className="h-4 w-4" />
            Advanced Settings
            {showAdvanced ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>

          {showAdvanced && (
            <div className="space-y-6 border rounded-lg p-4 bg-slate-50">
              {/* Attribution Section */}
              <div className="space-y-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-primary">
                  Attribution
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Desired Logit Probability"
                    value={desiredLogitProb}
                    onChange={setDesiredLogitProb}
                    min={0}
                    max={1}
                    step={0.01}
                    showValue={false}
                  />
                  <Slider
                    label="Max # Nodes"
                    value={maxNodes}
                    onChange={setMaxNodes}
                    min={100}
                    max={10000}
                    step={100}
                    showValue={false}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Max # Logits"
                    value={maxLogits}
                    onChange={setMaxLogits}
                    min={1}
                    max={50}
                    step={1}
                    showValue={false}
                  />
                  <Slider
                    label="QK Tracing Top-K"
                    value={qkTracingTopk}
                    onChange={setQkTracingTopk}
                    min={1}
                    max={50}
                    step={1}
                    showValue={false}
                  />
                </div>
              </div>

              {/* Pruning Section */}
              <div className="space-y-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-primary">
                  Pruning
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Node Threshold"
                    value={nodeThreshold}
                    onChange={setNodeThreshold}
                    min={0}
                    max={1}
                    step={0.01}
                    showValue={false}
                  />
                  <Slider
                    label="Edge Threshold"
                    value={edgeThreshold}
                    onChange={setEdgeThreshold}
                    min={0}
                    max={1}
                    step={0.01}
                    showValue={false}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Prompt Input */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5 block">
              Prompt to Complete
            </label>
            <p className="text-sm text-slate-600 mb-2">
              In general, you want your prompt to be missing a word at the end,
              because we want to analyze how the model comes up with the word{' '}
              <strong>after</strong> your prompt. (Eg &ldquo;The capital of the
              state containing Dallas is&rdquo;)
            </p>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter the prompt to visualize..."
              className="min-h-[100px] bg-white"
            />
          </div>

          {generateError && (
            <p className="text-sm text-red-600">
              {generateError.message || 'Failed to generate graph'}
            </p>
          )}
        </div>

        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={isGenerating}
          >
            Reset
          </Button>
          <Button onClick={handleStartGeneration} disabled={!canGenerate}>
            {isGenerating ? (
              <>
                <Spinner isAnimating={true} className="mr-2" />
                Generating...
              </>
            ) : (
              'Start Generation'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
