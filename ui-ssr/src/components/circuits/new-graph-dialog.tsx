import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Plus,
  Search,
  Settings2,
  Type,
  X,
} from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import type {
  ChatMessage,
  CircuitInput,
  GenerateCircuitParams,
} from '@/api/circuits'
import {
  applyChatTemplate,
  circuitQueryOptions,
  createSaeSet,
  generateCircuit,
} from '@/api/circuits'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
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
import { useDebounce } from '@/hooks/use-debounce'

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
  const [useChatTemplate, setUseChatTemplate] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: 'user', content: '' },
    { role: 'assistant', content: '' },
  ])

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

  const handleAddMessage = () => {
    const lastRole = chatMessages[chatMessages.length - 1]?.role
    const nextRole: ChatMessage['role'] =
      lastRole === 'user' ? 'assistant' : 'user'
    setChatMessages([...chatMessages, { role: nextRole, content: '' }])
  }

  const handleRemoveMessage = (index: number) => {
    if (chatMessages.length <= 1) return
    setChatMessages(chatMessages.filter((_, i) => i !== index))
  }

  const handleMessageChange = (index: number, content: string) => {
    setChatMessages(
      chatMessages.map((msg, i) => (i === index ? { ...msg, content } : msg)),
    )
  }

  const handleMessageRoleChange = (
    index: number,
    role: ChatMessage['role'],
  ) => {
    setChatMessages(
      chatMessages.map((msg, i) => (i === index ? { ...msg, role } : msg)),
    )
  }

  const hasValidInput = useChatTemplate || prompt.trim()

  const debouncedChatMessages = useDebounce(chatMessages, 500)
  const debouncedSelectedSaeSet = useDebounce(selectedSaeSet, 500)

  const { data: chatTemplateData, isLoading: isPreviewLoading } = useQuery({
    queryKey: ['chat-template', debouncedSelectedSaeSet, debouncedChatMessages],
    queryFn: () =>
      applyChatTemplate({
        data: {
          saeSetName: debouncedSelectedSaeSet,
          messages: debouncedChatMessages,
        },
      }),
    enabled: useChatTemplate && !!debouncedSelectedSaeSet,
  })

  const previewPrompt = chatTemplateData?.prompt ?? ''

  const handleStartGeneration = () => {
    if (!selectedSaeSet || !hasValidInput) return

    const input: CircuitInput = useChatTemplate
      ? {
          inputType: 'chat_template',
          messages: chatMessages,
        }
      : { inputType: 'plain_text', text: prompt }

    const params: GenerateCircuitParams = {
      saeSetName: selectedSaeSet,
      input,
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
    setUseChatTemplate(false)
    setPrompt('')
    setChatMessages([
      { role: 'user', content: '' },
      { role: 'assistant', content: '' },
    ])
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

  const canGenerate = selectedSaeSet && hasValidInput && !isGenerating

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
            Generate a new attribution graph for a custom prompt.
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
                    min={0}
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

          {/* Prompt Input Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Input
              </label>
              {/* Input Mode Tabs */}
              <div className="flex rounded-lg bg-slate-100 p-0.5">
                <button
                  type="button"
                  onClick={() => setUseChatTemplate(false)}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all',
                    !useChatTemplate
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900',
                  )}
                >
                  <Type className="h-3.5 w-3.5" />
                  Plain Text
                </button>
                <button
                  type="button"
                  onClick={() => setUseChatTemplate(true)}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all',
                    useChatTemplate
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900',
                  )}
                >
                  <MessageSquare className="h-3.5 w-3.5" />
                  Chat
                </button>
              </div>
            </div>

            {useChatTemplate ? (
              <div className="space-y-2">
                <p className="text-xs text-slate-500">
                  Build a conversation. The model will predict the next token
                  after your messages.
                </p>
                <div className="space-y-2 rounded-lg border bg-slate-50/50 p-3">
                  {chatMessages.map((message, index) => {
                    return (
                      <div
                        key={index}
                        className="relative rounded-lg border bg-white p-3 shadow-sm"
                      >
                        <div className="flex items-start gap-2">
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center gap-2">
                              <Select
                                value={message.role}
                                onValueChange={(value: ChatMessage['role']) =>
                                  handleMessageRoleChange(index, value)
                                }
                              >
                                <SelectTrigger className="h-6 w-auto gap-1 border-0 px-2 py-0 text-[10px] font-bold uppercase tracking-wider shadow-none bg-slate-100 text-slate-700">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="user">User</SelectItem>
                                  <SelectItem value="assistant">
                                    Assistant
                                  </SelectItem>
                                  <SelectItem value="system">System</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <Textarea
                              value={message.content}
                              onChange={(e) =>
                                handleMessageChange(index, e.target.value)
                              }
                              placeholder={`What does the ${message.role} say?`}
                              className="min-h-[50px] resize-none border-0 bg-transparent p-0 text-sm shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
                            />
                          </div>
                          {chatMessages.length > 1 && (
                            <button
                              type="button"
                              onClick={() => handleRemoveMessage(index)}
                              className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                              title="Remove message"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                  <button
                    type="button"
                    onClick={handleAddMessage}
                    className="flex w-full items-center justify-center gap-1.5 rounded-lg border-2 border-dashed border-slate-200 py-2 text-xs font-medium text-slate-500 transition-colors hover:border-slate-300 hover:bg-white/50 hover:text-slate-700"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Add{' '}
                    {chatMessages[chatMessages.length - 1]?.role === 'user'
                      ? 'Assistant'
                      : 'User'}{' '}
                    Message
                  </button>
                </div>
                {previewPrompt && (
                  <div className="space-y-1.5">
                    <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      Formatted Prompt Preview
                    </label>
                    <div className="rounded-md border bg-slate-50 p-2.5 font-mono text-xs text-slate-600 break-all whitespace-pre-wrap">
                      {isPreviewLoading ? (
                        <div className="flex items-center gap-2 text-slate-400">
                          <Spinner isAnimating={true} className="h-3 w-3" />
                          Updating preview...
                        </div>
                      ) : (
                        previewPrompt
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-xs text-slate-500">
                  Enter a prompt ending mid-sentence. We&apos;ll analyze how the
                  model predicts the next token.
                </p>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder='e.g., "The capital of the state containing Dallas is"'
                  className="min-h-[100px] bg-white"
                />
              </div>
            )}
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
