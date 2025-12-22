import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, MessageCircle, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { CircuitInput, CircuitListItem } from '@/api/circuits'
import { deleteCircuit } from '@/api/circuits'
import { cn } from '@/lib/utils'

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSeconds = Math.floor(diffMs / 1000)
  const diffMinutes = Math.floor(diffSeconds / 60)
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffSeconds < 60) {
    return 'just now'
  }
  if (diffMinutes < 60) {
    return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`
  }
  if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`
  }
  if (diffDays < 7) {
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`
  }

  return date.toLocaleDateString()
}

function formatInputDisplay(
  input: CircuitInput | undefined,
  prompt: string,
  maxLength: number,
): { text: string; isChatTemplate: boolean } {
  if (!input || input.inputType === 'plain_text') {
    return {
      text:
        prompt.length > maxLength ? `${prompt.slice(0, maxLength)}...` : prompt,
      isChatTemplate: false,
    }
  }

  // For chat template, show a summary of messages
  const messages = input.messages
  if (messages.length === 0) {
    return { text: '(empty chat)', isChatTemplate: true }
  }

  // Create a compact representation of the conversation
  const parts = messages.map((m) => {
    const rolePrefix =
      m.role === 'user' ? 'U' : m.role === 'assistant' ? 'A' : 'S'
    const contentPreview =
      m.content.length > 30 ? `${m.content.slice(0, 30)}...` : m.content
    return `[${rolePrefix}] ${contentPreview}`
  })

  const combined = parts.join(' → ')
  return {
    text:
      combined.length > maxLength
        ? `${combined.slice(0, maxLength)}...`
        : combined,
    isChatTemplate: true,
  }
}

interface GraphSelectorProps {
  circuits: CircuitListItem[]
  selectedCircuitId: string
  onSelect: (circuitId: string) => void
}

export function GraphSelector({
  circuits,
  selectedCircuitId,
  onSelect,
}: GraphSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()

  const { mutate: mutateDeleteCircuit } = useMutation({
    mutationFn: deleteCircuit,
    onSuccess: async (_, variables) => {
      await queryClient.invalidateQueries({ queryKey: ['circuits'] })
      if (variables.data.circuitId === selectedCircuitId) {
        onSelect('')
      }
    },
  })

  const handleDeleteCircuit = (
    e: React.MouseEvent,
    circuit: CircuitListItem,
  ) => {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm(`Delete graph "${circuit.name || circuit.id}"?`)) return
    mutateDeleteCircuit({ data: { circuitId: circuit.id } })
  }

  const handleSelect = (circuitId: string) => {
    onSelect(circuitId)
    setIsOpen(false)
  }

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedCircuit = circuits.find((c) => c.id === selectedCircuitId)

  return (
    <div ref={containerRef} className="relative w-full">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex h-14 w-full items-center justify-between rounded-md border border-input bg-white px-4 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
      >
        {selectedCircuit ? (
          <div className="flex flex-col items-start text-left gap-0.5 overflow-hidden flex-1 mr-2">
            {(() => {
              const display = formatInputDisplay(
                selectedCircuit.input,
                selectedCircuit.prompt,
                60,
              )
              return (
                <span className="text-sm font-medium truncate w-full flex items-center gap-1.5">
                  {display.isChatTemplate && (
                    <MessageCircle className="h-3.5 w-3.5 shrink-0 text-primary" />
                  )}
                  {display.text}
                </span>
              )
            })()}
            <div className="flex items-center gap-2 text-xs text-slate-500 truncate w-full">
              <span className="truncate">
                {selectedCircuit.name || selectedCircuit.id}
              </span>
              <span className="shrink-0">·</span>
              <span className="shrink-0">
                {formatRelativeTime(selectedCircuit.createdAt)}
              </span>
            </div>
          </div>
        ) : (
          <span className="text-slate-500">Select a graph</span>
        )}
        <ChevronDown
          className={cn(
            'h-4 w-4 opacity-50 transition-transform shrink-0',
            isOpen && 'rotate-180',
          )}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 max-h-[400px] overflow-auto rounded-md border bg-white shadow-md">
          {circuits.length === 0 ? (
            <div className="py-4 px-3 text-sm text-slate-500 text-center">
              No graphs available. Create one to get started.
            </div>
          ) : (
            <div className="p-1">
              {circuits.map((c) => (
                <div
                  key={c.id}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded-sm px-3 py-2 cursor-pointer group',
                    c.id === selectedCircuitId
                      ? 'bg-slate-100'
                      : 'hover:bg-slate-50',
                  )}
                  onClick={() => handleSelect(c.id)}
                >
                  <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                    {(() => {
                      const display = formatInputDisplay(c.input, c.prompt, 50)
                      return (
                        <span className="text-sm font-medium truncate flex items-center gap-1.5">
                          {display.isChatTemplate && (
                            <MessageCircle className="h-3.5 w-3.5 shrink-0 text-primary" />
                          )}
                          {display.text}
                        </span>
                      )
                    })()}
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <span className="truncate">{c.name || c.id}</span>
                      <span className="shrink-0">·</span>
                      <span className="shrink-0">
                        {formatRelativeTime(c.createdAt)}
                      </span>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => handleDeleteCircuit(e, c)}
                    className="p-1.5 rounded hover:bg-red-100 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                    title="Delete graph"
                  >
                    <Trash2 className="h-4 w-4 text-red-500" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
