import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Search } from 'lucide-react'
import { useMemo, useRef, useState } from 'react'
import { createSaeSet } from '@/api/circuits'
import { fetchDictionaries } from '@/api/features'
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
import { Spinner } from '@/components/ui/spinner'

interface CreateSaeSetDialogProps {
  onSaeSetCreated: (setName: string) => void
}

export function CreateSaeSetDialog({
  onSaeSetCreated,
}: CreateSaeSetDialogProps) {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newSetName, setNewSetName] = useState('')
  const [selectedSaeNames, setSelectedSaeNames] = useState<string[]>([])
  const [filterText, setFilterText] = useState('')
  const lastClickedIndexRef = useRef<number | null>(null)

  const { data: availableSaes = [], isLoading: isLoadingSaes } = useQuery({
    queryKey: ['dictionaries'],
    queryFn: () => fetchDictionaries(),
    enabled: dialogOpen,
  })

  const filteredSaes = useMemo(() => {
    if (!filterText.trim()) return availableSaes
    const lowerFilter = filterText.toLowerCase()
    return availableSaes.filter((sae) =>
      sae.toLowerCase().includes(lowerFilter),
    )
  }, [availableSaes, filterText])

  const filteredSaesSet = useMemo(() => new Set(filteredSaes), [filteredSaes])

  const allFilteredSelected = useMemo(() => {
    if (filteredSaes.length === 0) return false
    return filteredSaes.every((sae) => selectedSaeNames.includes(sae))
  }, [filteredSaes, selectedSaeNames])

  const {
    mutate: mutateCreateSaeSet,
    isPending: isCreating,
    error: createError,
  } = useMutation({
    mutationFn: createSaeSet,
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ['sae-sets'],
      })
      onSaeSetCreated(newSetName)
      setDialogOpen(false)
      setNewSetName('')
      setSelectedSaeNames([])
      lastClickedIndexRef.current = null
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
    const originalIndex = availableSaes.indexOf(saeName)

    if (event.shiftKey && lastClickedIndexRef.current !== null) {
      const lastClickedSae = availableSaes[lastClickedIndexRef.current]
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

  const handleDialogOpenChange = (open: boolean) => {
    setDialogOpen(open)
    if (!open) {
      setNewSetName('')
      setSelectedSaeNames([])
      setFilterText('')
      lastClickedIndexRef.current = null
    }
  }

  return (
    <Dialog open={dialogOpen} onOpenChange={handleDialogOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="icon"
          className="h-10 w-10 shrink-0"
          title="Create new SAE set"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Create SAE Set</DialogTitle>
          <DialogDescription>
            Create a new SAE set by selecting multiple SAEs. Use Shift+Click to
            select a range.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4 py-4">
          <div>
            <label className="text-sm font-medium text-slate-700 mb-1.5 block">
              Set Name
            </label>
            <input
              type="text"
              value={newSetName}
              onChange={(e) => setNewSetName(e.target.value)}
              placeholder="Enter a name for the SAE set"
              className="w-full h-10 px-3 rounded-md border border-input bg-white text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:ring-offset-2"
              onKeyDown={(e) => {
                if (
                  e.key === 'Enter' &&
                  newSetName &&
                  selectedSaeNames.length > 0 &&
                  !isCreating
                ) {
                  handleCreateSaeSet()
                }
              }}
            />
          </div>
          <div className="flex flex-col gap-2 flex-1 min-h-0">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700">
                Select SAEs ({selectedSaeNames.length} selected)
              </label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleSelectAll}
                className="h-8 px-3 text-xs"
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
            <div className="border rounded-md p-3 overflow-y-auto flex-1 min-h-[200px] max-h-[300px] bg-slate-50">
              {isLoadingSaes ? (
                <div className="flex items-center justify-center h-full">
                  <Spinner isAnimating={true} />
                </div>
              ) : filteredSaes.length === 0 ? (
                <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                  {filterText.trim()
                    ? 'No SAEs match the filter'
                    : 'No SAEs available'}
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  {filteredSaes.map((sae, filteredIndex) => {
                    const isSelected = selectedSaeNames.includes(sae)
                    return (
                      <label
                        key={sae}
                        className="flex items-center gap-2 p-2 rounded hover:bg-slate-100 cursor-pointer"
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
                        <span className="text-sm text-slate-700">{sae}</span>
                      </label>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
          {createError && (
            <p className="text-sm text-red-600">
              {createError.message || 'Failed to create SAE set'}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => setDialogOpen(false)}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button
            onClick={handleCreateSaeSet}
            disabled={
              !newSetName || selectedSaeNames.length === 0 || isCreating
            }
          >
            {isCreating ? 'Creating...' : 'Create SAE Set'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
