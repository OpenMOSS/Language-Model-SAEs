import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Filter, Plus, Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { orderBy } from 'natural-orderby'
import { createSaeSet } from '@/api/circuits'
import { fetchAdminSaes } from '@/api/admin'
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
import { cn } from '@/lib/utils'

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
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState<Record<string, string | null>>({})
  const lastClickedIndexRef = useRef<number | null>(null)

  const { data: adminSaesData, isLoading: isLoadingSaes } = useQuery({
    queryKey: ['admin', 'saes', { limit: 1000 }],
    queryFn: () => fetchAdminSaes({ data: { limit: 1000 } }),
    enabled: dialogOpen,
  })

  const availableSaes = useMemo(
    () => adminSaesData?.saes || [],
    [adminSaesData],
  )

  const stringify = (value: any) => {
    if (value === undefined || value === null) return 'None'
    if (typeof value === 'string') return value
    return JSON.stringify(value)
  }

  const saeConfigs = useMemo(() => {
    const configMap: Record<string, Record<string, any>> = {}
    const sortedSaes = orderBy(availableSaes, [(s) => s.name])
    sortedSaes.forEach((sae) => {
      if (sae.cfg) {
        const {
          hookPointIn,
          hookPointOut,
          saePretrainedNameOrPath,
          saeType,
          ...filteredCfg
        } = sae.cfg
        // Try to extract layer number from hookPointIn if matches blocks.{layer}
        const layerMatch = hookPointIn?.match(/blocks\.(\d+)/)
        const layer = layerMatch ? parseInt(layerMatch[1], 10) : undefined
        const dictionaryType = (() => {
          if (saeType === 'sae') {
            if (hookPointIn === hookPointOut) {
              return 'SAE'
            } else {
              return 'Transcoder'
            }
          } else if (saeType === 'clt') {
            return 'Cross Layer Transcoder'
          } else if (saeType === 'crosscoder') {
            return 'CrossCoder'
          } else if (saeType === 'lorsa') {
            return 'Lorsa'
          } else if (saeType === 'molt') {
            return 'MoLT'
          } else {
            return saeType
          }
        })()
        configMap[sae.name] = { dictionaryType, ...filteredCfg, layer }
      }
    })
    return configMap
  }, [availableSaes])

  const varyingFields = useMemo(() => {
    const configs = Object.values(saeConfigs)
    const allKeys = Array.from(new Set(configs.flatMap((c) => Object.keys(c))))
    return allKeys.filter((key) => {
      const values = configs.map((c) => c[key])
      const uniqueValues = new Set(
        values.filter((v) => v !== null && v !== undefined).map(stringify),
      )
      return uniqueValues.size > 1
    })
  }, [saeConfigs])

  useEffect(() => {
    if (dialogOpen) {
      const initialFilters: Record<string, string | null> = {}
      varyingFields.forEach((field) => {
        initialFilters[field] = null
      })
      setFilters(initialFilters)
      setSearchQuery('')
    }
  }, [dialogOpen, varyingFields])

  const filteredSaes = useMemo(() => {
    return Object.entries(saeConfigs)
      .filter(([name, cfg]) => {
        const matchesSearch = name
          .toLowerCase()
          .includes(searchQuery.toLowerCase())
        if (!matchesSearch) return false

        if (!cfg) return searchQuery === ''

        return Object.entries(filters).every(([key, value]) => {
          if (value === null) return true
          return stringify(cfg[key]) === value
        })
      })
      .map(([name]) => name)
  }, [saeConfigs, filters, searchQuery])

  const activeFilterCount = useMemo(() => {
    return Object.values(filters).filter((v) => v !== null).length
  }, [filters])

  const handleFilterChange = (key: string, value: string | null) => {
    setFilters((prev) => {
      const newFilters = { ...prev, [key]: value }
      const fieldIdx = varyingFields.indexOf(key)
      if (fieldIdx !== -1) {
        varyingFields.slice(fieldIdx + 1).forEach((f) => {
          newFilters[f] = null
        })
      }
      return newFilters
    })
  }

  const handleClearFilters = () => {
    const clearedFilters: Record<string, null> = {}
    varyingFields.forEach((field) => {
      clearedFilters[field] = null
    })
    setFilters(clearedFilters)
    setSearchQuery('')
  }

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
    const originalIndex = availableSaes.findIndex((s) => s.name === saeName)

    if (event.shiftKey && lastClickedIndexRef.current !== null) {
      const lastClickedSaeName = availableSaes[lastClickedIndexRef.current].name
      const lastClickedFilteredIndex = filteredSaes.indexOf(lastClickedSaeName)

      if (lastClickedFilteredIndex !== -1) {
        const start = Math.min(lastClickedFilteredIndex, filteredIndex)
        const end = Math.max(lastClickedFilteredIndex, filteredIndex)
        const range = filteredSaes.slice(start, end + 1)

        setSelectedSaeNames((prev) => {
          const newSelection = new Set(prev)
          const isAdding = !prev.includes(lastClickedSaeName)

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
      setSearchQuery('')
      setFilters({})
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
      <DialogContent className="max-w-6xl w-[90vw] max-h-[90vh] overflow-hidden flex flex-col p-0">
        <DialogHeader className="px-6 pt-6">
          <DialogTitle>Create SAE Set</DialogTitle>
          <DialogDescription>
            Create a new SAE set by selecting multiple SAEs. Use Shift+Click to
            select a range.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 flex overflow-hidden min-h-0 mt-4 border-y">
          {/* Left: Filters */}
          {varyingFields.length > 0 && (
            <div className="w-64 border-r bg-slate-50/50 p-5 overflow-y-auto space-y-6 shrink-0">
              <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                <div className="flex items-center gap-2">
                  <Filter className="w-3 h-3" />
                  Filter by
                </div>
                {(activeFilterCount > 0 || searchQuery) && (
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      handleClearFilters()
                    }}
                    className="text-blue-500 hover:text-blue-700 transition-colors font-bold text-[10px] tracking-wider uppercase"
                  >
                    Clear All
                  </button>
                )}
              </div>
              {varyingFields.map((field, fieldIdx) => {
                const previousFilters = varyingFields.slice(0, fieldIdx).reduce(
                  (acc, f) => {
                    if (filters[f] !== null) acc[f] = filters[f]
                    return acc
                  },
                  {} as Record<string, string | null>,
                )

                const refinedSaes = Object.values(saeConfigs).filter((cfg) =>
                  Object.entries(previousFilters).every(
                    ([key, val]) => stringify(cfg[key]) === val,
                  ),
                )

                const options = orderBy(
                  Array.from(
                    new Set(refinedSaes.map((s) => stringify(s[field]))),
                  ),
                )

                return (
                  <div key={field} className="space-y-2.5">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">
                      {field.replace(
                        /[A-Z]/g,
                        (letter) => ` ${letter.toLowerCase()}`,
                      )}
                    </label>
                    <div className="flex flex-wrap gap-1">
                      <button
                        className={cn(
                          'px-2 py-1 text-[10px] font-medium rounded transition-all',
                          filters[field] === null
                            ? 'bg-slate-200 text-slate-900'
                            : 'bg-white text-slate-500 hover:bg-slate-100',
                        )}
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          handleFilterChange(field, null)
                        }}
                      >
                        ALL
                      </button>
                      {options.map((v) => (
                        <button
                          key={v}
                          className={cn(
                            'px-2 py-1 text-[10px] font-medium rounded transition-all',
                            filters[field] === v
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-white text-slate-500 hover:bg-slate-100',
                          )}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            handleFilterChange(field, v)
                          }}
                        >
                          {v}
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Right: Content */}
          <div className="flex-1 flex flex-col min-w-0 bg-white p-6 gap-6">
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

            <div className="flex flex-col gap-4 flex-1 min-h-0">
              <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                  <label className="text-sm font-medium text-slate-700">
                    Select SAEs ({selectedSaeNames.length} selected)
                  </label>
                  <div className="text-xs text-slate-500">
                    {filteredSaes.length} SAEs match filters
                  </div>
                </div>
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
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search SAEs by name..."
                  className="pl-9 h-9"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2"
                  >
                    <X className="h-4 w-4 text-slate-400 hover:text-slate-600" />
                  </button>
                )}
              </div>

              <div className="border rounded-md p-3 overflow-y-auto flex-1 min-h-[200px] bg-slate-50">
                {isLoadingSaes ? (
                  <div className="flex items-center justify-center h-full">
                    <Spinner isAnimating={true} />
                  </div>
                ) : filteredSaes.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                    {searchQuery.trim() || activeFilterCount > 0
                      ? 'No SAEs match the current filters'
                      : 'No SAEs available'}
                  </div>
                ) : (
                  <div className="flex flex-col gap-1">
                    {filteredSaes.map((sae, filteredIndex) => {
                      const isSelected = selectedSaeNames.includes(sae)
                      return (
                        <label
                          key={sae}
                          className={cn(
                            'flex items-center gap-2 p-2 rounded hover:bg-slate-100 cursor-pointer transition-colors',
                            isSelected && 'bg-blue-50/50 hover:bg-blue-50',
                          )}
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
                            onCheckedChange={() => {}}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleSaeClick(sae, filteredIndex, e)
                            }}
                          />
                          <span
                            className={cn(
                              'text-sm text-slate-700',
                              isSelected && 'font-medium text-blue-700',
                            )}
                          >
                            {sae}
                          </span>
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
        </div>

        <DialogFooter className="px-6 pb-6 pt-4">
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
