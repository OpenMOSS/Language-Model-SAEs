import { memo, useEffect, useMemo, useState } from 'react'
import { Check, Filter, Search, X } from 'lucide-react'
import { orderBy } from 'natural-orderby'
import type { AdminSae } from '@/api/admin'
import { Select, SelectContent, SelectTrigger } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

interface DictionarySelectProps {
  saes: AdminSae[]
  selectedDictionary: string
  onSelect: (name: string) => void
}

const DictionarySelect = memo(
  ({ saes, selectedDictionary, onSelect }: DictionarySelectProps) => {
    const [filters, setFilters] = useState<Record<string, string | null>>({})
    const [searchQuery, setSearchQuery] = useState('')
    const [open, setOpen] = useState(false)

    const stringify = (value: any) => {
      if (value === undefined || value === null) return 'None'
      if (typeof value === 'string') return value
      return JSON.stringify(value)
    }

    const saeConfigs = useMemo(() => {
      const configMap: Record<string, Record<string, any>> = {}
      const sortedSaes = orderBy(saes, [(s) => s.name])
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
    }, [saes])

    const varyingFields = useMemo(() => {
      const configs = Object.values(saeConfigs)

      const allKeys = Array.from(
        new Set(configs.flatMap((c) => Object.keys(c))),
      )
      return allKeys.filter((key) => {
        const values = configs.map((c) => c[key])
        const uniqueValues = new Set(
          values.filter((v) => v !== null && v !== undefined).map(stringify),
        )
        return uniqueValues.size > 1
      })
    }, [saeConfigs])

    const currentSaeConfig = useMemo(
      () => saeConfigs[selectedDictionary] || {},
      [saeConfigs, selectedDictionary],
    )

    useEffect(() => {
      if (open) {
        const initialFilters: Record<string, string | null> = {}
        varyingFields.forEach((field) => {
          initialFilters[field] = stringify(currentSaeConfig[field])
        })
        setFilters(initialFilters)
        setSearchQuery('')
      }
    }, [open, varyingFields, currentSaeConfig])

    const filteredDictionaries = useMemo(() => {
      return Object.entries(saeConfigs).filter(([name, cfg]) => {
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
    }, [saeConfigs, filters, searchQuery])

    const activeFilterCount = useMemo(() => {
      return Object.values(filters).filter((v) => v !== null).length
    }, [filters])

    const handleFilterChange = (key: string, value: string | null) => {
      setFilters((prev) => {
        const newFilters = { ...prev, [key]: value }
        // When a field changes, reset all subsequent varying fields to "ALL"
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
    }

    const handleSelect = (name: string) => {
      onSelect(name)
      setOpen(false)
    }

    return (
      <Select open={open} onOpenChange={setOpen}>
        <SelectTrigger
          className={cn(
            'flex h-12 w-[400px] items-center justify-between rounded-md border border-input bg-white px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 select-none',
          )}
        >
          <div className="flex flex-col items-start text-left gap-0.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none">
              Dictionary
            </span>
            <span className="font-medium leading-none truncate max-w-[320px]">
              {selectedDictionary || 'Select a dictionary'}
            </span>
          </div>
        </SelectTrigger>
        <SelectContent
          className="max-w-6xl w-[900px] max-h-[85vh] p-0 overflow-hidden select-none"
          align="center"
        >
          <div className="flex h-[60vh] overflow-hidden">
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
                        setSearchQuery('')
                      }}
                      className="text-blue-500 hover:text-blue-700 transition-colors font-bold text-[10px] tracking-wider uppercase"
                    >
                      Clear All
                    </button>
                  )}
                </div>
                {varyingFields.map((field, fieldIdx) => {
                  // Get SAEs that match all filters before this one
                  const previousFilters = varyingFields
                    .slice(0, fieldIdx)
                    .reduce(
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

            {/* Right: Dictionary List */}
            <div className="flex-1 flex flex-col min-w-0 bg-white">
              <div className="px-4 py-3 border-b sticky top-0 z-10 space-y-3 bg-white">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold text-slate-500">
                    {filteredDictionaries.length} dictionaries
                    {(activeFilterCount > 0 || searchQuery) && (
                      <span className="text-slate-400">
                        {' '}
                        (filtered from {Object.values(saeConfigs).length})
                      </span>
                    )}
                  </div>
                </div>
                <div className="relative">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search dictionary by name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.stopPropagation()}
                    className="pl-9 h-9 text-sm"
                  />
                  {searchQuery && (
                    <button
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setSearchQuery('')
                      }}
                      className="absolute right-2.5 top-2.5"
                    >
                      <X className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                    </button>
                  )}
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {filteredDictionaries.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center p-8">
                    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
                      <Search className="w-5 h-5 text-slate-400" />
                    </div>
                    <p className="text-sm font-medium text-slate-600">
                      No dictionaries match
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      Try adjusting your filters or search
                    </p>
                  </div>
                ) : (
                  <div className="grid gap-1">
                    {filteredDictionaries.map(([name, _]) => (
                      <button
                        key={name}
                        onClick={() => handleSelect(name)}
                        className={cn(
                          'w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center gap-3 group',
                          selectedDictionary === name
                            ? 'bg-blue-50 text-blue-700'
                            : 'hover:bg-slate-50 text-slate-700',
                        )}
                      >
                        <div
                          className={cn(
                            'w-4 h-4 rounded-full border flex items-center justify-center shrink-0 transition-colors',
                            selectedDictionary === name
                              ? 'border-blue-600 bg-blue-600'
                              : 'border-slate-300 group-hover:border-slate-400',
                          )}
                        >
                          {selectedDictionary === name && (
                            <Check className="w-2.5 h-2.5 text-white" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium truncate">
                            {name}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </SelectContent>
      </Select>
    )
  },
)

DictionarySelect.displayName = 'DictionarySelect'

export { DictionarySelect }
