import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import {
  AlertCircle,
  Bookmark,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  Edit2,
  FolderPlus,
  FolderTree,
  GitBranch,
  Loader2,
  Search,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import type { AdminCircuit, AdminSae, AdminSaeSet } from '@/api/admin'
import {
  adminCircuitsQueryOptions,
  adminSaeSetsQueryOptions,
  adminSaesQueryOptions,
  adminStatsQueryOptions,
  bulkGroupCircuits,
  deleteAdminCircuit,
  deleteSae,
  deleteSaeSet,
  updateCircuit,
  updateSae,
  updateSaeSet,
} from '@/api/admin'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { useDebounce } from '@/hooks/use-debounce'

export const Route = createFileRoute('/admin/')({
  component: AdminPage,
  staticData: {
    fullScreen: true,
  },
  loader: async ({ context }) => {
    const [stats, saes, saeSets, circuits] = await Promise.all([
      context.queryClient.ensureQueryData(adminStatsQueryOptions()),
      context.queryClient.ensureQueryData(adminSaesQueryOptions()),
      context.queryClient.ensureQueryData(adminSaeSetsQueryOptions()),
      context.queryClient.ensureQueryData(adminCircuitsQueryOptions()),
    ])
    return { stats, saes, saeSets, circuits }
  },
})

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: typeof Database
  label: string
  value: number
  color: string
}) {
  return (
    <Card
      className={cn(
        'p-6 flex items-center gap-4 border-l-4',
        color === 'emerald' && 'border-l-emerald-500',
        color === 'blue' && 'border-l-blue-500',
        color === 'purple' && 'border-l-purple-500',
        color === 'amber' && 'border-l-amber-500',
      )}
    >
      <div
        className={cn(
          'p-3 rounded-lg',
          color === 'emerald' && 'bg-emerald-100 text-emerald-600',
          color === 'blue' && 'bg-blue-100 text-blue-600',
          color === 'purple' && 'bg-purple-100 text-purple-600',
          color === 'amber' && 'bg-amber-100 text-amber-600',
        )}
      >
        <Icon className="h-6 w-6" />
      </div>
      <div>
        <p className="text-sm text-slate-500">{label}</p>
        <p className="text-2xl font-bold text-slate-800">{value}</p>
      </div>
    </Card>
  )
}

function DeleteConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  onConfirm,
  isDeleting,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  onConfirm: () => void
  isDeleting: boolean
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function SaesTab() {
  const queryClient = useQueryClient()
  const [page, setPage] = useState(0)
  const [searchTerm, setSearchTerm] = useState('')
  const debouncedSearch = useDebounce(searchTerm, 300)
  const pageSize = 20

  const { data, isLoading } = useQuery(
    adminSaesQueryOptions(
      pageSize,
      page * pageSize,
      debouncedSearch || undefined,
    ),
  )
  const saes = data?.saes ?? []
  const totalCount = data?.totalCount ?? 0
  const totalPages = Math.ceil(totalCount / pageSize)

  const [deleteTarget, setDeleteTarget] = useState<AdminSae | null>(null)
  const [editTarget, setEditTarget] = useState<AdminSae | null>(null)
  const [editName, setEditName] = useState('')
  const [editPath, setEditPath] = useState('')
  const [editModelName, setEditModelName] = useState('')

  const deleteMutation = useMutation({
    mutationFn: deleteSae,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setDeleteTarget(null)
    },
  })

  const updateMutation = useMutation({
    mutationFn: updateSae,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setEditTarget(null)
    },
  })

  const handleEdit = (sae: AdminSae) => {
    setEditTarget(sae)
    setEditName(sae.name)
    setEditPath(sae.path || '')
    setEditModelName(sae.modelName || '')
  }

  const handleSaveEdit = () => {
    if (!editTarget) return
    updateMutation.mutate({
      data: {
        name: editTarget.name,
        newName: editName !== editTarget.name ? editName : undefined,
        path: editPath || undefined,
        modelName: editModelName || undefined,
      },
    })
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Spinner isAnimating />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search SAEs..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value)
              setPage(0)
            }}
            className="pl-9"
          />
        </div>
        <span className="text-sm text-slate-500">
          {totalCount > 0
            ? `Showing ${page * pageSize + 1}-${Math.min((page + 1) * pageSize, totalCount)} of ${totalCount}`
            : '0 SAEs'}
        </span>
      </div>

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50">
              <TableHead>Name</TableHead>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Features</TableHead>
              <TableHead className="text-right">Analyses</TableHead>
              <TableHead>Path</TableHead>
              <TableHead className="w-24">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {saes.map((sae) => (
              <TableRow key={sae.name}>
                <TableCell className="font-medium">{sae.name}</TableCell>
                <TableCell className="text-slate-600">
                  {sae.modelName || '-'}
                </TableCell>
                <TableCell className="text-right">
                  {sae.featureCount.toLocaleString()}
                </TableCell>
                <TableCell className="text-right">
                  {sae.analysisCount}
                </TableCell>
                <TableCell className="max-w-[200px] truncate text-slate-500 text-xs">
                  {sae.path || '-'}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => handleEdit(sae)}
                    >
                      <Edit2 className="h-4 w-4 text-slate-500" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setDeleteTarget(sae)}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {saes.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={6}
                  className="text-center text-slate-500 py-8"
                >
                  No SAEs found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>

      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-slate-600">
            Page {page + 1} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      <DeleteConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete SAE"
        description={`Are you sure you want to delete "${deleteTarget?.name}"? This will delete all associated features and analyses. This action cannot be undone.`}
        onConfirm={() =>
          deleteTarget &&
          deleteMutation.mutate({ data: { name: deleteTarget.name } })
        }
        isDeleting={deleteMutation.isPending}
      />

      <Dialog
        open={!!editTarget}
        onOpenChange={(open) => !open && setEditTarget(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit SAE</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Name
              </label>
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="SAE name"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Path
              </label>
              <Input
                value={editPath}
                onChange={(e) => setEditPath(e.target.value)}
                placeholder="Path to SAE checkpoint"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Model Name
              </label>
              <Input
                value={editModelName}
                onChange={(e) => setEditModelName(e.target.value)}
                placeholder="Associated model name"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditTarget(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={updateMutation.isPending || !editName.trim()}
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function SaeSetsTab() {
  const queryClient = useQueryClient()
  const { data: saeSets = [], isLoading } = useQuery(adminSaeSetsQueryOptions())
  // Fetch all SAEs for selection (large limit)
  const { data: saesData } = useQuery(adminSaesQueryOptions(1000, 0))
  const allSaes = saesData?.saes ?? []
  const [searchTerm, setSearchTerm] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<AdminSaeSet | null>(null)
  const [editTarget, setEditTarget] = useState<AdminSaeSet | null>(null)
  const [editName, setEditName] = useState('')
  const [editSaeNames, setEditSaeNames] = useState<string[]>([])
  const [saeFilter, setSaeFilter] = useState('')

  const filteredSaeSets = useMemo(() => {
    if (!searchTerm.trim()) return saeSets
    const lower = searchTerm.toLowerCase()
    return saeSets.filter((set) => set.name.toLowerCase().includes(lower))
  }, [saeSets, searchTerm])

  const filteredSaeOptions = useMemo(() => {
    if (!saeFilter.trim()) return allSaes
    const lower = saeFilter.toLowerCase()
    return allSaes.filter((sae) => sae.name.toLowerCase().includes(lower))
  }, [allSaes, saeFilter])

  const deleteMutation = useMutation({
    mutationFn: deleteSaeSet,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setDeleteTarget(null)
    },
  })

  const updateMutation = useMutation({
    mutationFn: updateSaeSet,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setEditTarget(null)
    },
  })

  const handleEdit = (saeSet: AdminSaeSet) => {
    setEditTarget(saeSet)
    setEditName(saeSet.name)
    setEditSaeNames([...saeSet.saeNames])
  }

  const handleSaveEdit = () => {
    if (!editTarget) return
    updateMutation.mutate({
      data: {
        name: editTarget.name,
        newName: editName !== editTarget.name ? editName : undefined,
        saeNames: editSaeNames,
      },
    })
  }

  const toggleSaeSelection = (saeName: string) => {
    setEditSaeNames((prev) =>
      prev.includes(saeName)
        ? prev.filter((n) => n !== saeName)
        : [...prev, saeName],
    )
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Spinner isAnimating />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search SAE Sets..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9"
          />
        </div>
        <span className="text-sm text-slate-500">
          {filteredSaeSets.length} of {saeSets.length} SAE Sets
        </span>
      </div>

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50">
              <TableHead>Name</TableHead>
              <TableHead className="text-right">SAE Count</TableHead>
              <TableHead>SAEs</TableHead>
              <TableHead className="w-24">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredSaeSets.map((saeSet) => (
              <TableRow key={saeSet.name}>
                <TableCell className="font-medium">{saeSet.name}</TableCell>
                <TableCell className="text-right">
                  {saeSet.saeNames.length}
                </TableCell>
                <TableCell className="max-w-[400px]">
                  <div className="flex flex-wrap gap-1">
                    {saeSet.saeNames.slice(0, 5).map((name) => (
                      <span
                        key={name}
                        className="px-2 py-0.5 text-xs bg-slate-100 rounded-full text-slate-600"
                      >
                        {name}
                      </span>
                    ))}
                    {saeSet.saeNames.length > 5 && (
                      <span className="px-2 py-0.5 text-xs bg-slate-100 rounded-full text-slate-500">
                        +{saeSet.saeNames.length - 5} more
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => handleEdit(saeSet)}
                    >
                      <Edit2 className="h-4 w-4 text-slate-500" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setDeleteTarget(saeSet)}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {filteredSaeSets.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={4}
                  className="text-center text-slate-500 py-8"
                >
                  No SAE Sets found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>

      <DeleteConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete SAE Set"
        description={`Are you sure you want to delete "${deleteTarget?.name}"? This action cannot be undone.`}
        onConfirm={() =>
          deleteTarget &&
          deleteMutation.mutate({ data: { name: deleteTarget.name } })
        }
        isDeleting={deleteMutation.isPending}
      />

      <Dialog
        open={!!editTarget}
        onOpenChange={(open) => !open && setEditTarget(null)}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Edit SAE Set</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Name
              </label>
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="SAE Set name"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Selected SAEs ({editSaeNames.length})
              </label>
              <div className="relative mb-2">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  placeholder="Filter SAEs..."
                  value={saeFilter}
                  onChange={(e) => setSaeFilter(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="border rounded-md p-3 max-h-[300px] overflow-y-auto bg-slate-50">
                {filteredSaeOptions.map((sae) => (
                  <label
                    key={sae.name}
                    className="flex items-center gap-2 p-2 rounded hover:bg-slate-100 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={editSaeNames.includes(sae.name)}
                      onChange={() => toggleSaeSelection(sae.name)}
                      className="rounded border-slate-300"
                    />
                    <span className="text-sm text-slate-700">{sae.name}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditTarget(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={
                updateMutation.isPending ||
                editSaeNames.length === 0 ||
                !editName.trim()
              }
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function CircuitsTab() {
  const queryClient = useQueryClient()
  const [page, setPage] = useState(0)
  const pageSize = 10 // Reduced page size for larger items

  const { data, isLoading } = useQuery(
    adminCircuitsQueryOptions(pageSize, page * pageSize),
  )
  const circuits = data?.circuits ?? []
  const totalCount = data?.totalCount ?? 0
  const totalPages = Math.ceil(totalCount / pageSize)

  const [searchTerm, setSearchTerm] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<AdminCircuit | null>(null)
  const [editTarget, setEditTarget] = useState<AdminCircuit | null>(null)
  const [editName, setEditName] = useState('')
  const [editGroup, setEditGroup] = useState('')

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkGroupName, setBulkGroupName] = useState('')

  const filteredCircuits = useMemo(() => {
    if (!searchTerm.trim()) return circuits
    const lower = searchTerm.toLowerCase()
    return circuits.filter(
      (circuit) =>
        circuit.name?.toLowerCase().includes(lower) ||
        circuit.prompt.toLowerCase().includes(lower) ||
        circuit.saeSetName.toLowerCase().includes(lower) ||
        circuit.group?.toLowerCase().includes(lower),
    )
  }, [circuits, searchTerm])

  const deleteMutation = useMutation({
    mutationFn: deleteAdminCircuit,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setDeleteTarget(null)
    },
  })

  const updateMutation = useMutation({
    mutationFn: updateCircuit,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setEditTarget(null)
    },
  })

  const bulkGroupMutation = useMutation({
    mutationFn: bulkGroupCircuits,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin'] })
      setSelectedIds(new Set())
      setBulkGroupName('')
    },
  })

  const handleEdit = (circuit: AdminCircuit) => {
    setEditTarget(circuit)
    setEditName(circuit.name || '')
    setEditGroup(circuit.group || '')
  }

  const handleSaveEdit = () => {
    if (!editTarget) return
    updateMutation.mutate({
      data: {
        circuitId: editTarget.id,
        name: editName || undefined,
        group: editGroup || null,
      },
    })
  }

  const toggleSelect = (id: string) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  const toggleSelectAll = () => {
    if (
      selectedIds.size === filteredCircuits.length &&
      filteredCircuits.length > 0
    ) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredCircuits.map((c) => c.id)))
    }
  }

  const handleBulkGroup = () => {
    if (selectedIds.size === 0 || !bulkGroupName.trim()) return
    bulkGroupMutation.mutate({
      data: {
        circuitIds: Array.from(selectedIds),
        group: bulkGroupName.trim(),
      },
    })
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Spinner isAnimating />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search Circuits..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex-1 flex items-center gap-2">
          {selectedIds.size > 0 && (
            <div className="flex items-center gap-2 bg-slate-50 border rounded-md px-2 py-1 shadow-xs">
              <span className="text-xs font-medium text-slate-600 shrink-0">
                {selectedIds.size} selected
              </span>
              <Input
                placeholder="Group name..."
                value={bulkGroupName}
                onChange={(e) => setBulkGroupName(e.target.value)}
                className="h-8 text-xs w-48"
              />
              <Button
                size="sm"
                variant="outline"
                className="h-8 px-3 text-xs"
                onClick={handleBulkGroup}
                disabled={!bulkGroupName.trim() || bulkGroupMutation.isPending}
              >
                <FolderPlus className="h-3.5 w-3.5 mr-1.5" />
                Apply Group
              </Button>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm text-slate-500">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSelectAll}
            className="text-xs h-8"
          >
            {selectedIds.size === filteredCircuits.length &&
            filteredCircuits.length > 0
              ? 'Deselect All'
              : 'Select All'}
          </Button>
          <span>
            Showing {page * pageSize + 1}-
            {Math.min((page + 1) * pageSize, totalCount)} of {totalCount}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {filteredCircuits.map((circuit) => (
          <Card
            key={circuit.id}
            className={cn(
              'relative p-5 transition-all border-l-4 group',
              selectedIds.has(circuit.id)
                ? 'bg-slate-50 border-primary shadow-sm'
                : 'hover:bg-slate-50 border-l-slate-200',
            )}
          >
            <div className="flex gap-4">
              <div
                className={cn(
                  'mt-1 h-5 w-5 rounded border border-slate-300 flex items-center justify-center transition-colors cursor-pointer shrink-0',
                  selectedIds.has(circuit.id)
                    ? 'bg-primary border-primary text-white'
                    : 'bg-white hover:border-slate-400',
                )}
                onClick={() => toggleSelect(circuit.id)}
              >
                {selectedIds.has(circuit.id) && (
                  <Check className="h-3.5 w-3.5 stroke-3" />
                )}
              </div>

              <div className="flex-1 min-w-0 space-y-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="font-bold text-slate-800">
                        {circuit.name || 'Unnamed Circuit'}
                      </h3>
                      {circuit.group && (
                        <span className="px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-purple-100 text-purple-700 rounded-full">
                          {circuit.group}
                        </span>
                      )}
                      <span className="px-2 py-0.5 text-[10px] font-medium bg-slate-100 text-slate-600 rounded-md">
                        {circuit.saeSetName}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 font-mono">
                      ID: {circuit.id}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 hover:bg-slate-200"
                      onClick={() => handleEdit(circuit)}
                    >
                      <Edit2 className="h-4 w-4 text-slate-500" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 hover:bg-red-100"
                      onClick={() => setDeleteTarget(circuit)}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold uppercase tracking-tight text-slate-400">
                    Prompt
                  </label>
                  <div className="p-3 bg-white border rounded-md text-sm text-slate-700 whitespace-pre-wrap wrap-break-word font-mono leading-relaxed">
                    {circuit.prompt}
                  </div>
                </div>

                <div className="flex items-center justify-between pt-1">
                  <div className="flex items-center gap-6">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold uppercase tracking-tight text-slate-400">
                        Status
                      </span>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        {circuit.status === 'completed' ? (
                          <div className="flex items-center gap-1 text-emerald-600">
                            <Check className="h-3.5 w-3.5" />
                            <span className="text-sm font-semibold">Ready</span>
                          </div>
                        ) : circuit.status === 'failed' ? (
                          <div className="flex items-center gap-1 text-red-600">
                            <AlertCircle className="h-3.5 w-3.5" />
                            <span className="text-sm font-semibold">
                              Failed
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-1.5 text-blue-600">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            <span className="text-sm font-semibold capitalize">
                              {circuit.status} ({Math.round(circuit.progress)}%)
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                    {circuit.progressPhase && (
                      <div className="flex flex-col">
                        <span className="text-[10px] font-bold uppercase tracking-tight text-slate-400">
                          Phase
                        </span>
                        <span className="text-sm font-semibold text-slate-600">
                          {circuit.progressPhase}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] font-bold uppercase tracking-tight text-slate-400">
                      Created
                    </div>
                    <div className="text-sm text-slate-500 font-medium">
                      {new Date(circuit.createdAt).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        ))}
        {filteredCircuits.length === 0 && (
          <Card className="p-12 text-center text-slate-500">
            No Circuits found matching your criteria
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-slate-600">
            Page {page + 1} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}

      <DeleteConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete Circuit"
        description={`Are you sure you want to delete "${deleteTarget?.name || deleteTarget?.id}"? This action cannot be undone.`}
        onConfirm={() =>
          deleteTarget &&
          deleteMutation.mutate({ data: { circuitId: deleteTarget.id } })
        }
        isDeleting={deleteMutation.isPending}
      />

      <Dialog
        open={!!editTarget}
        onOpenChange={(open) => !open && setEditTarget(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Circuit</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Name
              </label>
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="Circuit name"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-slate-700 block mb-1.5">
                Group
              </label>
              <Input
                value={editGroup}
                onChange={(e) => setEditGroup(e.target.value)}
                placeholder="Group name (leave empty for ungrouped)"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditTarget(null)}>
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={updateMutation.isPending}
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function AdminPage() {
  const { data: stats } = useQuery(adminStatsQueryOptions())

  return (
    <div className="h-full overflow-y-auto bg-linear-to-br from-slate-50 to-slate-100">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">
            Admin Dashboard
          </h1>
          <p className="text-slate-500">
            Manage SAEs, SAE Sets, and Circuits
            {stats && (
              <span className="ml-2 text-sm">
                (Series:{' '}
                <code className="text-slate-700">{stats.saeSeries}</code>)
              </span>
            )}
          </p>
        </div>

        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <StatCard
              icon={Database}
              label="SAEs"
              value={stats.saeCount}
              color="emerald"
            />
            <StatCard
              icon={FolderTree}
              label="SAE Sets"
              value={stats.saeSetCount}
              color="blue"
            />
            <StatCard
              icon={GitBranch}
              label="Circuits"
              value={stats.circuitCount}
              color="purple"
            />
            <StatCard
              icon={Bookmark}
              label="Bookmarks"
              value={stats.bookmarkCount}
              color="amber"
            />
          </div>
        )}

        <Card className="p-0 overflow-hidden">
          <Tabs defaultValue="saes" className="w-full">
            <TabsList className="w-full justify-start rounded-none border-b bg-slate-50 p-0 h-auto">
              <TabsTrigger
                value="saes"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-emerald-500 data-[state=active]:bg-transparent data-[state=active]:shadow-none px-6 py-3"
              >
                <Database className="h-4 w-4 mr-2" />
                SAEs
              </TabsTrigger>
              <TabsTrigger
                value="sae-sets"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:bg-transparent data-[state=active]:shadow-none px-6 py-3"
              >
                <FolderTree className="h-4 w-4 mr-2" />
                SAE Sets
              </TabsTrigger>
              <TabsTrigger
                value="circuits"
                className="rounded-none border-b-2 border-transparent data-[state=active]:border-purple-500 data-[state=active]:bg-transparent data-[state=active]:shadow-none px-6 py-3"
              >
                <GitBranch className="h-4 w-4 mr-2" />
                Circuits
              </TabsTrigger>
            </TabsList>

            <div className="p-6">
              <TabsContent value="saes" className="mt-0">
                <SaesTab />
              </TabsContent>
              <TabsContent value="sae-sets" className="mt-0">
                <SaeSetsTab />
              </TabsContent>
              <TabsContent value="circuits" className="mt-0">
                <CircuitsTab />
              </TabsContent>
            </div>
          </Tabs>
        </Card>
      </div>
    </div>
  )
}
