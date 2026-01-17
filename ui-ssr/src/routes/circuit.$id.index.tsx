import {
  createFileRoute,
  redirect,
  useNavigate,
  useRouter,
} from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, Loader2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { z } from 'zod'
import type { CircuitData, FeatureNode, VisState } from '@/types/circuit'
import {
  circuitQueryOptions,
  circuitStatusQueryOptions,
  circuitsQueryOptions,
  generateCircuit,
  saeSetsQueryOptions,
} from '@/api/circuits'
import { GraphSelector } from '@/components/circuits/graph-selector'
import { NewGraphDialog } from '@/components/circuits/new-graph-dialog'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { SelectedFeaturesList } from '@/components/circuits/selected-features-list'
import { NodeConnections } from '@/components/circuits/node-connections'
import { ThresholdControls } from '@/components/circuits/threshold-controls'
import { FeatureCardHorizontal } from '@/components/feature/feature-card-horizontal'
import { Card } from '@/components/ui/card'
import { createRawEdgeIndex, createRawNodeIndex } from '@/utils/circuit-index'

const searchParamsSchema = z.object({
  clickedId: z.string().optional(),
  hiddenIds: z.string().optional(),
  selectedIds: z.string().optional(),
  nodeThreshold: z.coerce.number().optional(),
  edgeThreshold: z.coerce.number().optional(),
})

export const Route = createFileRoute('/circuit/$id/')({
  validateSearch: searchParamsSchema,
  staticData: {
    fullScreen: true,
  },
  component: CircuitPage,
  loader: async ({ context, params }) => {
    const [circuits, saeSets] = await Promise.all([
      context.queryClient.fetchQuery(circuitsQueryOptions()),
      context.queryClient.ensureQueryData(saeSetsQueryOptions()),
    ])
    if (!circuits.find((c) => c.id === params.id)) {
      console.log('redirecting to circuits')
      throw redirect({
        to: '/circuits',
      })
    }
    return { circuits, saeSets, circuitId: params.id }
  },
  gcTime: 0,
})

function CircuitPage() {
  const navigate = useNavigate()
  const { circuits, saeSets, circuitId } = Route.useLoaderData()
  const search = Route.useSearch()
  const router = useRouter()

  const clickedId = search.clickedId || null
  const hiddenIds = useMemo(
    () => (search.hiddenIds ? search.hiddenIds.split(',') : []),
    [search.hiddenIds],
  )
  const selectedIds = useMemo(
    () => (search.selectedIds ? search.selectedIds.split(',') : []),
    [search.selectedIds],
  )
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  // Threshold state from URL params
  const nodeThreshold = search.nodeThreshold ?? 0.8
  const edgeThreshold = search.edgeThreshold ?? 0.98

  // Query circuit status first
  const { data: statusData, isLoading: isLoadingStatus } = useQuery(
    circuitStatusQueryOptions(circuitId),
  )

  const isCircuitReady = statusData?.status === 'completed'
  const isCircuitPending = statusData?.status === 'pending'
  const isCircuitRunning = statusData?.status === 'running'
  const isCircuitFailed = statusData?.status === 'failed'
  const isCircuitInProgress = isCircuitPending || isCircuitRunning

  // Only fetch circuit data when it's ready
  const {
    data: circuitData,
    isLoading: isLoadingCircuit,
    isFetching: isFetchingCircuit,
    error: circuitError,
  } = useQuery({
    ...circuitQueryOptions(circuitId, nodeThreshold, edgeThreshold),
    enabled: isCircuitReady,
  })

  const queryClient = useQueryClient()

  // Invalidate circuits list when status changes to completed
  useEffect(() => {
    queryClient.invalidateQueries({ queryKey: ['circuits'] })
    router.invalidate()
  }, [statusData?.status, queryClient])

  const { mutate: doGenerateCircuit, isPending: isGenerating } = useMutation({
    mutationFn: generateCircuit,
    onSuccess: async (data) => {
      handleGraphCreated(data.circuitId)
      updateSearchParams({ selectedIds: [] })
    },
  })

  const updateSearchParams = useCallback(
    (updates: {
      clickedId?: string | null
      hiddenIds?: string[]
      selectedIds?: string[]
      nodeThreshold?: number
      edgeThreshold?: number
    }) => {
      navigate({
        to: '/circuit/$id',
        params: { id: circuitId },
        search: (prev) => ({
          ...prev,
          clickedId:
            updates.clickedId === null
              ? undefined
              : (updates.clickedId ?? prev.clickedId),
          hiddenIds:
            updates.hiddenIds !== undefined
              ? updates.hiddenIds.length > 0
                ? updates.hiddenIds.join(',')
                : undefined
              : prev.hiddenIds,
          selectedIds:
            updates.selectedIds !== undefined
              ? updates.selectedIds.length > 0
                ? updates.selectedIds.join(',')
                : undefined
              : prev.selectedIds,
          nodeThreshold: updates.nodeThreshold ?? prev.nodeThreshold,
          edgeThreshold: updates.edgeThreshold ?? prev.edgeThreshold,
        }),
        replace: true,
      })
    },
    [navigate, circuitId],
  )

  const handleThresholdsChange = useCallback(
    (newNodeThreshold: number, newEdgeThreshold: number) => {
      updateSearchParams({
        nodeThreshold: newNodeThreshold,
        edgeThreshold: newEdgeThreshold,
      })
    },
    [updateSearchParams],
  )

  const handleClearSelected = useCallback(() => {
    updateSearchParams({ selectedIds: [] })
  }, [updateSearchParams])

  const handleRemoveSelected = useCallback(
    (idToRemove: string) => {
      updateSearchParams({
        selectedIds: selectedIds.filter((id) => id !== idToRemove),
      })
    },
    [selectedIds, updateSearchParams],
  )

  const handleTraceSelected = useCallback(() => {
    if (!circuitData || selectedIds.length === 0) return

    const circuit = circuitData.graphData
    const listOfFeatures: (number | boolean)[][] = selectedIds.map((id) => {
      const node = circuit.nodes.find((n) => n.nodeId === id) as FeatureNode
      return [
        Math.floor(node.layer / 2),
        node.feature.featureIndex,
        node.ctxIdx,
        node.featureType === 'lorsa',
      ]
    })

    doGenerateCircuit({
      data: {
        saeSetName: circuitData.saeSetName,
        input: circuitData.input,
        desiredLogitProb: circuitData.config.desiredLogitProb,
        maxFeatureNodes: circuitData.config.maxFeatureNodes,
        maxNLogits: circuitData.config.maxNLogits,
        qkTracingTopk: circuitData.config.qkTracingTopk,
        listOfFeatures,
        parentId: circuitId,
      },
    })
  }, [circuitData, selectedIds, doGenerateCircuit, circuitId])

  const circuit: CircuitData | undefined = circuitData?.graphData

  const rawNodeIndex = useMemo(
    () => (circuit ? createRawNodeIndex(circuit.nodes) : null),
    [circuit],
  )

  const rawEdgeIndex = useMemo(
    () => (circuit ? createRawEdgeIndex(circuit.edges) : null),
    [circuit],
  )

  const handleGraphCreated = (newCircuitId: string) => {
    queryClient.invalidateQueries({ queryKey: ['circuits'] })
    router.invalidate()

    router.navigate({
      to: '/circuit/$id',
      params: { id: newCircuitId },
    })
    setHoveredId(null)
  }

  const visState: VisState = useMemo(
    () => ({
      clickedId,
      hoveredId,
      selectedIds,
    }),
    [clickedId, hoveredId, selectedIds],
  )

  const featureData = useMemo(() => {
    if (!clickedId || !circuit) return null

    const node = circuit.nodes.find((n) => n.nodeId === clickedId)
    if (
      !node ||
      (node.featureType !== 'cross layer transcoder' &&
        node.featureType !== 'lorsa')
    )
      return null

    return node.feature
  }, [clickedId, circuit])

  const handleNodeClick = useCallback(
    (nodeId: string, isMultiSelect: boolean) => {
      if (isMultiSelect) {
        if (!circuit) return
        const node = circuit.nodes.find((n) => n.nodeId === nodeId)
        if (
          !node ||
          (node.featureType !== 'cross layer transcoder' &&
            node.featureType !== 'lorsa')
        )
          return

        const newSelectedIds = selectedIds.includes(nodeId)
          ? selectedIds.filter((id) => id !== nodeId)
          : [...selectedIds, nodeId]
        updateSearchParams({ selectedIds: newSelectedIds })
      } else {
        const newClickedId = clickedId === nodeId ? null : nodeId
        updateSearchParams({ clickedId: newClickedId })
      }
    },
    [circuit, clickedId, selectedIds, updateSearchParams],
  )

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  const handleCircuitSelect = (selectedCircuitId: string) => {
    router.navigate({
      to: '/circuit/$id',
      params: { id: selectedCircuitId },
      search: {
        // Reset thresholds when switching circuits
        nodeThreshold: 0.8,
        edgeThreshold: 0.98,
      },
    })
    setHoveredId(null)
  }

  // Render loading state while checking status
  if (isLoadingStatus) {
    return (
      <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
        <div className="pt-4 pb-6 px-20 flex items-center">
          <div className="flex-1" />
          <div className="flex justify-center items-center gap-3">
            <div className="w-[500px]">
              <GraphSelector
                circuits={circuits}
                selectedCircuitId={circuitId}
                onSelect={handleCircuitSelect}
              />
            </div>
            <NewGraphDialog
              saeSets={saeSets}
              onGraphCreated={handleGraphCreated}
            />
          </div>
          <div className="flex-1" />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Loading Circuit
                </h3>
                <p className="text-sm text-slate-500">Checking status...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render in-progress state
  if (isCircuitInProgress) {
    return (
      <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
        <div className="pt-4 pb-6 px-20 flex items-center">
          <div className="flex-1" />
          <div className="flex justify-center items-center gap-3">
            <div className="w-[500px]">
              <GraphSelector
                circuits={circuits}
                selectedCircuitId={circuitId}
                onSelect={handleCircuitSelect}
              />
            </div>
            <NewGraphDialog
              saeSets={saeSets}
              onGraphCreated={handleGraphCreated}
            />
          </div>
          <div className="flex-1" />
        </div>
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>

            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Generating Circuit
                </h3>
                <p className="text-sm text-slate-500">
                  This may take a few minutes depending on the complexity
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium text-slate-700">
                  <span>{statusData?.progressPhase || 'Initializing...'}</span>
                  <span>{Math.round(statusData?.progress ?? 0)}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-600 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${statusData?.progress ?? 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render failed state
  if (isCircuitFailed) {
    return (
      <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
        <div className="pt-4 pb-6 px-20 flex items-center">
          <div className="flex-1" />
          <div className="flex justify-center items-center gap-3">
            <div className="w-[500px]">
              <GraphSelector
                circuits={circuits}
                selectedCircuitId={circuitId}
                onSelect={handleCircuitSelect}
              />
            </div>
            <NewGraphDialog
              saeSets={saeSets}
              onGraphCreated={handleGraphCreated}
            />
          </div>
          <div className="flex-1" />
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="flex justify-center mb-4">
              <AlertCircle className="h-12 w-12 text-red-500" />
            </div>
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Circuit Generation Failed
            </h3>
            <p className="text-gray-600 max-w-md">
              {statusData?.errorMessage || 'An unknown error occurred'}
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
      <div className="pt-4 pb-6 px-20 flex items-center">
        <div className="flex-1">
          <ThresholdControls
            nodeThreshold={nodeThreshold}
            edgeThreshold={edgeThreshold}
            onThresholdsChange={handleThresholdsChange}
            isLoading={isFetchingCircuit}
          />
        </div>
        <div className="flex justify-center items-center gap-3">
          <div className="w-[500px]">
            <GraphSelector
              circuits={circuits}
              selectedCircuitId={circuitId}
              onSelect={handleCircuitSelect}
            />
          </div>
          <NewGraphDialog
            saeSets={saeSets}
            onGraphCreated={handleGraphCreated}
          />
        </div>
        <div className="flex-1" />
      </div>

      {circuitError ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Failed to load circuit visualization
            </h3>
            <p className="text-gray-600">
              {circuitError.message || 'Failed to load circuit'}
            </p>
          </div>
        </div>
      ) : isLoadingCircuit ? (
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Loading Visualization
                </h3>
                <p className="text-sm text-slate-500">Fetching graph data...</p>
              </div>
            </div>
          </div>
        </div>
      ) : circuit ? (
        <div className="flex-1 flex flex-col overflow-hidden px-20 pb-20">
          <div className="flex gap-6 flex-1 overflow-hidden">
            <div className="flex flex-col gap-6 min-w-0 w-3/4 shrink-0">
              <LinkGraphContainer
                data={circuit}
                visState={visState}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
              {featureData && (
                <FeatureCardHorizontal className="grow" feature={featureData} />
              )}
            </div>

            <div className="flex flex-col flex-1 gap-4 min-w-0 h-full overflow-hidden">
              {clickedId && rawNodeIndex && rawEdgeIndex && (
                <NodeConnections
                  nodeIndex={rawNodeIndex}
                  edgeIndex={rawEdgeIndex}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  hiddenIds={hiddenIds}
                  onNodeClick={handleNodeClick}
                  onNodeHover={handleNodeHover}
                  className={
                    selectedIds.length > 0
                      ? 'flex-[3] min-h-0'
                      : 'flex-1 min-h-0'
                  }
                />
              )}
              {selectedIds.length > 0 && circuit && (
                <SelectedFeaturesList
                  className={clickedId ? 'flex-[2] min-h-0' : 'flex-1 min-h-0'}
                  selectedIds={selectedIds}
                  circuit={circuit}
                  isGenerating={isGenerating}
                  onRemove={handleRemoveSelected}
                  onClear={handleClearSelected}
                  onTrace={handleTraceSelected}
                />
              )}
              {!clickedId && selectedIds.length === 0 && (
                <Card className="flex flex-col h-full gap-4 p-6 items-center justify-center text-slate-500 text-xl">
                  <div className="flex flex-col items-center justify-center">
                    <span>No node selected</span>
                    <span className="text-sm text-slate-500">
                      Click a node on the left to see its connections
                    </span>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
