import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useMemo, useState } from 'react'
import type { CircuitData, VisState } from '@/types/circuit'
import {
  circuitQueryOptions,
  circuitsQueryOptions,
  saeSetsQueryOptions,
} from '@/api/circuits'
import { dictionariesQueryOptions } from '@/hooks/useFeatures'
import { GraphSelector } from '@/components/circuits/graph-selector'
import { NewGraphDialog } from '@/components/circuits/new-graph-dialog'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { NodeConnections } from '@/components/circuits/node-connections'
import { FeatureCardHorizontal } from '@/components/feature/feature-card-horizontal'
import { Card } from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { createRawEdgeIndex, createRawNodeIndex } from '@/utils/circuit-index'

export const Route = createFileRoute('/circuits/')({
  component: CircuitsPage,
  loader: async ({ context }) => {
    const [circuits, saeSets, dictionaries] = await Promise.all([
      context.queryClient.ensureQueryData(circuitsQueryOptions()),
      context.queryClient.ensureQueryData(saeSetsQueryOptions()),
      context.queryClient.ensureQueryData(dictionariesQueryOptions()),
    ])
    return { circuits, saeSets, dictionaries }
  },
})

function CircuitsPage() {
  const { dictionaries } = Route.useLoaderData()
  const [clickedId, setClickedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [hiddenIds, setHiddenIds] = useState<string[]>([])
  const [selectedCircuitId, setSelectedCircuitId] = useState<string>('')

  const { data: circuits = [] } = useQuery(circuitsQueryOptions())
  const { data: saeSets = [] } = useQuery(saeSetsQueryOptions())

  const {
    data: circuitData,
    isLoading: isLoadingCircuit,
    error: circuitError,
  } = useQuery({
    ...circuitQueryOptions(selectedCircuitId),
    enabled: !!selectedCircuitId,
  })

  const circuit: CircuitData | undefined = circuitData?.graphData

  const rawNodeIndex = useMemo(
    () => (circuit ? createRawNodeIndex(circuit.nodes) : null),
    [circuit],
  )

  const rawEdgeIndex = useMemo(
    () => (circuit ? createRawEdgeIndex(circuit.edges) : null),
    [circuit],
  )

  const handleGraphCreated = (circuitId: string) => {
    setSelectedCircuitId(circuitId)
    setClickedId(null)
    setHoveredId(null)
    setHiddenIds([])
  }

  const visState: VisState = useMemo(
    () => ({
      clickedId,
      hoveredId,
    }),
    [clickedId, hoveredId],
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

  const handleNodeClick = useCallback((nodeId: string) => {
    setClickedId((prev) => (prev === nodeId ? null : nodeId))
  }, [])

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  const handleCircuitSelect = (circuitId: string) => {
    setSelectedCircuitId(circuitId)
    setClickedId(null)
    setHoveredId(null)
    setHiddenIds([])
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
      <div className="pt-4 pb-6 px-20 flex justify-center items-center">
        <div className="flex justify-center items-center gap-3">
          <div className="w-[500px]">
            <GraphSelector
              circuits={circuits}
              selectedCircuitId={selectedCircuitId}
              onSelect={handleCircuitSelect}
            />
          </div>
          <NewGraphDialog
            saeSets={saeSets}
            dictionaries={dictionaries}
            onGraphCreated={handleGraphCreated}
          />
        </div>
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
        <div className="flex flex-col items-center justify-center gap-4 pt-10">
          <Spinner isAnimating={true} />
          <p className="text-gray-600">Loading circuit visualization...</p>
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
                <FeatureCardHorizontal
                  className="grow"
                  feature={featureData}
                  sampleGroups={[
                    {
                      name: 'top_activations',
                      samples: featureData?.samples ?? [],
                    },
                  ]}
                />
              )}
            </div>

            {clickedId && rawNodeIndex && rawEdgeIndex && (
              <NodeConnections
                nodeIndex={rawNodeIndex}
                edgeIndex={rawEdgeIndex}
                clickedId={clickedId}
                hoveredId={hoveredId}
                hiddenIds={hiddenIds}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
            )}
            {!clickedId && (
              <Card className="flex flex-col basis-1/2 min-w-10 gap-4 p-6 items-center justify-center text-slate-500 text-xl">
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
      ) : !selectedCircuitId ? (
        <div className="flex flex-col items-center justify-center gap-4 pt-10">
          <p className="text-gray-600">
            Select a graph or create a new one to get started.
          </p>
        </div>
      ) : null}
    </div>
  )
}
