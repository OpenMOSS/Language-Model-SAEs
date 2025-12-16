import { useMutation, useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useMemo, useState } from 'react'
import type { CircuitData, VisState } from '@/types/circuit'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { NodeConnections } from '@/components/circuits/node-connections'
import { FeatureCard } from '@/components/feature/feature-card'
import { featureQueryOptions } from '@/hooks/useFeatures'
import { extractLayerAndFeature } from '@/utils/circuit'
import { fetchSaeSets, traceCircuit } from '@/api/circuits'
import { LabeledSelect } from '@/components/ui/labeled-select'
import { Button } from '@/components/ui/button'

export const Route = createFileRoute('/circuits/')({
  component: CircuitsPage,
  loader: async () => {
    const saeSets = await fetchSaeSets()
    return { saeSets }
  },
})

function CircuitsPage() {
  const { saeSets } = Route.useLoaderData()
  const [error, setError] = useState<string | null>(null)
  const [clickedId, setClickedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [hiddenIds, setHiddenIds] = useState<string[]>([])

  const [selectedSaeSet, setSelectedSaeSet] = useState<string>(saeSets[0])
  const [text, setText] = useState<string>('')

  const {
    mutate: mutateTraceCircuit,
    isPending,
    data: circuit,
  } = useMutation({
    mutationFn: traceCircuit,
    onSuccess: (data) => {
      setClickedId(null)
      setHoveredId(null)
      setHiddenIds([])
      setError(null)
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : 'Failed to trace circuit')
    },
  })

  const handleTrace = () => {
    if (!text) return
    mutateTraceCircuit({ data: { saeSetName: selectedSaeSet, text } })
  }

  const visState: VisState = useMemo(
    () => ({
      clickedId,
      hoveredId,
    }),
    [clickedId, hoveredId],
  )

  const featureQueryParams = useMemo(() => {
    if (!clickedId || !circuit) return null

    const node = circuit.nodes.find((n) => n.nodeId === clickedId)
    if (
      !node ||
      (node.featureType !== 'cross layer transcoder' &&
        node.featureType !== 'lorsa')
    )
      return null

    const layerAndFeature = extractLayerAndFeature(clickedId)
    if (!layerAndFeature) return null

    const { featureId } = layerAndFeature
    // Use saeName directly from node data
    const dictionaryName = node.saeName
    if (!dictionaryName) return null

    return { dictionary: dictionaryName, featureIndex: featureId }
  }, [clickedId, circuit])

  const featureQuery = useQuery({
    ...featureQueryOptions(
      featureQueryParams ?? { dictionary: '', featureIndex: 0 },
    ),
    enabled: featureQueryParams !== null,
  })

  const handleNodeClick = useCallback((nodeId: string) => {
    setClickedId((prev) => (prev === nodeId ? null : nodeId))
  }, [])

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  return (
    <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
      <div className="pt-4 pb-6 px-20 flex justify-center items-center">
        <div className="flex justify-center items-center gap-3">
          <div className="w-[300px]">
            <LabeledSelect
              label="SAE Set"
              placeholder="Select an SAE set"
              value={selectedSaeSet}
              onValueChange={setSelectedSaeSet}
              options={saeSets.map((s) => ({ value: s, label: s }))}
              triggerClassName="bg-white w-full"
            />
          </div>
          <div className="w-[600px]">
            <div className="h-12 px-3 bg-white border border-input rounded-md flex flex-col justify-center">
              <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none">
                Prompt
              </span>
              <input
                placeholder="Enter text to trace..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isPending && text) {
                    handleTrace()
                  }
                }}
                className="font-medium leading-none bg-transparent border-none outline-none focus:outline-none focus:ring-0 p-0 placeholder:text-muted-foreground"
              />
            </div>
          </div>
          <Button
            onClick={handleTrace}
            disabled={!text || isPending}
            className="h-12 px-4"
          >
            {isPending ? 'Tracing...' : 'Trace'}
          </Button>
        </div>
      </div>

      {error ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Failed to load circuit visualization
            </h3>
            <p className="text-gray-600">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Try Again
            </button>
          </div>
        </div>
      ) : isPending ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading circuit visualization...</p>
          </div>
        </div>
      ) : circuit ? (
        <div className="flex-1 flex flex-col overflow-hidden px-20 pb-20">
          <div className="flex gap-6 flex-1 overflow-hidden">
            <div className="flex-1 min-w-0 overflow-hidden">
              <LinkGraphContainer
                data={circuit}
                visState={visState}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
            </div>

            <div className="w-96 shrink-0 border border-slate-200 bg-white overflow-hidden">
              <NodeConnections
                data={circuit}
                clickedId={clickedId}
                hoveredId={hoveredId}
                hiddenIds={hiddenIds}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
            </div>
          </div>

          {clickedId && (
            <div className="mt-6 border border-slate-200 bg-white p-6">
              <h3 className="text-lg font-semibold mb-4">
                Selected Feature Details
              </h3>
              {featureQuery.isPending && featureQueryParams ? (
                <div className="flex items-center justify-center p-8">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                    <p className="text-gray-600">Loading feature...</p>
                  </div>
                </div>
              ) : featureQuery.data ? (
                <FeatureCard feature={featureQuery.data} />
              ) : (
                <div className="flex items-center justify-center p-8">
                  <div className="text-center">
                    <p className="text-gray-600">
                      No feature is available for this node
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
