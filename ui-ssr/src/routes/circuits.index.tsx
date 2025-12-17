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
import { LabeledInput } from '@/components/ui/labeled-input'
import { Spinner } from '@/components/ui/spinner'

export const Route = createFileRoute('/circuits/')({
  component: CircuitsPage,
  loader: async () => {
    const saeSets = await fetchSaeSets()
    return { saeSets }
  },
})

function CircuitsPage() {
  const { saeSets } = Route.useLoaderData()
  const [clickedId, setClickedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [hiddenIds, setHiddenIds] = useState<string[]>([])

  const [selectedSaeSet, setSelectedSaeSet] = useState<string>(saeSets[0])
  const [text, setText] = useState<string>('')

  const {
    mutate: mutateTraceCircuit,
    isPending,
    data: circuit,
    error,
  } = useMutation({
    mutationFn: traceCircuit,
    onSuccess: (data) => {
      setClickedId(null)
      setHoveredId(null)
      setHiddenIds([])
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
            <LabeledInput
              label="Prompt"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isPending && text) {
                  handleTrace()
                }
              }}
              placeholder="Enter your prompt to trace..."
            />
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
            <p className="text-gray-600">
              {error.message || 'Failed to trace circuit'}
            </p>
          </div>
        </div>
      ) : isPending ? (
        <div className="flex flex-col items-center justify-center gap-4 pt-10">
          <Spinner isAnimating={true} />
          <p className="text-gray-600">Loading circuit visualization...</p>
        </div>
      ) : circuit ? (
        <div className="flex-1 flex flex-col overflow-hidden px-20 pb-20">
          <div className="flex gap-6 flex-1 overflow-hidden">
            <LinkGraphContainer
              data={circuit}
              visState={visState}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
            />

            <div className="w-[500px] shrink-0 border border-slate-200 bg-white overflow-hidden">
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

          {/* {clickedId && (
            <div className="mt-6 border border-slate-200 bg-white p-6">
              <h3 className="text-lg font-semibold mb-4">
                Selected Feature Details
              </h3>
              {featureQuery.isPending && featureQueryParams ? (
                <div className="flex items-center justify-center p-8">
                  <div className="text-center">
                    <Spinner isAnimating={true} />
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
          )} */}
        </div>
      ) : null}
    </div>
  )
}
