import { useQuery } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useMemo, useState } from 'react'
import type { CircuitData, CircuitJsonData, VisState } from '@/types/circuit'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { NodeConnections } from '@/components/circuits/node-connections'
import { FeatureCard } from '@/components/feature/feature-card'
import { featureQueryOptions } from '@/hooks/useFeatures'
import {
  extractLayerAndFeature,
  getDictionaryName,
  transformCircuitData,
} from '@/utils/circuit'

export const Route = createFileRoute('/circuits/')({
  component: CircuitsPage,
})

function CircuitsPage() {
  const [circuitData, setCircuitData] = useState<CircuitData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clickedId, setClickedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [pinnedIds, setPinnedIds] = useState<string[]>([])
  const [hiddenIds, setHiddenIds] = useState<string[]>([])
  const [isDragOver, setIsDragOver] = useState(false)

  const visState: VisState = useMemo(
    () => ({
      pinnedIds,
      clickedId,
      hoveredId,
    }),
    [pinnedIds, clickedId, hoveredId],
  )

  const featureQueryParams = useMemo(() => {
    if (!clickedId || !circuitData) return null

    const node = circuitData.nodes.find((n) => n.nodeId === clickedId)
    if (!node) return null

    if (
      node.featureType !== 'cross layer transcoder' &&
      node.featureType !== 'lorsa'
    ) {
      return null
    }

    const layerAndFeature = extractLayerAndFeature(clickedId)
    if (!layerAndFeature) return null

    const { layer, featureId, isLorsa } = layerAndFeature
    const dictionaryName = getDictionaryName(
      circuitData.metadata,
      layer,
      isLorsa,
    )
    if (!dictionaryName) return null

    return { dictionary: dictionaryName, featureIndex: featureId }
  }, [clickedId, circuitData])

  const featureQuery = useQuery({
    ...featureQueryOptions(
      featureQueryParams ?? { dictionary: '', featureIndex: 0 },
    ),
    enabled: featureQueryParams !== null,
  })

  const handleNodeClick = useCallback((nodeId: string, metaKey: boolean) => {
    if (metaKey) {
      setPinnedIds((prev) =>
        prev.includes(nodeId)
          ? prev.filter((id) => id !== nodeId)
          : [...prev, nodeId],
      )
    } else {
      setClickedId((prev) => (prev === nodeId ? null : nodeId))
    }
  }, [])

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith('.json')) {
      setError('Please upload a JSON file')
      return
    }

    try {
      setIsLoading(true)
      setError(null)

      const text = await file.text()
      const jsonData: CircuitJsonData = JSON.parse(text)
      const data = transformCircuitData(jsonData)
      setCircuitData(data)
      setClickedId(null)
      setHoveredId(null)
      setPinnedIds([])
      setHiddenIds([])
    } catch (err) {
      console.error('Failed to load circuit data:', err)
      setError(
        err instanceof Error ? err.message : 'Failed to load circuit data',
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)

      const files = e.dataTransfer.files
      if (files.length > 0) {
        handleFileUpload(files[0])
      }
    },
    [handleFileUpload],
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files && files.length > 0) {
        handleFileUpload(files[0])
      }
    },
    [handleFileUpload],
  )

  return (
    <div className="h-full overflow-y-auto pt-4 pb-20 px-8 flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Circuit Tracing</h1>
        <p className="text-gray-600 mt-2">
          Upload circuit data and click on nodes to view detailed feature
          information.
        </p>
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
      ) : isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading circuit visualization...</p>
          </div>
        </div>
      ) : !circuitData ? (
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragOver
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
              <svg
                className="w-8 h-8 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Upload Circuit Data
              </h3>
              <p className="text-gray-600 mb-4">
                Drag and drop a JSON file here, or click to browse
              </p>
              <input
                type="file"
                accept=".json"
                onChange={handleFileInput}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 cursor-pointer transition-colors"
              >
                Choose File
              </label>
            </div>
            <p className="text-sm text-gray-500">
              Supports JSON files with circuit visualization data
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-6 w-full max-w-full overflow-hidden">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <h2 className="text-l font-bold">Prompt:</h2>
              <h2 className="text-l">
                {circuitData.metadata.promptTokens.join(' ')}
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCircuitData(null)}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
              >
                Upload New File
              </button>
            </div>
          </div>

          <div className="space-y-6 w-full max-w-full overflow-hidden">
            <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
              <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                <div className="w-full h-full overflow-hidden relative">
                  <LinkGraphContainer
                    data={circuitData}
                    visState={visState}
                    onNodeClick={handleNodeClick}
                    onNodeHover={handleNodeHover}
                  />
                </div>
              </div>

              <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                <NodeConnections
                  data={circuitData}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  pinnedIds={pinnedIds}
                  hiddenIds={hiddenIds}
                  onNodeClick={handleNodeClick}
                  onNodeHover={handleNodeHover}
                />
              </div>
            </div>

            {clickedId && (
              <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
                <h3 className="text-lg font-semibold mb-4">
                  Selected Feature Details
                </h3>
                {featureQuery.isPending && featureQueryParams ? (
                  <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                      <p className="text-gray-600">Loading feature...</p>
                    </div>
                  </div>
                ) : featureQuery.data ? (
                  <FeatureCard feature={featureQuery.data} />
                ) : (
                  <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
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
        </div>
      )}
    </div>
  )
}
