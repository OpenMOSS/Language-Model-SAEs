import { createFileRoute, redirect, useNavigate } from '@tanstack/react-router'
import { Plus } from 'lucide-react'
import { circuitsQueryOptions, saeSetsQueryOptions } from '@/api/circuits'
import { NewGraphDialog } from '@/components/circuits/new-graph-dialog'
import { Card } from '@/components/ui/card'

export const Route = createFileRoute('/circuits/')({
  loader: async ({ context }) => {
    const [circuits, saeSets] = await Promise.all([
      context.queryClient.fetchQuery(circuitsQueryOptions()),
      context.queryClient.ensureQueryData(saeSetsQueryOptions()),
    ])

    if (circuits.length > 0) {
      const latestCircuit = [...circuits].sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )[0]
      throw redirect({
        to: '/circuit/$id',
        params: {
          id: latestCircuit.id,
        },
      })
    }

    return { saeSets }
  },
  component: CircuitsIndexPage,
})

function CircuitsIndexPage() {
  const { saeSets } = Route.useLoaderData()
  const navigate = useNavigate()

  const handleGraphCreated = (newCircuitId: string) => {
    navigate({
      to: '/circuit/$id',
      params: { id: newCircuitId },
    })
  }

  return (
    <div className="h-full flex flex-col items-center justify-center bg-slate-50/50 p-10">
      <Card className="max-w-md w-full p-10 flex flex-col items-center text-center gap-6">
        <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center">
          <Plus className="w-8 h-8 text-slate-400" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-semibold text-slate-900">
            No circuits found
          </h2>
          <p className="text-slate-500">
            Create your first attribution graph to start exploring the
            model&apos;s behavior.
          </p>
        </div>
        <NewGraphDialog saeSets={saeSets} onGraphCreated={handleGraphCreated} />
      </Card>
    </div>
  )
}
