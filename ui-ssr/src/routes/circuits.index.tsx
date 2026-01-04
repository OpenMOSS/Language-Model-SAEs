import { createFileRoute, redirect } from '@tanstack/react-router'
import { circuitsQueryOptions } from '@/api/circuits'

export const Route = createFileRoute('/circuits/')({
  beforeLoad: async ({ context }) => {
    const circuits = await context.queryClient.ensureQueryData(
      circuitsQueryOptions(),
    )
    if (circuits.length === 0) {
      return
    }
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
  },
})
