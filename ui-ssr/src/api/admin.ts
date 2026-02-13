import { queryOptions } from '@tanstack/react-query'
import { createServerFn } from '@tanstack/react-start'
import camelcaseKeys from 'camelcase-keys'

// Types
export interface AdminSae {
  name: string
  series: string
  path: string | null
  modelName: string | null
  analysisCount: number
  featureCount: number
  cfg: Record<string, any> | null
}

export interface AdminSaeSet {
  name: string
  saeSeries: string
  saeNames: string[]
}

export interface AdminCircuit {
  id: string
  name: string | null
  group: string | null
  saeSetName: string
  saeSeries: string
  prompt: string
  config: Record<string, any>
  createdAt: string
  status: string
  progress: number
  progressPhase: string | null
  errorMessage: string | null
}

export interface AdminStats {
  saeCount: number
  saeSetCount: number
  circuitCount: number
  bookmarkCount: number
  saeSeries: string
}

// SAEs
export const fetchAdminSaes = createServerFn({ method: 'GET' })
  .inputValidator(
    (data: { limit?: number; skip?: number; search?: string }) => data,
  )
  .handler(async ({ data: { limit = 100, skip = 0, search } }) => {
    const params = new URLSearchParams({
      limit: String(limit),
      skip: String(skip),
    })
    if (search) {
      params.append('search', search)
    }
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/saes?${params}`,
    )
    if (!response.ok) {
      throw new Error(`Failed to fetch SAEs: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as {
      saes: AdminSae[]
      totalCount: number
    }
  })

export const updateSae = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: {
      name: string
      newName?: string
      path?: string
      modelName?: string
    }) => data,
  )
  .handler(async ({ data: { name, newName, path, modelName } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/saes/${encodeURIComponent(name)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_name: newName ?? null,
          path: path ?? null,
          model_name: modelName ?? null,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to update SAE: ${await response.text()}`)
    }
    return await response.json()
  })

export const deleteSae = createServerFn({ method: 'POST' })
  .inputValidator((data: { name: string }) => data)
  .handler(async ({ data: { name } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/saes/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    )

    if (!response.ok) {
      throw new Error(`Failed to delete SAE: ${await response.text()}`)
    }
    return await response.json()
  })

// SAE Sets
export const fetchAdminSaeSets = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/admin/sae-sets`)
    if (!response.ok) {
      throw new Error(`Failed to fetch SAE sets: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as AdminSaeSet[]
  },
)

export const updateSaeSet = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: { name: string; newName?: string; saeNames?: string[] }) => data,
  )
  .handler(async ({ data: { name, newName, saeNames } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/sae-sets/${encodeURIComponent(name)}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_name: newName ?? null,
          sae_names: saeNames ?? null,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to update SAE set: ${await response.text()}`)
    }
    return await response.json()
  })

export const deleteSaeSet = createServerFn({ method: 'POST' })
  .inputValidator((data: { name: string }) => data)
  .handler(async ({ data: { name } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/sae-sets/${encodeURIComponent(name)}`,
      { method: 'DELETE' },
    )

    if (!response.ok) {
      throw new Error(`Failed to delete SAE set: ${await response.text()}`)
    }
    return await response.json()
  })

// Circuits
export const fetchAdminCircuits = createServerFn({ method: 'GET' })
  .inputValidator((data: { limit?: number; skip?: number }) => data)
  .handler(async ({ data: { limit = 100, skip = 0 } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/circuits?limit=${limit}&skip=${skip}`,
    )
    if (!response.ok) {
      throw new Error(`Failed to fetch circuits: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as {
      circuits: AdminCircuit[]
      totalCount: number
    }
  })

export const updateCircuit = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: { circuitId: string; name?: string; group?: string | null }) => data,
  )
  .handler(async ({ data: { circuitId, name, group } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/circuits/${circuitId}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name ?? null,
          group: group === undefined ? undefined : group,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to update circuit: ${await response.text()}`)
    }
    return await response.json()
  })

export const bulkGroupCircuits = createServerFn({ method: 'POST' })
  .inputValidator(
    (data: { circuitIds: string[]; group: string | null }) => data,
  )
  .handler(async ({ data: { circuitIds, group } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/admin/circuits/bulk-group`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          circuit_ids: circuitIds,
          group,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to bulk group circuits: ${await response.text()}`)
    }

    return await response.json()
  })

export const deleteAdminCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: { circuitId: string }) => data)
  .handler(async ({ data: { circuitId } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits/${circuitId}`,
      { method: 'DELETE' },
    )

    if (!response.ok) {
      throw new Error(`Failed to delete circuit: ${await response.text()}`)
    }
    return await response.json()
  })

// Stats
export const fetchAdminStats = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/admin/stats`)
    if (!response.ok) {
      throw new Error(`Failed to fetch stats: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as AdminStats
  },
)

// Query options
export const adminSaesQueryOptions = (limit = 100, skip = 0, search?: string) =>
  queryOptions({
    queryKey: ['admin', 'saes', { limit, skip, search }],
    queryFn: () => fetchAdminSaes({ data: { limit, skip, search } }),
  })

export const adminSaeSetsQueryOptions = () =>
  queryOptions({
    queryKey: ['admin', 'sae-sets'],
    queryFn: () => fetchAdminSaeSets(),
  })

export const adminCircuitsQueryOptions = (limit = 100, skip = 0) =>
  queryOptions({
    queryKey: ['admin', 'circuits', { limit, skip }],
    queryFn: () => fetchAdminCircuits({ data: { limit, skip } }),
  })

export const adminStatsQueryOptions = () =>
  queryOptions({
    queryKey: ['admin', 'stats'],
    queryFn: () => fetchAdminStats(),
  })
