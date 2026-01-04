import { createServerFn } from '@tanstack/react-start'
import { z } from 'zod'
import camelcaseKeys from 'camelcase-keys'
import type { CircuitData } from '@/types/circuit'

export const fetchSaeSets = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`)
    const data = await response.json()
    return z.array(z.string()).parse(data)
  },
)

export const fetchAvailableSaes = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/dictionaries`)
    const data = await response.json()
    return z.array(z.string()).parse(data)
  },
)

export const createSaeSet = createServerFn({ method: 'POST' })
  .inputValidator((data: { name: string; saeNames: string[] }) => data)
  .handler(async ({ data: { name, saeNames } }) => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name,
        sae_names: saeNames,
      }),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Failed to create SAE set')
    }

    return await response.json()
  })

export const traceCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: { saeSetName: string; text: string }) => data)
  .handler(async ({ data: { saeSetName, text } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuit/${saeSetName}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to trace circuit: ${await response.text()}`)
    }

    const data = await response.json()

    // Rename links to edges
    const transformedData = {
      ...data,
      edges: data.links,
      links: undefined,
    }

    return camelcaseKeys(transformedData, {
      deep: true,
    }) as CircuitData
  })
