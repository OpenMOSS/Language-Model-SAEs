import { createServerFn } from '@tanstack/react-start'
import { z } from 'zod'
import { transformCircuitData } from '@/utils/circuit'

export const fetchSaeSets = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`)
    const data = await response.json()
    return z.array(z.string()).parse(data)
  },
)

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
    return transformCircuitData(data)
  })
