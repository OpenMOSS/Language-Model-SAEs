import { queryOptions } from '@tanstack/react-query'
import { createServerFn } from '@tanstack/react-start'
import camelcaseKeys from 'camelcase-keys'
import { z } from 'zod'
import type { CircuitData } from '@/types/circuit'

export interface CircuitConfig {
  desiredLogitProb: number
  maxFeatureNodes: number
  qkTracingTopk: number
  nodeThreshold: number
  edgeThreshold: number
  maxNLogits: number
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface PlainTextInput {
  inputType: 'plain_text'
  text: string
}

export interface ChatTemplateInput {
  inputType: 'chat_template'
  messages: ChatMessage[]
}

export type CircuitInput = PlainTextInput | ChatTemplateInput

export interface CircuitListItem {
  id: string
  name: string | null
  saeSetName: string
  saeSetSeries: string
  prompt: string
  input: CircuitInput
  config: CircuitConfig
  createdAt: string
}

export interface GenerateCircuitParams {
  saeSetName: string
  input: CircuitInput
  name?: string
  desiredLogitProb?: number
  maxFeatureNodes?: number
  qkTracingTopk?: number
  nodeThreshold?: number
  edgeThreshold?: number
  maxNLogits?: number
}

export const fetchSaeSets = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`)
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

export const previewInput = createServerFn({ method: 'POST' })
  .inputValidator((data: { saeSetName: string; input: CircuitInput }) => data)
  .handler(async ({ data: { saeSetName, input } }) => {
    // Convert input to backend format
    const backendInput =
      input.inputType === 'plain_text'
        ? { input_type: 'plain_text', text: input.text }
        : {
            input_type: 'chat_template',
            messages: input.messages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }

    const response = await fetch(`${process.env.BACKEND_URL}/preview`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sae_set_name: saeSetName,
        input: backendInput,
      }),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Failed to preview input')
    }

    const result = await response.json()
    return camelcaseKeys(result, { deep: true }) as {
      prompt: string
      nextTokens: { token: string; prob: number }[]
    }
  })

export const fetchCircuits = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/circuits`)
    if (!response.ok) {
      throw new Error(`Failed to fetch circuits: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as CircuitListItem[]
  },
)

export const fetchCircuit = createServerFn({ method: 'GET' })
  .inputValidator((data: { circuitId: string }) => data)
  .handler(async ({ data: { circuitId } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits/${circuitId}`,
    )

    if (!response.ok) {
      throw new Error(`Failed to fetch circuit: ${await response.text()}`)
    }

    const data = await response.json()

    const transformedData = {
      ...data,
      graphData: {
        ...data.graph_data,
        edges: data.graph_data.links,
        links: undefined,
      },
    }
    delete transformedData.graph_data

    const result = camelcaseKeys(transformedData, {
      deep: true,
    }) as {
      circuitId: string
      name: string | null
      saeSetName: string
      prompt: string
      input: CircuitInput
      config: CircuitConfig
      graphData: CircuitData
      createdAt: string
    }

    return result
  })

export const generateCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: GenerateCircuitParams) => data)
  .handler(async ({ data }) => {
    const {
      saeSetName,
      input,
      name,
      desiredLogitProb,
      maxFeatureNodes,
      qkTracingTopk,
      nodeThreshold,
      edgeThreshold,
      maxNLogits,
    } = data

    // Convert input to backend format
    const backendInput =
      input.inputType === 'plain_text'
        ? { input_type: 'plain_text', text: input.text }
        : {
            input_type: 'chat_template',
            messages: input.messages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }

    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits?sae_set_name=${encodeURIComponent(saeSetName)}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: backendInput,
          name: name || null,
          desired_logit_prob: desiredLogitProb,
          max_feature_nodes: maxFeatureNodes,
          qk_tracing_topk: qkTracingTopk,
          node_threshold: nodeThreshold,
          edge_threshold: edgeThreshold,
          max_n_logits: maxNLogits,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to generate circuit: ${await response.text()}`)
    }

    const result = await response.json()

    const transformedData = {
      circuitId: result.circuit_id,
      graphData: {
        ...result.graph_data,
        edges: result.graph_data.links,
        links: undefined,
      },
    }

    return camelcaseKeys(transformedData, {
      deep: true,
    }) as {
      circuitId: string
      graphData: CircuitData
      createdAt: string
      name: string | null
      saeSetName: string
      prompt: string
      input: CircuitInput
      config: CircuitConfig
    }
  })

export const deleteCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: { circuitId: string }) => data)
  .handler(async ({ data: { circuitId } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits/${circuitId}`,
      {
        method: 'DELETE',
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to delete circuit: ${await response.text()}`)
    }

    return await response.json()
  })

export const saeSetsQueryOptions = () =>
  queryOptions({
    queryKey: ['sae-sets'],
    queryFn: () => fetchSaeSets(),
  })

export const circuitsQueryOptions = () =>
  queryOptions({
    queryKey: ['circuits'],
    queryFn: () => fetchCircuits(),
  })

export const circuitQueryOptions = (circuitId: string) =>
  queryOptions({
    queryKey: ['circuit', circuitId],
    queryFn: () => fetchCircuit({ data: { circuitId } }),
    enabled: !!circuitId,
  })
