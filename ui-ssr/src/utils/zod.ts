import { z } from 'zod'

export function parseWithPrettify<T>(schema: z.ZodType<T>, data: unknown): T {
  try {
    return schema.parse(data)
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Use built-in prettifyError from Zod 4
      const message = z.prettifyError(error)
      throw new Error(message)
    }
    throw error
  }
}
