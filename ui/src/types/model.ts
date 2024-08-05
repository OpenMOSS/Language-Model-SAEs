import { z } from "zod";

export const ModelGenerationSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  inputMask: z.array(z.number()),
  logits: z.array(z.array(z.number())),
  logitsTokens: z.array(z.array(z.instanceof(Uint8Array))),
});

export type ModelGeneration = z.infer<typeof ModelGenerationSchema>;
