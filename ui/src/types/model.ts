import { z } from "zod";

const FeatureNodeSchema = z.object({
  type: z.literal("feature"),
  id: z.string(),
  sae: z.string(),
  featureIndex: z.number(),
  position: z.number(),
  activation: z.number(),
  maxActivation: z.number(),
});

const LogitsNodeSchema = z.object({
  type: z.literal("logits"),
  id: z.string(),
  position: z.number(),
  tokenId: z.number(),
  activation: z.number(),
});

export const TracingNodeSchema = z.discriminatedUnion("type", [FeatureNodeSchema, LogitsNodeSchema]);

export type TracingNode = z.infer<typeof TracingNodeSchema>;

export type TracingAction =
  | {
      type: "feature";
      sae: string;
      featureIndex: number;
      position: number;
    }
  | {
      type: "logits";
      tokenId: number;
      position: number;
    };

export const ModelGenerationSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  tokenIds: z.array(z.number()),
  inputMask: z.array(z.number()),
  logits: z.object({
    logits: z.array(z.array(z.number())),
    tokens: z.array(z.array(z.instanceof(Uint8Array))),
    tokenIds: z.array(z.array(z.number())),
  }),
  saeInfo: z.array(
    z.object({
      name: z.string(),
      featureActsIndices: z.array(z.array(z.number())),
      featureActs: z.array(z.array(z.number())),
      maxFeatureActs: z.array(z.array(z.number())),
    })
  ),
});

export type ModelGeneration = z.infer<typeof ModelGenerationSchema>;

export const TracingSchema = z.object({
  node: TracingNodeSchema,
  contributors: z.array(
    z.object({
      node: TracingNodeSchema,
      attribution: z.number(),
    })
  ),
});

export type Tracing = z.infer<typeof TracingSchema>;

export const TracingOutputSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  tokenIds: z.array(z.number()),
  tracings: z.array(TracingSchema),
});

export type TracingOutput = z.infer<typeof TracingOutputSchema>;
