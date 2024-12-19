import { z } from "zod";

export const TextTokenOriginSchema = z.object({
  key: z.literal("text"),
  range: z.tuple([z.number(), z.number()]),
});

export type TextTokenOrigin = z.infer<typeof TextTokenOriginSchema>;

export const ImageTokenOriginSchema = z.object({
  key: z.literal("image"),
  imageIndex: z.number(),
  rect: z.tuple([z.number(), z.number(), z.number(), z.number()]),
});

export type ImageTokenOrigin = z.infer<typeof ImageTokenOriginSchema>;

export const TokenOriginSchema = z.union([TextTokenOriginSchema, ImageTokenOriginSchema]);

export type TokenOrigin = z.infer<typeof TokenOriginSchema>;

export const FeatureSampleCompactSchema = z.object({
  text: z.string().nullish(),
  images: z.array(z.string()).nullish(),
  origins: z.array(TokenOriginSchema.nullable()),
  featureActs: z.array(z.number()),
});

export type FeatureSampleCompact = z.infer<typeof FeatureSampleCompactSchema>;

export const InterpretationSchema = z.object({
  text: z.string(),
  validation: z.array(
    z.object({
      method: z.string(),
      passed: z.boolean(),
      detail: z
        .object({
          prompt: z.string(),
          response: z.string(),
        })
        .optional(),
    })
  ),
  detail: z
    .object({
      prompt: z.string(),
      response: z.string(),
    })
    .optional(),
});

export type Interpretation = z.infer<typeof InterpretationSchema>;

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  dictionaryName: z.string(),
  featureActivationHistogram: z.any().nullable(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  sampleGroups: z.array(
    z.object({
      analysisName: z.string(),
      samples: z.array(FeatureSampleCompactSchema),
    })
  ),
  logits: z
    .object({
      topPositive: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      topNegative: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      histogram: z.any(),
    })
    .nullable(),
  interpretation: InterpretationSchema.nullable(),
});

export type Feature = z.infer<typeof FeatureSchema>;
