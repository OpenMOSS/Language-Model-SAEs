import { z } from "zod";

export const SampleSchema = z.object({
  context: z.array(z.instanceof(Uint8Array)),
  featureActs: z.array(z.number()),
});

export type Sample = z.infer<typeof SampleSchema>;

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  dictionaryName: z.string(),
  featureActivationHistogram: z.any(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  sampleGroups: z.array(
    z.object({
      analysisName: z.string(),
      samples: z.array(SampleSchema),
    })
  ),
});

export type Feature = z.infer<typeof FeatureSchema>;

export type Token = {
  token: Uint8Array;
  featureAct: number;
};
