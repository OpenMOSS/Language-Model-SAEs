import { z } from "zod";

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  samples: z.array(
    z.object({
      context: z.array(z.instanceof(Uint8Array)),
      featureActs: z.array(z.number()),
    })
  ),
});

export type Feature = z.infer<typeof FeatureSchema>;

export type Token = {
  token: Uint8Array;
  featureAct: number;
};
