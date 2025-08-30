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
  stockfishAnalysis: z
    .object({
      bestMove: z.string().nullish(),
      ponder: z.string().nullish(),
      status: z.string().nullish(),
      error: z.any().nullish(),
      fen: z.string().nullish(),
      isCheck: z.boolean().nullish(),
      model: z
        .object({
          best_move: z.string().nullish(),
          best_move_san: z.string().nullish(),
          best_move_probability: z.number().nullish(),
          policy_analysis: z
            .object({
              best_moves: z.array(
                z.object({
                  san: z.string(),
                  probability: z.number(),
                })
              ).nullish(),
            })
            .nullish(),
          wdl_analysis: z
            .object({
              win_percent: z.number(),
              draw_percent: z.number(),
              loss_percent: z.number(),
            })
            .nullish(),
          value_analysis: z
            .object({
              raw_value: z.number(),
              normalized_value: z.number().nullish(),
            })
            .nullish(),
          raw_outputs: z
            .object({
              policy_logits_shape: z.array(z.number()).nullish(),
              wdl_probs_shape: z.array(z.number()).nullish(),
              value_shape: z.array(z.number()).nullish(),
            })
            .nullish(),
          error: z.string().nullish(),
        })
        .nullish(),
      rules: z
        .object({
          is_rook_under_attack: z.boolean().nullish(),
          is_knight_under_attack: z.boolean().nullish(),
          is_bishop_under_attack: z.boolean().nullish(),
          is_queen_under_attack: z.boolean().nullish(),
          is_can_capture_rook: z.boolean().nullish(),
          is_can_capture_knight: z.boolean().nullish(),
          is_can_capture_bishop: z.boolean().nullish(),
          is_can_capture_queen: z.boolean().nullish(),
          is_king_in_check: z.boolean().nullish(),
          is_checkmate: z.boolean().nullish(),
          is_stalemate: z.boolean().nullish(),
        })
        .nullish(),
      material: z
        .object({
          white_material: z.number().nullish(),
          black_material: z.number().nullish(),
          material_advantage: z.number().nullish(),
          error: z.string().nullish(),
        })
        .nullish(),
      wdl: z
        .object({
          win_probability: z.number().nullish(),
          draw_probability: z.number().nullish(),
          loss_probability: z.number().nullish(),
          error: z.string().nullish(),
        })
        .nullish(),
    })
    .nullish(),
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
          response: z.any(),
        })
        .optional(),
    })
  ),
  detail: z
    .object({
      userPrompt: z.string(),
      systemPrompt: z.string(),
      response: z.object({
        steps: z.array(z.string()),
        finalExplanation: z.string(),
        complexity: z.number(),
        activationConsistency: z.number(),
      }),
    })
    .optional(),
  complexity: z.number().optional(),
  consistency: z.number().optional(),
  passed: z.boolean().optional(),
  time: z.any().optional(),
});

export type Interpretation = z.infer<typeof InterpretationSchema>;

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  dictionaryName: z.string(),
  analysisName: z.string(),
  featureActivationHistogram: z.any().nullish(),
  decoderNorms: z.array(z.number()).nullish(),
  decoderSimilarityMatrix: z.array(z.array(z.number())).nullish(),
  decoderInnerProductMatrix: z.array(z.array(z.number())).nullish(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  nAnalyzedTokens: z.number().nullish(),
  actTimesModalities: z.record(z.string(), z.number()).nullish(),
  maxFeatureActsModalities: z.record(z.string(), z.number()).nullish(),
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
    .nullish(),
  interpretation: InterpretationSchema.nullish(),
  isBookmarked: z.boolean().optional(),
});

export type Feature = z.infer<typeof FeatureSchema>;
