import { z } from 'zod'

const BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/

// ============================================
// Load-time config
// ============================================

export const lipsyncConfigSchema = z.object({
  backend: z
    .enum(['auto', 'cpu'])
    .optional()
    .describe(
      'Backend selection passed to `LipsyncModel.load({ backend })`. The addon ' +
        "is CPU-first; both values currently run on CPU, `'auto'` reserves GPU opt-in."
    ),
  verbosity: z
    .number()
    .int()
    .optional()
    .describe('Native log verbosity forwarded to the addon (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG).')
})

export type LipsyncConfig = z.input<typeof lipsyncConfigSchema>

// ============================================
// Hparams (returned by the addon after load)
// ============================================

export const lipsyncHparamsSchema = z.object({
  sampleRate: z.number().int().positive().describe('Expected PCM sample rate (16000).'),
  fps: z.number().int().positive().describe('Output frame rate (30).'),
  nCoeffs: z.number().int().positive().describe('Blendshape coefficients per frame (52).'),
  nIdentity: z.number().int().positive().describe('Number of identity conditioning classes.'),
  windowFrames: z
    .number()
    .int()
    .positive()
    .describe('Native streaming context window length in frames (64 ≈ 2.13 s).'),
  coeffNames: z
    .array(z.string())
    .describe('Blendshape names in output order (ARKit-52 convention).')
})

export type LipsyncHparams = z.infer<typeof lipsyncHparamsSchema>

// ============================================
// Stats
// ============================================

export const lipsyncStatsSchema = z.object({
  total_ms: z.number().optional(),
  frames: z.number().optional(),
  backendDevice: z.number().optional().describe('0 = CPU backend, 1 = GPU backend.')
})

export type LipsyncStats = z.infer<typeof lipsyncStatsSchema>

// ============================================
// Run request / response (wire format)
//
// The PCM buffer travels as a base64-encoded ArrayBuffer because JSON-RPC
// can't carry typed arrays natively. The client API (`lipsync()` in
// client/api/lipsync.ts) keeps the consumer-facing `Float32Array` shape.
// ============================================

export const lipsyncRunRequestSchema = z.object({
  type: z.literal('lipsyncRun'),
  modelId: z.string(),
  pcm: z
    .string()
    .min(1)
    .regex(BASE64_PATTERN)
    .describe(
      'Base64-encoded `Float32Array` of 16 kHz mono PCM in [-1, 1] ' +
        '(the underlying ArrayBuffer bytes).'
    ),
  idIdx: z
    .number()
    .int()
    .nonnegative()
    .optional()
    .describe('Identity conditioning index, 0..hparams.nIdentity-1 (default 0).')
})

export type LipsyncRunRequest = z.input<typeof lipsyncRunRequestSchema>

export const lipsyncRunResponseSchema = z.object({
  frames: z
    .string()
    .min(1)
    .regex(BASE64_PATTERN)
    .describe(
      'Base64-encoded `Float32Array` of `frameCount * nCoeffs` blendshape ' +
        'coefficients, frame-major, sigmoid range [0, 1].'
    ),
  frameCount: z.number().int().positive(),
  fps: z.number().int().positive(),
  nCoeffs: z.number().int().positive(),
  stats: lipsyncStatsSchema.optional()
})

export type LipsyncRunResponse = z.infer<typeof lipsyncRunResponseSchema>

// ============================================
// Hparams request / response (plugin handler)
// ============================================

export const lipsyncHparamsRequestSchema = z.object({
  type: z.literal('lipsyncHparams'),
  modelId: z.string()
})

export type LipsyncHparamsRequest = z.input<typeof lipsyncHparamsRequestSchema>

export const lipsyncHparamsResponseSchema = z.object({
  hparams: lipsyncHparamsSchema,
  backendName: z.string().nullable()
})

export type LipsyncHparamsResponse = z.infer<typeof lipsyncHparamsResponseSchema>

// ============================================
// Client-facing input shapes
// ============================================

export interface LipsyncClientRunParams {
  modelId: string
  /** 16 kHz mono PCM in [-1, 1]. */
  pcm: Float32Array
  /** Identity conditioning index (default 0). */
  idIdx?: number
}

export interface LipsyncClientRunResult {
  /** frameCount × nCoeffs coefficients, frame-major, ARKit-52 order. */
  frames: Float32Array
  frameCount: number
  fps: number
  nCoeffs: number
  stats?: LipsyncStats
}

export interface LipsyncStreamBatch {
  frames: Float32Array
  frameCount: number
  /** Timeline position of the first frame in this batch. */
  startTimeMs: number
}
