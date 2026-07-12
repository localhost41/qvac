import {
  type LipsyncClientRunParams,
  type LipsyncClientRunResult,
  type LipsyncHparams,
  type LipsyncHparamsRequest,
  type LipsyncHparamsResponse,
  type LipsyncRunRequest,
  type LipsyncRunResponse,
  type LipsyncStreamBatch,
  lipsyncHparamsResponseSchema,
  lipsyncRunResponseSchema
} from '@/schemas'
import { decodeBase64, encodeBase64 } from '@/utils/encoding'
import { invokePlugin } from './invoke-plugin'

const LIPSYNC_RUN_HANDLER = 'lipsyncRun'
const LIPSYNC_HPARAMS_HANDLER = 'lipsyncHparams'

// LAM streaming geometry (matches the addon's hparams defaults; the
// authoritative values come from `lipsyncHparams()`).
const SAMPLE_RATE = 16000
const FPS = 30
const WINDOW_FRAMES = 64
const WINDOW_SAMPLES = Math.floor((SAMPLE_RATE * WINDOW_FRAMES) / FPS)

function bytesOf(arr: Float32Array): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)
}

function f32FromBase64(b64: string): Float32Array {
  const bytes = decodeBase64(b64)
  // base64 round-trip produces a fresh Uint8Array whose buffer starts at 0
  // and isn't shared, so the underlying ArrayBuffer is safe to reinterpret.
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}

/**
 * Run lipsync inference over a PCM buffer and return ARKit-52 blendshape
 * coefficient frames at 30 fps.
 *
 * @param params - Inference inputs.
 * @param params.modelId - Identifier of the loaded lipsync model (returned
 *   by `loadModel({ modelType: "lipsync", ... })`).
 * @param params.pcm - 16 kHz mono PCM in `[-1, 1]`.
 * @param params.idIdx - Optional identity conditioning index
 *   (`0..hparams.nIdentity-1`, default 0).
 * @returns Frames (`Float32Array`, `frameCount × nCoeffs`, frame-major,
 *   sigmoid range `[0, 1]`), the frame geometry, and optional stats.
 *   Coefficient order is `hparams.coeffNames` (ARKit-52).
 *
 * @example
 * ```typescript
 * import { loadModel, lipsync, lipsyncHparams } from "@qvac/sdk";
 *
 * const modelId = await loadModel({ modelSrc: "/path/lam-audio2exp-f32.gguf", modelType: "lipsync" });
 * const { hparams } = await lipsyncHparams({ modelId });
 * const { frames, frameCount } = await lipsync({ modelId, pcm });
 * // frames[f * hparams.nCoeffs + hparams.coeffNames.indexOf("jawOpen")]
 * ```
 */
export async function lipsync(params: LipsyncClientRunParams): Promise<LipsyncClientRunResult> {
  const wireRequest: LipsyncRunRequest = {
    type: 'lipsyncRun',
    modelId: params.modelId,
    pcm: encodeBase64(bytesOf(params.pcm)),
    ...(params.idIdx !== undefined && { idIdx: params.idIdx })
  }

  const result = await invokePlugin<LipsyncRunResponse, LipsyncRunRequest>({
    modelId: params.modelId,
    handler: LIPSYNC_RUN_HANDLER,
    params: wireRequest
  })

  const parsed = lipsyncRunResponseSchema.parse(result)
  return {
    frames: f32FromBase64(parsed.frames),
    frameCount: parsed.frameCount,
    fps: parsed.fps,
    nCoeffs: parsed.nCoeffs,
    ...(parsed.stats && { stats: parsed.stats })
  }
}

/**
 * Streaming lipsync: consume an async iterable of PCM chunks (16 kHz mono
 * `Float32Array`, e.g. teed from `textToSpeechStream` output) and yield
 * blendshape frame batches per chunk.
 *
 * Implements the upstream LAM windowing policy client-side: each chunk is
 * right-aligned into a rolling 64-frame (~2.13 s) context window, the window
 * is inferred whole, and only the chunk's new frames are emitted. Chunks
 * larger than the window are split. ~320 ms chunks are recommended; delay
 * audio playback by one chunk for exact sync.
 *
 * @param pcmStream - (Async) iterable of 16 kHz mono PCM chunks.
 * @param params.modelId - Identifier of the loaded lipsync model.
 * @param params.idIdx - Optional identity conditioning index (default 0).
 * @returns Async generator of `{ frames, frameCount, startTimeMs }` batches.
 */
export async function* lipsyncStream(
  pcmStream: AsyncIterable<Float32Array> | Iterable<Float32Array>,
  params: { modelId: string; idIdx?: number }
): AsyncGenerator<LipsyncStreamBatch> {
  let previousAudio = new Float32Array(WINDOW_SAMPLES) // zeros = silence
  let framesEmitted = 0

  for await (const chunk of pcmStream) {
    for (let off = 0; off < chunk.length; off += WINDOW_SAMPLES) {
      const piece = chunk.subarray(off, Math.min(off + WINDOW_SAMPLES, chunk.length))
      const keep = WINDOW_SAMPLES - piece.length
      const window = new Float32Array(WINDOW_SAMPLES)
      window.set(previousAudio.subarray(previousAudio.length - keep), 0)
      window.set(piece, keep)
      previousAudio = window

      const { frames, nCoeffs } = await lipsync({
        modelId: params.modelId,
        pcm: window,
        ...(params.idIdx !== undefined && { idIdx: params.idIdx })
      })

      // Upstream: start_frame = int(windowFrames - chunkSamples / sr * fps)
      const emitFromFrame = Math.floor(WINDOW_FRAMES - (piece.length / SAMPLE_RATE) * FPS)
      const out = frames.subarray(emitFromFrame * nCoeffs)
      const outCount = out.length / nCoeffs
      yield {
        frames: out,
        frameCount: outCount,
        startTimeMs: (framesEmitted / FPS) * 1000
      }
      framesEmitted += outCount
    }
  }
}

/**
 * Fetch the loaded lipsync model's hyperparameters (sample rate, fps,
 * coefficient count and ARKit-52 name order) and the active ggml backend
 * name.
 *
 * @param params - Identifier of the loaded lipsync model.
 */
export async function lipsyncHparams(params: {
  modelId: string
}): Promise<{ hparams: LipsyncHparams; backendName: string | null }> {
  const wireRequest: LipsyncHparamsRequest = {
    type: 'lipsyncHparams',
    modelId: params.modelId
  }
  const result = await invokePlugin<LipsyncHparamsResponse, LipsyncHparamsRequest>({
    modelId: params.modelId,
    handler: LIPSYNC_HPARAMS_HANDLER,
    params: wireRequest
  })
  return lipsyncHparamsResponseSchema.parse(result)
}
