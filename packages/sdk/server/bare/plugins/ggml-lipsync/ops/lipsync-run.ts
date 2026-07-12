import { getModel } from '@/server/bare/registry/model-registry'
import { type LipsyncRunRequest, type LipsyncRunResponse, type LipsyncStats } from '@/schemas'
import { decodeBase64, encodeBase64 } from '@/utils/encoding'

interface LipsyncModelLike {
  run(input: { pcm: Float32Array; idIdx?: number }): Promise<{
    frames: Float32Array
    frameCount: number
    fps: number
    stats?: Record<string, number> | null
  }>
  hparams: { nCoeffs: number } | null
}

function f32FromBase64(b64: string): Float32Array {
  const bytes = decodeBase64(b64)
  // ArrayBuffer view must be aligned; copy if needed.
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return new Float32Array(bytes.buffer)
  }
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  return new Float32Array(copy.buffer)
}

function f32ToBase64(arr: Float32Array): string {
  return encodeBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength))
}

function pickStats(raw: unknown): LipsyncStats | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const r = raw as Record<string, unknown>
  const out: LipsyncStats = {}
  if (typeof r['total_ms'] === 'number') out.total_ms = r['total_ms']
  if (typeof r['frames'] === 'number') out.frames = r['frames']
  if (typeof r['backendDevice'] === 'number') out.backendDevice = r['backendDevice']
  return Object.keys(out).length > 0 ? out : undefined
}

export async function lipsyncRun(request: LipsyncRunRequest): Promise<LipsyncRunResponse> {
  const model = getModel(request.modelId) as unknown as LipsyncModelLike

  const result = await model.run({
    pcm: f32FromBase64(request.pcm),
    idIdx: request.idIdx ?? 0
  })

  const hp = model.hparams
  if (!hp) {
    throw new Error('Lipsync model hparams unavailable after run')
  }

  const stats = pickStats(result.stats)
  return {
    frames: f32ToBase64(result.frames),
    frameCount: result.frameCount,
    fps: result.fps,
    nCoeffs: hp.nCoeffs,
    ...(stats && { stats })
  }
}
