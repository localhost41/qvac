/**
 * Lipsync demo generator: WAV → ARKit-52 blendshape frames via the QVAC SDK.
 *
 * Exercises the full stack: loadModel({ modelType: "lipsync" }) → bare
 * worker → ggml-lipsync plugin → @qvac/lipsync-ggml addon, streamed through
 * the client `lipsyncStream` API (320 ms chunks, rolling 64-frame window).
 *
 * Usage:
 *   bun run examples/lipsync/lipsync-frames.ts <model.gguf> <in-16k-mono.wav> <out.json>
 *
 * The output JSON ({ fps, coeffNames, frames[][] }) drives the browser demo
 * in packages/lipsync-ggml/examples/demo/.
 */
import { loadModel, unloadModel, lipsyncHparams, lipsyncStream } from '@qvac/sdk'
import { readFileSync, writeFileSync } from 'fs'

const CHUNK_SAMPLES = 5120 // 320 ms @ 16 kHz

function readWav16kMonoS16(path: string): Float32Array {
  const buf = readFileSync(path)
  // Minimal RIFF parse: locate the "data" chunk rather than assuming a
  // 44-byte header.
  let off = 12
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4)
    const size = buf.readUInt32LE(off + 4)
    if (id === 'data') {
      const pcm = new Float32Array(size / 2)
      for (let i = 0; i < pcm.length; i++) {
        pcm[i] = buf.readInt16LE(off + 8 + i * 2) / 32768
      }
      return pcm
    }
    off += 8 + size + (size % 2)
  }
  throw new Error('no data chunk found in WAV')
}

const [modelSrc, wavPath, outPath] = process.argv.slice(2)
if (!modelSrc || !wavPath || !outPath) {
  console.error('usage: bun run examples/lipsync/lipsync-frames.ts <model.gguf> <in.wav> <out.json>')
  process.exit(1)
}

const pcm = readWav16kMonoS16(wavPath)
console.error(`audio: ${pcm.length} samples (${(pcm.length / 16000).toFixed(2)} s)`)

console.error('▸ Loading lipsync model through the SDK...')
const modelId = await loadModel({ modelSrc, modelType: 'lipsync' })
const { hparams, backendName } = await lipsyncHparams({ modelId })
console.error(
  `▸ Loaded: ${hparams.nCoeffs} coeffs @ ${hparams.fps} fps, backend ${backendName ?? 'unknown'}`
)

function* chunks(): Generator<Float32Array> {
  for (let off = 0; off < pcm.length; off += CHUNK_SAMPLES) {
    yield pcm.subarray(off, Math.min(off + CHUNK_SAMPLES, pcm.length))
  }
}

const frames: number[][] = []
const t0 = Date.now()
for await (const batch of lipsyncStream(chunks(), { modelId })) {
  for (let f = 0; f < batch.frameCount; f++) {
    const row: number[] = []
    for (let c = 0; c < hparams.nCoeffs; c++) {
      row.push(Math.round((batch.frames[f * hparams.nCoeffs + c] ?? 0) * 1000) / 1000)
    }
    frames.push(row)
  }
}
const elapsed = (Date.now() - t0) / 1000
console.error(
  `▸ ${frames.length} frames @ ${hparams.fps} fps ` +
    `(inference ${elapsed.toFixed(2)} s, ${(pcm.length / 16000 / elapsed).toFixed(1)}x realtime)`
)

writeFileSync(outPath, JSON.stringify({ fps: hparams.fps, coeffNames: hparams.coeffNames, frames }))
console.error(`▸ wrote ${outPath}`)

await unloadModel({ modelId })
process.exit(0)
