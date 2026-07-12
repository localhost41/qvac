/**
 * Text → speech → blendshapes, entirely through the QVAC SDK.
 *
 * Synthesizes the text with Supertonic TTS (tts-ggml), resamples the audio
 * to the lipsync model's 16 kHz input rate, runs it through lipsync-ggml
 * (streamed, 320 ms chunks), and writes both the WAV and the frames JSON
 * that the browser demo consumes.
 *
 * Usage:
 *   bun run examples/lipsync/say.ts \
 *     --text "Hello, ask me anything." \
 *     --lipsync /abs/path/lam-audio2exp-f32.gguf \
 *     --out-wav /tmp/say.wav --out-frames /tmp/say-frames.json \
 *     [--voice F1] [--speed 1.05]
 */
import {
  loadModel,
  unloadModel,
  textToSpeech,
  lipsyncHparams,
  lipsyncStream,
  TTS_EN_SUPERTONIC_Q8_0
} from '@qvac/sdk'
import { writeFileSync } from 'fs'

const TTS_SAMPLE_RATE = 44100
const LIPSYNC_SAMPLE_RATE = 16000
// Chunk size MUST be a whole number of 30 fps frames (multiples of 1600
// samples = 0.1 s = 3 frames). The model slices its 64-frame window with
// int(64 - samples/sr*30); a fractional frame count (e.g. 5120 = 9.6
// frames) truncates and emits an extra frame per chunk — ~4% cumulative
// audio/animation drift that visibly desyncs lips on clips >10 s.
const CHUNK_SAMPLES = 4800 // 300 ms = exactly 9 frames @ 16 kHz / 30 fps

function arg(name: string, dflt?: string): string {
  const i = process.argv.indexOf('--' + name)
  if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1] as string
  if (dflt !== undefined) return dflt
  console.error(`missing --${name}`)
  process.exit(1)
}

const text = arg('text')
const lipsyncGguf = arg('lipsync')
const outWav = arg('out-wav')
const outFrames = arg('out-frames')
const voice = arg('voice', 'F1')
const speed = Number(arg('speed', '1.05'))

console.error('▸ Loading Supertonic TTS...')
const ttsModelId = await loadModel({
  modelSrc: TTS_EN_SUPERTONIC_Q8_0,
  modelConfig: {
    ttsEngine: 'supertonic',
    language: 'en',
    voice,
    ttsSpeed: speed,
    ttsNumInferenceSteps: 5
  }
})

console.error(`▸ Synthesizing: "${text}"`)
const ttsResult = textToSpeech({
  modelId: ttsModelId,
  text,
  inputType: 'text',
  stream: false
})
const samples = await ttsResult.buffer // int16 values @ 44.1 kHz
console.error(`▸ TTS audio: ${samples.length} samples (${(samples.length / TTS_SAMPLE_RATE).toFixed(2)} s)`)

// 44.1 kHz int16 → 16 kHz f32 (linear interpolation)
const ratio = TTS_SAMPLE_RATE / LIPSYNC_SAMPLE_RATE
const outLen = Math.floor(samples.length / ratio)
const pcm = new Float32Array(outLen)
for (let i = 0; i < outLen; i++) {
  const pos = i * ratio
  const i0 = Math.floor(pos)
  const i1 = Math.min(i0 + 1, samples.length - 1)
  const w = pos - i0
  pcm[i] = ((samples[i0] ?? 0) * (1 - w) + (samples[i1] ?? 0) * w) / 32768
}

// write 16 kHz mono s16 WAV (what the demo page embeds)
function writeWav(path: string, f32: Float32Array, sr: number) {
  const data = Buffer.alloc(44 + f32.length * 2)
  data.write('RIFF', 0)
  data.writeUInt32LE(36 + f32.length * 2, 4)
  data.write('WAVE', 8)
  data.write('fmt ', 12)
  data.writeUInt32LE(16, 16)
  data.writeUInt16LE(1, 20)
  data.writeUInt16LE(1, 22)
  data.writeUInt32LE(sr, 24)
  data.writeUInt32LE(sr * 2, 28)
  data.writeUInt16LE(2, 32)
  data.writeUInt16LE(16, 34)
  data.write('data', 36)
  data.writeUInt32LE(f32.length * 2, 40)
  for (let i = 0; i < f32.length; i++) {
    data.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round((f32[i] ?? 0) * 32767))), 44 + i * 2)
  }
  writeFileSync(path, data)
}
writeWav(outWav, pcm, LIPSYNC_SAMPLE_RATE)
console.error(`▸ wrote ${outWav}`)

console.error('▸ Loading lipsync model...')
const lipsyncModelId = await loadModel({ modelSrc: lipsyncGguf, modelType: 'lipsync' })
const { hparams } = await lipsyncHparams({ modelId: lipsyncModelId })

function* chunks(): Generator<Float32Array> {
  for (let off = 0; off < pcm.length; off += CHUNK_SAMPLES) {
    yield pcm.subarray(off, Math.min(off + CHUNK_SAMPLES, pcm.length))
  }
}

const frames: number[][] = []
const t0 = Date.now()
for await (const batch of lipsyncStream(chunks(), { modelId: lipsyncModelId })) {
  for (let f = 0; f < batch.frameCount; f++) {
    const row: number[] = []
    for (let c = 0; c < hparams.nCoeffs; c++) {
      row.push(Math.round((batch.frames[f * hparams.nCoeffs + c] ?? 0) * 1000) / 1000)
    }
    frames.push(row)
  }
}
console.error(
  `▸ ${frames.length} frames @ ${hparams.fps} fps ` +
    `(lipsync ${((Date.now() - t0) / 1000).toFixed(2)} s)`
)

writeFileSync(outFrames, JSON.stringify({ fps: hparams.fps, coeffNames: hparams.coeffNames, frames }))
console.error(`▸ wrote ${outFrames}`)

await unloadModel({ modelId: lipsyncModelId })
await unloadModel({ modelId: ttsModelId })
process.exit(0)
