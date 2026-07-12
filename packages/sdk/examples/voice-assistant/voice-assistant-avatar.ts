/**
 * Voice Assistant with avatar lipsync:
 * mic → Whisper (Silero VAD) → Llama → Supertonic TTS
 *                                        ├─→ speakers (delayed one chunk)
 *                                        └─→ lipsync-ggml → ARKit-52 frames
 *                                              → WebSocket ws://localhost:8787
 *
 * Usage:
 *   LIPSYNC_GGUF=/abs/path/lam-audio2exp-f32.gguf \
 *     bun run examples/voice-assistant/voice-assistant-avatar.ts
 *
 * A renderer (three.js + @pixiv/three-vrm, or any ARKit-blendshape consumer)
 * connects to the WebSocket and receives JSON events:
 *   { type: "hparams",  coeffNames: string[], fps: 30 }
 *   { type: "utterance", role: "user" | "assistant", text: string }
 *   { type: "frames",   startTimeMs: number, fps: 30, frames: number[][] }
 *   { type: "audioStart" }   — playback begins now; sync frame t=startTimeMs
 *   { type: "idle" }         — coefficient stream ended; return to idle loop
 *
 * Requirements: FFmpeg installed, microphone access, speakers, converted
 * lipsync GGUF (packages/lipsync-ggml/scripts/convert-lam-to-gguf.py).
 */
import {
  loadModel,
  unloadModel,
  transcribeStream,
  completion,
  textToSpeech,
  lipsyncStream,
  lipsyncHparams,
  WHISPER_TINY,
  VAD_SILERO_5_1_2,
  LLAMA_3_2_1B_INST_Q4_0,
  TTS_EN_SUPERTONIC_Q8_0
} from '@qvac/sdk'
import { spawnSync } from 'child_process'
import { startMicrophone } from '../audio/mic-input'
import { createWavHeader, int16ArrayToBuffer, playAudio } from '../tts/utils'

const MIC_SAMPLE_RATE = 16000
const TTS_SAMPLE_RATE = 44100
const LIPSYNC_SAMPLE_RATE = 16000
const LIPSYNC_CHUNK_SAMPLES = 5120 // 320 ms @ 16 kHz — recommended hop
const WS_PORT = 8787

const SYSTEM_PROMPT =
  'You are a concise, friendly voice assistant. Keep responses under two sentences. ' +
  'Never use markdown, lists, or code blocks — your output will be spoken aloud.'

const VAD_PARAMS = {
  threshold: 0.6,
  min_speech_duration_ms: 300,
  min_silence_duration_ms: 700,
  max_speech_duration_s: 15.0,
  speech_pad_ms: 200
}

const POST_PLAYBACK_COOLDOWN_MS = 300
const MIN_UTTERANCE_CHARS = 3

function isMeaningfulTranscript(text: string): boolean {
  const trimmed = text.trim()
  if (trimmed.length === 0) return false
  if (trimmed.includes('[No speech detected]')) return false
  if (/^\[[^\]]+\]$/.test(trimmed)) return false
  const letters = trimmed.replace(/[^\p{L}\p{N}]/gu, '')
  return letters.length >= MIN_UTTERANCE_CHARS
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * int16 @ 44.1 kHz → f32 [-1,1] @ 16 kHz (linear interpolation). Good enough
 * for the lipsync feature extractor; the audible path keeps the original.
 */
function toLipsyncPcm(samples: number[]): Float32Array {
  const ratio = TTS_SAMPLE_RATE / LIPSYNC_SAMPLE_RATE
  const outLen = Math.floor(samples.length / ratio)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, samples.length - 1)
    const w = pos - i0
    out[i] = ((samples[i0] ?? 0) * (1 - w) + (samples[i1] ?? 0) * w) / 32768
  }
  return out
}

function* inChunks(pcm: Float32Array): Generator<Float32Array> {
  for (let off = 0; off < pcm.length; off += LIPSYNC_CHUNK_SAMPLES) {
    yield pcm.subarray(off, Math.min(off + LIPSYNC_CHUNK_SAMPLES, pcm.length))
  }
}

// ── Main ──

for (const tool of ['ffmpeg', 'ffplay']) {
  const r = spawnSync(tool, ['-version'], { stdio: 'ignore' })
  if (r.error || r.status !== 0) {
    console.error(`✖ ${tool} not found on PATH. Install ffmpeg (ffplay ships with it) and retry.`)
    process.exit(1)
  }
}

const lipsyncGguf = process.env['LIPSYNC_GGUF'] as string | undefined
if (!lipsyncGguf) {
  console.error('✖ Set LIPSYNC_GGUF to the converted lam-audio2exp GGUF path.')
  process.exit(1)
}

// ── WebSocket broadcast for the booth renderer ──
const clients = new Set<{ send(data: string): void }>()
function broadcast(event: Record<string, unknown>) {
  const payload = JSON.stringify(event)
  for (const ws of clients) ws.send(payload)
}

// Bun's built-in HTTP+WS server; no extra dependency.
declare const Bun: {
  serve(opts: unknown): unknown
}
let hparamsEvent: Record<string, unknown> | null = null
Bun.serve({
  port: WS_PORT,
  fetch(req: Request, server: { upgrade(req: Request): boolean }) {
    if (server.upgrade(req)) return undefined
    return new Response('lipsync frame stream: connect via WebSocket', { status: 200 })
  },
  websocket: {
    open(ws: { send(data: string): void }) {
      clients.add(ws)
      if (hparamsEvent) ws.send(JSON.stringify(hparamsEvent))
    },
    close(ws: { send(data: string): void }) {
      clients.delete(ws)
    },
    message() {}
  }
})
console.log(`▸ Renderer WebSocket on ws://localhost:${WS_PORT}`)

console.log('▸ Loading whisper-tiny + Silero VAD...')
const asrModelId = await loadModel({
  modelSrc: WHISPER_TINY,
  modelConfig: {
    vadModelSrc: VAD_SILERO_5_1_2,
    audio_format: 'f32le',
    strategy: 'greedy',
    n_threads: 4,
    language: 'en',
    no_timestamps: true,
    suppress_blank: true,
    suppress_nst: true,
    temperature: 0.0,
    vad_params: VAD_PARAMS
  }
})

console.log('▸ Loading Llama 3.2 1B...')
const llmModelId = await loadModel({
  modelSrc: LLAMA_3_2_1B_INST_Q4_0,
  modelConfig: { ctx_size: 4096 }
})

console.log('▸ Loading Supertonic TTS...')
const ttsModelId = await loadModel({
  modelSrc: TTS_EN_SUPERTONIC_Q8_0,
  modelConfig: {
    ttsEngine: 'supertonic',
    language: 'en',
    voice: 'F1',
    ttsSpeed: 1.05,
    ttsNumInferenceSteps: 5
  }
})

console.log('▸ Loading lipsync (LAM Audio2Expression)...')
const lipsyncModelId = await loadModel({
  modelSrc: lipsyncGguf,
  modelType: 'lipsync'
})
const { hparams: lipsyncHp } = await lipsyncHparams({ modelId: lipsyncModelId })
hparamsEvent = { type: 'hparams', coeffNames: lipsyncHp.coeffNames, fps: lipsyncHp.fps }
broadcast(hparamsEvent)

console.log('▸ All models loaded.\n')

const ffmpeg = startMicrophone({ sampleRate: MIC_SAMPLE_RATE, format: 'f32le' })
const session = await transcribeStream({ modelId: asrModelId })

const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
  { role: 'system', content: SYSTEM_PROMPT }
]

let isSpeaking = false
ffmpeg.stdout.on('data', (chunk: Buffer) => {
  if (isSpeaking) return
  session.write(chunk)
})

let shuttingDown = false
async function cleanup() {
  if (shuttingDown) return
  shuttingDown = true
  console.log('\n\n▸ Stopping...')
  ffmpeg.kill()
  try {
    session.end()
  } catch {
    // session may already be closed
  }
  await unloadModel({ modelId: lipsyncModelId }).catch(() => {})
  await unloadModel({ modelId: ttsModelId }).catch(() => {})
  await unloadModel({ modelId: llmModelId }).catch(() => {})
  await unloadModel({ modelId: asrModelId }).catch(() => {})
  console.log('▸ Done.')
  process.exit(0)
}

process.on('SIGINT', () => void cleanup())
process.on('SIGTERM', () => void cleanup())

console.log('▸ Listening. Speak a question and pause. Ctrl+C to quit.\n')

for await (const rawText of session) {
  if (!isMeaningfulTranscript(rawText)) continue
  const userText = rawText.trim()

  console.log(`▸ You: ${userText}`)
  history.push({ role: 'user', content: userText })
  broadcast({ type: 'utterance', role: 'user', text: userText })

  isSpeaking = true
  try {
    console.log('▸ Assistant:')
    const llmResult = completion({ modelId: llmModelId, history, stream: true })
    let assistantText = ''
    for await (const token of llmResult.tokenStream) {
      process.stdout.write(token)
      assistantText += token
    }
    process.stdout.write('\n')
    history.push({ role: 'assistant', content: assistantText })
    broadcast({ type: 'utterance', role: 'assistant', text: assistantText })

    const spoken = assistantText.trim()
    if (spoken.length > 0) {
      const ttsResult = textToSpeech({
        modelId: ttsModelId,
        text: spoken,
        inputType: 'text',
        stream: false
      })
      const samples = await ttsResult.buffer

      // Tee: full utterance → lipsync coefficients first (fast: ~5× realtime),
      // broadcast frame batches, then start audio playback. The renderer
      // clocks frames against the audioStart event, so audio and mouth stay
      // in sync; the added latency is the lipsync compute time.
      const lipsyncPcm = toLipsyncPcm(samples)
      for await (const batch of lipsyncStream(inChunks(lipsyncPcm), {
        modelId: lipsyncModelId
      })) {
        const frames: number[][] = []
        for (let f = 0; f < batch.frameCount; f++) {
          frames.push(
            Array.from(
              batch.frames.subarray(f * lipsyncHp.nCoeffs, (f + 1) * lipsyncHp.nCoeffs),
              (v) => Math.round(v * 1000) / 1000
            )
          )
        }
        broadcast({ type: 'frames', startTimeMs: batch.startTimeMs, fps: lipsyncHp.fps, frames })
      }

      const audioData = int16ArrayToBuffer(samples)
      const wavBuffer = Buffer.concat([createWavHeader(audioData.length, TTS_SAMPLE_RATE), audioData])
      broadcast({ type: 'audioStart' })
      playAudio(wavBuffer)
      await sleep(POST_PLAYBACK_COOLDOWN_MS)
      broadcast({ type: 'idle' })
    }
  } catch (turnError) {
    console.error('\n✖ Turn failed:', turnError instanceof Error ? turnError.message : turnError)
  } finally {
    isSpeaking = false
    console.log('\n▸ Listening...\n')
  }
}
