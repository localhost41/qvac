'use strict'

// Demo helper: 16 kHz mono s16 WAV → blendshape frames JSON.
//
// Runs the WAV through runStreaming (the production path: 320 ms chunks,
// rolling 64-frame window) and writes { fps, coeffNames, frames[][] } for
// the browser demo renderer.
//
// Usage: bare examples/wav-to-frames-json.js <model.gguf> <in.wav> <out.json>

const fs = require('bare-fs')
const LipsyncModel = require('..')

const CHUNK_SAMPLES = 5120 // 320 ms @ 16 kHz

function readWav16kMonoS16 (path) {
  const buf = fs.readFileSync(path)
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

async function main () {
  const [modelPath, wavPath, outPath] = (global.Bare ? global.Bare.argv : process.argv).slice(2)
  if (!modelPath || !wavPath || !outPath) {
    console.error('usage: bare examples/wav-to-frames-json.js <model.gguf> <in.wav> <out.json>')
    return
  }

  const pcm = readWav16kMonoS16(wavPath)
  console.error(`audio: ${pcm.length} samples (${(pcm.length / 16000).toFixed(2)} s)`)

  const model = new LipsyncModel({ files: { model: [modelPath] } })
  await model.load()
  const { coeffNames, fps, nCoeffs } = model.hparams

  async function * chunks () {
    for (let off = 0; off < pcm.length; off += CHUNK_SAMPLES) {
      yield pcm.subarray(off, Math.min(off + CHUNK_SAMPLES, pcm.length))
    }
  }

  const frames = []
  const t0 = Date.now()
  for await (const batch of model.runStreaming(chunks(), { idIdx: 0 })) {
    for (let f = 0; f < batch.frameCount; f++) {
      const row = []
      for (let c = 0; c < nCoeffs; c++) {
        row.push(Math.round(batch.frames[f * nCoeffs + c] * 1000) / 1000)
      }
      frames.push(row)
    }
  }
  const elapsed = (Date.now() - t0) / 1000
  console.error(`frames: ${frames.length} @ ${fps} fps ` +
    `(inference ${elapsed.toFixed(2)} s, ${(pcm.length / 16000 / elapsed).toFixed(1)}x realtime)`)

  fs.writeFileSync(outPath, JSON.stringify({ fps, coeffNames, frames }))
  console.error(`wrote ${outPath}`)
  await model.unload()
}

main().catch((err) => {
  console.error(err)
})
