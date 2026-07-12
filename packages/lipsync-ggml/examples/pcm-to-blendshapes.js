'use strict'

// Minimal example: raw f32 PCM file (16 kHz mono) → ARKit-52 blendshape
// frames printed as JSON lines, one per 30 fps frame.
//
// Usage:
//   bare examples/pcm-to-blendshapes.js <model.gguf> <audio.f32> [idIdx]
//
// To make a .f32 file from a wav:
//   ffmpeg -i speech.wav -f f32le -acodec pcm_f32le -ac 1 -ar 16000 speech.f32

const fs = require('bare-fs')
const LipsyncModel = require('..')

async function main () {
  const [modelPath, pcmPath, idIdxArg] = (global.Bare ? global.Bare.argv : process.argv).slice(2)
  if (!modelPath || !pcmPath) {
    console.error('usage: bare examples/pcm-to-blendshapes.js <model.gguf> <audio.f32> [idIdx]')
    return
  }

  const buf = fs.readFileSync(pcmPath)
  const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)

  const model = new LipsyncModel({ files: { model: [modelPath] } })
  await model.load()
  console.error(`loaded: ${model.hparams.nCoeffs} coeffs @ ${model.hparams.fps} fps, backend ${model.backendName}`)

  const { frames, frameCount, fps } = await model.run({ pcm, idIdx: Number(idIdxArg || 0) })
  const names = model.hparams.coeffNames
  for (let f = 0; f < frameCount; f++) {
    const coeffs = {}
    for (let c = 0; c < names.length; c++) {
      const v = frames[f * names.length + c]
      if (v > 0.01) coeffs[names[c]] = Number(v.toFixed(4))
    }
    console.log(JSON.stringify({ tMs: Math.round(f / fps * 1000), ...coeffs }))
  }

  await model.unload()
}

main().catch((err) => {
  console.error(err)
})
