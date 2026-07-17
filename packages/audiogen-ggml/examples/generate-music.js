'use strict'

// Usage (once the native addon builds, i.e. after the tts-cpp acestep engine
// port lands — see tts-cpp/docs/QVAC-21921-audiogen-plan.md):
//
//   bare examples/generate-music.js "lo-fi hip hop, mellow piano, rainy night"
//
// Env:
//   AUDIOGEN_MODEL_DIR  directory holding the ACE-Step GGUFs

const fs = require('bare-fs')
const { AudioGen } = require('..')

async function main () {
  const caption =
    process.argv[2] ||
    'Upbeat pop rock with driving electric guitars, punchy drums and a catchy hook'
  const modelDir = process.env.AUDIOGEN_MODEL_DIR
  const outFile = process.env.AUDIOGEN_OUT || 'audiogen-output.pcm'

  console.log('[audiogen] prompt: ' + caption)

  const gen = new AudioGen({ modelDir })
  await gen.activate()

  const t0 = Date.now()
  const result = await gen.generate(caption, { lyrics: '[Instrumental]' })
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1)

  // result.outputArray is interleaved stereo Int16 @ result.sampleRate.
  fs.writeFileSync(outFile, Buffer.from(result.outputArray.buffer))

  console.log('[audiogen] done in ' + elapsed + 's')
  console.log('[audiogen] samples:   ' + result.outputArray.length)
  console.log('[audiogen] rate:      ' + result.sampleRate + ' Hz, ' + result.channels + 'ch')
  console.log('[audiogen] key:       ' + (result.metadata && result.metadata.keyscale))
  console.log('[audiogen] raw PCM ->  ' + outFile)

  await gen.destroy()
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
