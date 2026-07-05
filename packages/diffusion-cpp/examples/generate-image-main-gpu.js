'use strict'

const path = require('bare-path')
const process = require('bare-process')
const fs = require('bare-fs')
const ImgStableDiffusion = require('../index')

// main-gpu - pin diffusion to a specific GPU on multi-GPU hosts.
//
// `config['main-gpu']` selects the device the addon resolves against its own
// ggml enumeration, so it can never desync from the device the backend uses:
//
//   - 0, 1, 2, ...: physical GPU index in ggml's GPU-device list
//   - 'integrated': integrated GPU
//   - 'dedicated': discrete GPU with the most VRAM
//
// Omit it to keep the backend default (the first enumerated device). If an
// explicit request can't be satisfied (e.g. 'integrated' with no integrated
// GPU, 'dedicated' with no discrete GPU, or an out-of-range index), the addon
// falls back to CPU instead of substituting a different GPU.

const MODELS_DIR = path.resolve(__dirname, '../models')
const OUTPUT_DIR = path.resolve(__dirname, '../output')

const MODEL_NAME = 'flux-2-klein-4b-Q8_0.gguf'
const LLM_MODEL = 'Qwen3-4B-Q4_K_M.gguf'
const VAE_MODEL = 'flux2-vae.safetensors'

const MAIN_GPU = process.env.MAIN_GPU || 'dedicated'

const PROMPT = [
  'a compact workstation rendering a detailed image on a selected GPU,',
  'clean studio lighting, sharp technical product photography'
].join(' ')

const STEPS = 20
const WIDTH = 512
const HEIGHT = 512
const GUIDANCE = 3.5
const SEED = 42

function parseMainGpu (value) {
  if (value === 'integrated' || value === 'dedicated') return value

  const numeric = Number(value)
  if (Number.isInteger(numeric) && numeric >= 0) return numeric

  throw new Error(
    "MAIN_GPU must be a non-negative integer, 'integrated', or 'dedicated'"
  )
}

async function main () {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  const mainGpu = parseMainGpu(MAIN_GPU)

  console.log('FLUX.2 [klein] 4B - text-to-image with main-gpu config')
  console.log('========================================================')
  console.log('Prompt   :', PROMPT)
  console.log('Steps    :', STEPS)
  console.log('Size     :', `${WIDTH}x${HEIGHT}`)
  console.log('Seed     :', SEED)
  console.log('main-gpu :', mainGpu)
  console.log()

  const model = new ImgStableDiffusion({
    files: {
      model: path.join(MODELS_DIR, MODEL_NAME),
      llm: path.join(MODELS_DIR, LLM_MODEL),
      vae: path.join(MODELS_DIR, VAE_MODEL)
    },
    config: {
      device: 'gpu',
      threads: 4,
      diffusion_fa: true,
      'main-gpu': mainGpu
    },
    opts: { stats: true },
    logger: console
  })

  try {
    console.log('Loading model weights...')
    const tLoad = Date.now()
    await model.load()
    console.log(`Loaded in ${((Date.now() - tLoad) / 1000).toFixed(1)}s\n`)

    console.log('Starting generation...')
    const tGen = Date.now()

    const response = await model.run({
      prompt: PROMPT,
      steps: STEPS,
      width: WIDTH,
      height: HEIGHT,
      guidance: GUIDANCE,
      seed: SEED
    })

    const images = []
    let stats = null

    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) {
          images.push(data)
          return
        }

        if (typeof data !== 'string') return

        try {
          const tick = JSON.parse(data)
          if ('step' in tick && 'total' in tick) {
            const pct = Math.round((tick.step / tick.total) * 100)
            const bar = '#'.repeat(Math.floor(pct / 5)).padEnd(20, '.')
            process.stdout.write(`\r  [${bar}] ${tick.step}/${tick.total} steps`)
          } else if ('backendDevice' in tick) {
            stats = tick
          }
        } catch (_) {}
      })
      .await()

    process.stdout.write('\n')
    console.log(`\nGenerated in ${((Date.now() - tGen) / 1000).toFixed(1)}s`)
    console.log(`Got ${images.length} image(s)`)

    if (stats && stats.backendDevice) {
      // Confirms GPU vs CPU. Check the addon log line "main-gpu resolved to
      // backend 'VulkanN'" to confirm which physical GPU was chosen.
      console.log(`Backend device: ${stats.backendDevice}`)
    }

    for (let i = 0; i < images.length; i++) {
      const outPath = path.join(OUTPUT_DIR, `output_main_gpu_seed${SEED}_${i}.png`)
      fs.writeFileSync(outPath, images[i])
      console.log(`Saved -> ${outPath}`)
    }
  } finally {
    console.log('\nUnloading model...')
    await model.unload()
    console.log('Done.')
  }
}

main().catch(err => {
  console.error('Fatal:', err.message || err)
  process.exit(1)
})
