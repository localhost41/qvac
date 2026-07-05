'use strict'

// main-gpu probe — verify which physical device the addon selects for image
// generation on a multi-GPU host, and how each `main-gpu` value resolves.
//
// It drives one short text-to-image generation and captures the addon's own
// C++ backend-selection logs (via binding.setLogger with verbosity:2), so you
// can see:
//   - the ggml device enumeration (how many devices, and their class),
//   - what `main-gpu` resolved to ("main-gpu resolved to backend 'CUDA0'", or
//     "no matching device found; leaving backend unset"),
//   - which backend was pinned, and
//   - per-step timing (GPU vs CPU is obvious: GPU is ~10-100x faster/step).
//
// Configure via environment variables:
//   MAIN_GPU   0|1|...|'integrated'|'dedicated'   (omit -> backend default)
//   DEVICE     'gpu' (default) | 'cpu'
//   STEPS      sampling steps (default 4 — keep small, we only probe selection)
//   MODEL      diffusion model filename under ../models
//   LLM        text-encoder/LLM filename under ../models (if the model needs it)
//   VAE        VAE filename under ../models (if the model needs it)
//   WIDTH,HEIGHT,SEED
//
// Example (NVIDIA multi-GPU host):
//   MAIN_GPU=dedicated bare examples/generate-image-main-gpu-probe.js
//   MAIN_GPU=integrated bare examples/generate-image-main-gpu-probe.js
//   MAIN_GPU=0 bare examples/generate-image-main-gpu-probe.js
//   DEVICE=cpu bare examples/generate-image-main-gpu-probe.js   # CPU baseline

const path = require('bare-path')
const process = require('bare-process')
const fs = require('bare-fs')
const binding = require('../binding')
const ImgStableDiffusion = require('../index')

const MODELS_DIR = path.resolve(__dirname, '../models')
const OUTPUT_DIR = path.resolve(__dirname, '../output')

const MODEL_NAME = process.env.MODEL || 'flux-2-klein-4b-Q8_0.gguf'
const LLM_MODEL = process.env.LLM || 'Qwen3-4B-Q4_K_M.gguf'
const VAE_MODEL = process.env.VAE || 'flux2-vae.safetensors'

const DEVICE = process.env.DEVICE || 'gpu'
const STEPS = Number(process.env.STEPS || 4)
const WIDTH = Number(process.env.WIDTH || 512)
const HEIGHT = Number(process.env.HEIGHT || 512)
const SEED = Number(process.env.SEED || 42)
const RAW_MAIN_GPU = process.env.MAIN_GPU

const PROMPT =
  'a compact workstation rendering a detailed image on a selected GPU'

function parseMainGpu (value) {
  if (value === undefined || value === '') return undefined
  if (value === 'integrated' || value === 'dedicated') return value
  const numeric = Number(value)
  if (Number.isInteger(numeric) && numeric >= 0) return numeric
  throw new Error(
    "MAIN_GPU must be a non-negative integer, 'integrated', or 'dedicated'"
  )
}

// Capture the addon's C++ backend-selection logs.
const cppLogs = []
binding.setLogger((priority, message) => {
  const line = String(message)
  cppLogs.push(line)
  if (/main-gpu|Backend selection|GPU device|no matching device/i.test(line)) {
    console.log(`  [C++] ${line}`)
  }
})

function findFile (name) {
  const p = path.join(MODELS_DIR, name)
  return fs.existsSync(p) ? p : null
}

async function main () {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  const mainGpu = parseMainGpu(RAW_MAIN_GPU)

  console.log('main-gpu probe')
  console.log('==============')
  console.log('device   :', DEVICE)
  console.log('main-gpu :', mainGpu === undefined ? '(unset -> backend default)' : mainGpu)
  console.log('model    :', MODEL_NAME)
  console.log('steps    :', STEPS, `(${WIDTH}x${HEIGHT})`)
  console.log()

  // Only pass files that actually exist so the probe works with whatever
  // model set is present on the test host.
  const files = { model: path.join(MODELS_DIR, MODEL_NAME) }
  const llm = findFile(LLM_MODEL)
  const vae = findFile(VAE_MODEL)
  if (llm) files.llm = llm
  if (vae) files.vae = vae

  const config = { device: DEVICE, threads: 4, diffusion_fa: true, verbosity: 2 }
  if (mainGpu !== undefined) config['main-gpu'] = mainGpu

  const model = new ImgStableDiffusion({
    files,
    config,
    opts: { stats: true },
    logger: console
  })

  try {
    const tLoad = Date.now()
    await model.load()
    const loadMs = Date.now() - tLoad
    console.log(`\nloaded in ${(loadMs / 1000).toFixed(1)}s`)

    const tGen = Date.now()
    const response = await model.run({
      prompt: PROMPT,
      steps: STEPS,
      width: WIDTH,
      height: HEIGHT,
      guidance: 3.5,
      seed: SEED
    })

    let stats = null
    let images = 0
    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) { images++; return }
        if (typeof data !== 'string') return
        try {
          const tick = JSON.parse(data)
          if ('backendDevice' in tick) stats = tick
        } catch (_) {}
      })
      .await()
    const genMs = Date.now() - tGen

    const resolved = cppLogs.find(
      (l) => l.includes('main-gpu resolved to backend') || l.includes('no matching device')
    )
    const pinned = cppLogs.find((l) => l.includes('main-gpu pinning stable-diffusion backend'))

    console.log('\n--- probe result ---')
    console.log('main-gpu      :', mainGpu === undefined ? '(unset)' : mainGpu)
    console.log('resolved      :', resolved || '(no main-gpu resolution log)')
    console.log('pinned backend:', pinned || '(not pinned — backend default used)')
    console.log('backendDevice :', stats && stats.backendDevice ? stats.backendDevice : '(n/a)')
    console.log('load time     :', `${(loadMs / 1000).toFixed(1)}s`)
    console.log('gen time      :', `${(genMs / 1000).toFixed(1)}s  (${(genMs / STEPS).toFixed(0)} ms/step)`)
    console.log('images        :', images)
  } finally {
    await model.unload()
    try { binding.releaseLogger() } catch (_) {}
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('probe FATAL:', err.message || err)
    process.exit(1)
  })
