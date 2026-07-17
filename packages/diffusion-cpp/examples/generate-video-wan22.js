'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const process = require('bare-process')
const VideoStableDiffusion = require('../video')

// Download first:
//   ./scripts/download-model-wan2.2.sh
//   ./scripts/download-model-wan2.2.sh --t2v-a14b
//
// WAN22_VARIANT=ti2v-5b (default) uses the dense Wan 2.2 model.
// WAN22_VARIANT=t2v-a14b uses the two-expert MoE model.
const MODELS_DIR = path.resolve(__dirname, '../models')
const OUTPUT_DIR = path.resolve(__dirname, '../output')
const VARIANT = process.env.WAN22_VARIANT || 'ti2v-5b'
const isA14b = VARIANT === 't2v-a14b'

if (VARIANT !== 'ti2v-5b' && !isA14b) {
  throw new Error(`WAN22_VARIANT must be "ti2v-5b" or "t2v-a14b", got: ${VARIANT}`)
}

const PRECISION = process.env.WAN22_PRECISION || 'fp16'
if (!isA14b && PRECISION !== 'fp16') {
  throw new Error('The bundled TI2V-5B downloader publishes fp16 weights only.')
}
if (isA14b && PRECISION !== 'fp16' && PRECISION !== 'fp8') {
  throw new Error(`WAN22_PRECISION must be "fp16" or "fp8", got: ${PRECISION}`)
}

const expertSuffix = PRECISION === 'fp8' ? 'fp8_scaled' : 'fp16'
const files = isA14b
  ? {
      model: path.join(MODELS_DIR, `wan2.2_t2v_low_noise_14B_${expertSuffix}.safetensors`),
      highNoiseDiffusionModel: path.join(
        MODELS_DIR,
        `wan2.2_t2v_high_noise_14B_${expertSuffix}.safetensors`
      ),
      t5Xxl: path.join(
        MODELS_DIR,
        PRECISION === 'fp8'
          ? 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
          : 'umt5_xxl_fp16.safetensors'
      ),
      vae: path.join(MODELS_DIR, 'wan_2.1_vae.safetensors')
    }
  : {
      model: path.join(MODELS_DIR, 'wan2.2_ti2v_5B_fp16.safetensors'),
      t5Xxl: path.join(MODELS_DIR, 'umt5_xxl_fp16.safetensors'),
      vae: path.join(MODELS_DIR, 'wan2.2_vae.safetensors')
    }

const PROMPT =
  process.env.PROMPT || 'A cinematic drone shot over a misty mountain lake at sunrise, gentle motion'
const NEGATIVE_PROMPT = process.env.NEG_PROMPT || 'blurry, low quality, static, watermark'
const WIDTH = parseInt(process.env.WIDTH || '832', 10)
const HEIGHT = parseInt(process.env.HEIGHT || '480', 10)
const VIDEO_FRAMES = parseInt(process.env.FRAMES || '33', 10)
const FPS = parseInt(process.env.FPS || '16', 10)
const STEPS = parseInt(process.env.STEPS || (isA14b ? '50' : '40'), 10)
const CFG_SCALE = parseFloat(process.env.CFG_SCALE || '5.0')
const FLOW_SHIFT = parseFloat(process.env.FLOW_SHIFT || '5.0')
const SEED = parseInt(process.env.SEED || '-1', 10)
const MOE_BOUNDARY = parseFloat(process.env.MOE_BOUNDARY || '0.875')

function assertRunShape() {
  if (WIDTH <= 0 || HEIGHT <= 0 || WIDTH % 16 !== 0 || HEIGHT % 16 !== 0) {
    throw new Error(`WIDTH and HEIGHT must be positive multiples of 16, got ${WIDTH}x${HEIGHT}`)
  }
  if (VIDEO_FRAMES < 5 || (VIDEO_FRAMES - 1) % 4 !== 0) {
    throw new Error(`FRAMES must satisfy (4*k + 1), k >= 1, got ${VIDEO_FRAMES}`)
  }
}

async function main() {
  assertRunShape()
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  const model = new VideoStableDiffusion({
    files,
    config: {
      threads: 4,
      device: 'gpu',
      diffusion_fa: true,
      offload_to_cpu: true,
      vae_tiling: true
    },
    logger: console,
    opts: { stats: true }
  })

  try {
    console.log(`Loading Wan 2.2 ${isA14b ? 'T2V-A14B MoE' : 'TI2V-5B dense'}...`)
    await model.load()

    const params = {
      mode: 'txt2vid',
      prompt: PROMPT,
      negative_prompt: NEGATIVE_PROMPT,
      width: WIDTH,
      height: HEIGHT,
      video_frames: VIDEO_FRAMES,
      fps: FPS,
      steps: STEPS,
      cfg_scale: CFG_SCALE,
      flow_shift: FLOW_SHIFT,
      seed: SEED
    }
    if (isA14b) {
      params.high_noise_steps = parseInt(process.env.HIGH_NOISE_STEPS || String(STEPS), 10)
      params.high_noise_sampler = process.env.HIGH_NOISE_SAMPLER || 'euler'
      params.high_noise_scheduler = process.env.HIGH_NOISE_SCHEDULER || 'simple'
      params.high_noise_cfg_scale = parseFloat(process.env.HIGH_NOISE_CFG_SCALE || String(CFG_SCALE))
      params.high_noise_flow_shift = parseFloat(
        process.env.HIGH_NOISE_FLOW_SHIFT || String(FLOW_SHIFT)
      )
      // This is the native normalized timestep boundary, not an SNR threshold.
      params.moe_boundary = MOE_BOUNDARY
    }

    let avi = null
    const response = await model.run(params)
    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) avi = data
      })
      .await()

    if (!avi) throw new Error('No AVI output was received from the addon')
    const outputPath = path.join(OUTPUT_DIR, `wan22-${VARIANT}-seed${SEED}.avi`)
    fs.writeFileSync(outputPath, avi)
    console.log(`Saved → ${outputPath}`)
    console.log('Stats:', response.stats)
  } finally {
    await model.unload()
  }
}

main().catch((err) => {
  console.error('Fatal:', err.message || err)
  process.exit(1)
})
