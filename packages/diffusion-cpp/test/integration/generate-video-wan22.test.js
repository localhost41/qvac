'use strict'

// Opt-in Wan 2.2 smoke tests.
//
// The model is intentionally not downloaded by ordinary integration runs:
// it is ~10 GB plus the shared UMT5 encoder. Enable it on a Linux/CUDA runner:
//
//   WAN22_RUN_SMOKE=true WAN22_MODELS_DIR=/path/to/models \
//     bare test/integration/generate-video-wan22.test.js --exit
//
// Add WAN22_RUN_A14B_SMOKE=true to also load and run the ~57 GB paired
// T2V-A14B MoE model.

const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const proc = require('bare-process')
const test = require('brittle')
const VideoStableDiffusion = require('@qvac/diffusion-cpp/video')
const { verifyLocalModelPath } = require('./utils')

const enabled = proc.env && proc.env.WAN22_RUN_SMOKE === 'true'
const a14bEnabled = proc.env && proc.env.WAN22_RUN_A14B_SMOKE === 'true'
const modelsDir = proc.env && proc.env.WAN22_MODELS_DIR
const skip = !enabled || !modelsDir || os.platform() === 'darwin'
const skipA14b = !a14bEnabled || !modelsDir || os.platform() === 'darwin'

const FILES = [
  'wan2.2_ti2v_5B_fp16.safetensors',
  'wan2.2_vae.safetensors',
  // This file is byte-identical to the Wan 2.1 manifest entry and is reused
  // by the downloader, so preserve the established integrity key.
  'umt5_xxl_fp16.safetensors'
]

function isAvi(buf) {
  return (
    buf instanceof Uint8Array &&
    buf.length > 64 &&
    String.fromCharCode(...buf.subarray(0, 4)) === 'RIFF' &&
    String.fromCharCode(...buf.subarray(8, 12)) === 'AVI '
  )
}

test(
  'Wan 2.2 TI2V-5B — smoke (txt2vid) generates a valid AVI',
  { timeout: 900000, skip },
  async (t) => {
    const resolved = {}
    for (const name of FILES) {
      const filePath = path.join(modelsDir, name)
      t.ok(fs.existsSync(filePath), `model file is present: ${name}`)
      await verifyLocalModelPath({ modelName: name, filePath })
      resolved[name] = filePath
    }

    const model = new VideoStableDiffusion({
      files: {
        model: resolved['wan2.2_ti2v_5B_fp16.safetensors'],
        vae: resolved['wan2.2_vae.safetensors'],
        t5Xxl: resolved['umt5_xxl_fp16.safetensors']
      },
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

    let avi = null
    try {
      await model.load()
      const response = await model.run({
        mode: 'txt2vid',
        prompt: 'a red fox running through snow at dusk',
        width: 416,
        height: 240,
        video_frames: 5,
        fps: 16,
        steps: 1,
        cfg_scale: 5.0,
        flow_shift: 5.0,
        seed: 7
      })
      await response
        .onUpdate((data) => {
          if (data instanceof Uint8Array) avi = data
        })
        .await()

      t.ok(isAvi(avi), 'received a RIFF/AVI output buffer')
      t.ok(response.stats && response.stats.videoFrames === 5, 'stats report five output frames')
    } finally {
      await model.unload()
    }
  }
)

test(
  'Wan 2.2 T2V-A14B — MoE smoke (txt2vid) generates a valid AVI',
  { timeout: 1800000, skip: skipA14b },
  async (t) => {
    const required = [
      'wan2.2_t2v_low_noise_14B_fp16.safetensors',
      'wan2.2_t2v_high_noise_14B_fp16.safetensors',
      'wan_2.1_vae.safetensors',
      'umt5_xxl_fp16.safetensors'
    ]
    const resolved = {}
    for (const name of required) {
      const filePath = path.join(modelsDir, name)
      t.ok(fs.existsSync(filePath), `model file is present: ${name}`)
      await verifyLocalModelPath({ modelName: name, filePath })
      resolved[name] = filePath
    }

    const model = new VideoStableDiffusion({
      files: {
        model: resolved['wan2.2_t2v_low_noise_14B_fp16.safetensors'],
        highNoiseDiffusionModel: resolved['wan2.2_t2v_high_noise_14B_fp16.safetensors'],
        vae: resolved['wan_2.1_vae.safetensors'],
        t5Xxl: resolved['umt5_xxl_fp16.safetensors']
      },
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

    let avi = null
    try {
      await model.load()
      const response = await model.run({
        mode: 'txt2vid',
        prompt: 'a red fox running through snow at dusk',
        width: 416,
        height: 240,
        video_frames: 5,
        fps: 16,
        steps: 1,
        high_noise_steps: 1,
        cfg_scale: 5.0,
        high_noise_cfg_scale: 5.0,
        flow_shift: 5.0,
        high_noise_flow_shift: 5.0,
        moe_boundary: 0.875,
        seed: 7
      })
      await response
        .onUpdate((data) => {
          if (data instanceof Uint8Array) avi = data
        })
        .await()

      t.ok(isAvi(avi), 'received a RIFF/AVI output buffer')
      t.ok(response.stats && response.stats.videoFrames === 5, 'stats report five output frames')
      t.ok(
        response.stats && response.stats.totalSteps >= 2,
        'stats include both high- and low-noise schedules'
      )
    } finally {
      await model.unload()
    }
  }
)
