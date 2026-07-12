'use strict'

// Integration test: loads the converted GGUF, runs the PyTorch golden PCM
// fixture through the native addon, and asserts numerical parity on the
// final 52-coefficient output (tolerance 1e-3 vs the reference dump — the
// tensor-level per-stage checks live in test/parity/lam_parity_main.cpp).
//
// Model file resolution (first match wins):
//   1. env LIPSYNC_GGUF_PATH
//   2. <package>/models/lam-audio2exp-f32.gguf (produced by
//      scripts/convert-lam-to-gguf.py)

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const LipsyncModel = require('../..')

const PKG_ROOT = path.join(__dirname, '..', '..')
const FIXTURES = path.join(PKG_ROOT, 'test', 'fixtures', 'reference')

function resolveModelPath () {
  const candidates = [
    typeof process !== 'undefined' && process.env && process.env.LIPSYNC_GGUF_PATH,
    path.join(PKG_ROOT, 'models', 'lam-audio2exp-f32.gguf')
  ].filter(Boolean)
  return candidates.find((p) => fs.existsSync(p)) || null
}

function readF32 (file) {
  const buf = fs.readFileSync(file)
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
}

function maxAbsDiff (a, b) {
  let max = 0
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i])
    if (d > max) max = d
  }
  return max
}

const modelPath = resolveModelPath()
const manifest = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'manifest.json'), 'utf8'))

test('lipsync-ggml matches the PyTorch reference on golden fixtures', async (t) => {
  t.ok(modelPath, `model GGUF present (${modelPath}) — run scripts/convert-lam-to-gguf.py first`)

  const model = new LipsyncModel({ files: { model: [modelPath] } })
  await model.load()

  t.is(model.hparams.nCoeffs, 52)
  t.is(model.hparams.fps, 30)
  t.is(model.hparams.sampleRate, 16000)
  t.is(model.hparams.coeffNames.length, 52)
  t.is(model.hparams.coeffNames[24], 'jawOpen')

  for (const c of manifest.cases) {
    const pcm = readF32(path.join(FIXTURES, c.tensors.input_pcm.file))
    const expected = readF32(path.join(FIXTURES, c.tensors.expr.file))

    const { frames, frameCount, fps } = await model.run({ pcm, idIdx: c.id_idx })
    t.is(frameCount, c.time_steps, `${c.case}: frame count`)
    t.is(fps, 30)
    t.is(frames.length, expected.length, `${c.case}: output size`)

    const diff = maxAbsDiff(frames, expected)
    t.ok(diff <= 1e-3, `${c.case}: max|diff| vs PyTorch = ${diff.toExponential(2)} <= 1e-3`)
  }

  await model.unload()
})

test('runStreaming emits the same total frame count as batch mode', async (t) => {
  t.ok(modelPath, 'model GGUF present')

  const model = new LipsyncModel({ files: { model: [modelPath] } })
  await model.load()

  const c = manifest.cases.find((x) => x.case === 'sec2')
  const pcm = readF32(path.join(FIXTURES, c.tensors.input_pcm.file))

  // 320 ms chunks — the recommended streaming hop.
  const CHUNK = 5120
  async function * chunks () {
    for (let off = 0; off < pcm.length; off += CHUNK) {
      yield pcm.subarray(off, Math.min(off + CHUNK, pcm.length))
    }
  }

  let total = 0
  let lastStart = -1
  for await (const batch of model.runStreaming(chunks(), { idIdx: 0 })) {
    t.ok(batch.frames instanceof Float32Array)
    t.ok(batch.startTimeMs > lastStart, 'monotonic startTimeMs')
    lastStart = batch.startTimeMs
    for (let i = 0; i < batch.frames.length; i++) {
      const v = batch.frames[i]
      if (!(v >= 0 && v <= 1)) {
        t.fail(`coefficient out of sigmoid range: ${v}`)
        break
      }
    }
    total += batch.frameCount
  }
  // Each 5120-sample chunk emits ~10 of its window's 64 frames; totals can
  // differ from the batch path by at most one frame per chunk boundary.
  const batchFrames = Math.ceil(pcm.length / 16000 * 30)
  t.ok(Math.abs(total - batchFrames) <= Math.ceil(pcm.length / CHUNK),
    `streaming total ${total} ≈ batch ${batchFrames}`)

  await model.unload()
})

test('load errors: missing file surfaces MODEL_NOT_FOUND', async (t) => {
  const model = new LipsyncModel({ files: { model: ['/nonexistent/model.gguf'] } })
  await t.exception(model.load(), /not found/i)
})
