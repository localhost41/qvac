'use strict'

// Pure-JS tests — no native addon required (mirrors vla-ggml's
// test/unit/preprocess.test.js pattern).

const test = require('brittle')
const {
  SAMPLE_RATE, FPS, WINDOW_FRAMES, WINDOW_SAMPLES,
  framesForSamples, int16ToFloat32, StreamingWindow
} = require('../../addon.js')

test('constants match the LAM streaming configuration', (t) => {
  t.is(SAMPLE_RATE, 16000)
  t.is(FPS, 30)
  t.is(WINDOW_FRAMES, 64)
  t.is(WINDOW_SAMPLES, Math.floor(16000 * 64 / 30))
})

test('framesForSamples: ceil(n / sr * fps)', (t) => {
  t.is(framesForSamples(16000), 30)
  t.is(framesForSamples(32000), 60)
  t.is(framesForSamples(WINDOW_SAMPLES), 64)
  t.is(framesForSamples(1), 1)
})

test('int16ToFloat32 scales into [-1, 1)', (t) => {
  const out = int16ToFloat32(new Int16Array([0, 32767, -32768, 16384]))
  t.is(out.length, 4)
  t.is(out[0], 0)
  t.ok(Math.abs(out[1] - 32767 / 32768) < 1e-7)
  t.is(out[2], -1)
  t.is(out[3], 0.5)
})

test('StreamingWindow right-aligns chunks and tracks context', (t) => {
  const win = new StreamingWindow()
  const chunk = new Float32Array(5120).fill(0.25) // 320 ms

  const { window, emitFromFrame } = win.push(chunk)
  t.is(window.length, WINDOW_SAMPLES)
  // Left part is initial silence, right part is the chunk.
  t.is(window[0], 0)
  t.is(window[WINDOW_SAMPLES - 1], 0.25)
  t.is(window[WINDOW_SAMPLES - chunk.length], 0.25)
  t.is(window[WINDOW_SAMPLES - chunk.length - 1], 0)
  // start_frame = floor(64 - 5120/16000*30) = floor(64 - 9.6) = 54
  t.is(emitFromFrame, 54)

  // Second push: previous window shifts left.
  const chunk2 = new Float32Array(5120).fill(0.5)
  const second = win.push(chunk2)
  t.is(second.window[WINDOW_SAMPLES - 1], 0.5)
  t.is(second.window[WINDOW_SAMPLES - chunk2.length - 1], 0.25)

  // Oversized chunk throws.
  let threw = false
  try {
    win.push(new Float32Array(WINDOW_SAMPLES + 1))
  } catch (err) {
    threw = /exceeds window/.test(err.message)
  }
  t.ok(threw, 'oversized chunk throws RangeError')
})
