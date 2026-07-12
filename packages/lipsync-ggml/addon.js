'use strict'

// Pure-JS helpers for lipsync preprocessing / streaming window math. No
// dependency on the native `.bare` addon, so CI's ts-checks job can exercise
// these without a native build.

const SAMPLE_RATE = 16000
const FPS = 30
const WINDOW_FRAMES = 64
// Samples covered by one 64-frame model window (matches the upstream LAM
// streaming implementation: sr * frames // fps).
const WINDOW_SAMPLES = Math.floor(SAMPLE_RATE * WINDOW_FRAMES / FPS)

/**
 * Number of output frames the model produces for `numSamples` of 16 kHz
 * PCM: ceil(numSamples / sampleRate * fps).
 *
 * @param {number} numSamples
 * @returns {number}
 */
function framesForSamples (numSamples) {
  return Math.ceil(numSamples / SAMPLE_RATE * FPS)
}

/**
 * Convert interleaved signed 16-bit PCM to the f32 [-1, 1] format the model
 * consumes.
 *
 * @param {Int16Array} int16
 * @returns {Float32Array}
 */
function int16ToFloat32 (int16) {
  const out = new Float32Array(int16.length)
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 32768
  return out
}

/**
 * Rolling streaming context implementing the upstream LAM windowing policy:
 * every chunk is right-aligned into a fixed 64-frame window whose left part
 * is the most recent previous audio (zeros initially), and only the frames
 * belonging to the new chunk are emitted.
 */
class StreamingWindow {
  constructor () {
    this.previousAudio = new Float32Array(WINDOW_SAMPLES) // zeros = silence
  }

  /**
   * Build the model input window for a chunk (chunk.length must be
   * <= WINDOW_SAMPLES; split larger chunks first).
   *
   * @param {Float32Array} chunk
   * @returns {{ window: Float32Array, emitFromFrame: number }}
   */
  push (chunk) {
    if (chunk.length > WINDOW_SAMPLES) {
      throw new RangeError(`chunk length ${chunk.length} exceeds window ${WINDOW_SAMPLES}`)
    }
    const keep = WINDOW_SAMPLES - chunk.length
    const window = new Float32Array(WINDOW_SAMPLES)
    window.set(this.previousAudio.subarray(this.previousAudio.length - keep), 0)
    window.set(chunk, keep)
    this.previousAudio = window
    // Upstream: start_frame = int(64 - chunkSamples / sr * 30)
    const emitFromFrame = Math.floor(WINDOW_FRAMES - chunk.length / SAMPLE_RATE * FPS)
    return { window, emitFromFrame }
  }
}

module.exports = {
  SAMPLE_RATE,
  FPS,
  WINDOW_FRAMES,
  WINDOW_SAMPLES,
  framesForSamples,
  int16ToFloat32,
  StreamingWindow
}
