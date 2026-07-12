// Port of LAM_Audio2Expression's apply_random_brow_movement (models/utils.py):
// volume-gated canned brow-raise curves added into the 5 brow channels
// (browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight
// — columns 0..4, same order as the model's coeffNames).
// Deviation from upstream: an 8-frame linear fade-out after each inserted
// animation; upstream leaves the raw step and relies on a later
// Savitzky-Golay smoothing pass that our pipeline doesn't run.
// Usage: node apply-brows.cjs <frames.json> <audio.wav 16k mono s16> <out.json> [seed]
const fs = require('fs')

const SCRATCH = '/tmp/claude-1000/-home-olya-claude-folders-lipsync/cb27102d-ad5f-4aaa-8c5b-45f3122305ce/scratchpad'
const CURVES = JSON.parse(fs.readFileSync(SCRATCH + '/brow-curves.json', 'utf8'))
const BROWS = [CURVES.BROW1, CURVES.BROW2]

const FRAME_SEGMENT = 150
const HOLD_THRESHOLD = 10
const VOLUME_THRESHOLD = 0.08
const MIN_REGION_LENGTH = 6
// upstream range is (0.7, 1.3); the OpenAvatarChat avatars' brow
// calibration is muted, so raises below ~0.6 barely read — boosted range
// (values still clipped to 1.0 per channel on write).
const STRENGTH_RANGE = [1.2, 2.0]
const FADE_FRAMES = 8

// deterministic PRNG so regenerating a clip is reproducible
let seed = Number(process.argv[5] || 42)
function rand () {
  seed = (seed * 1664525 + 1013904223) >>> 0
  return seed / 4294967296
}

const frames = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const wav = fs.readFileSync(process.argv[3])
const sr = wav.readUInt32LE(24)
const pcm = wav.subarray(44)
const nSamp = Math.floor(pcm.length / 2)

// volume: RMS per animation frame (librosa rms with hop = frame = sr/30)
const hop = Math.floor(sr / frames.fps)
const N = frames.frames.length
const volume = new Float64Array(N)
for (let i = 0; i < N; i++) {
  let acc = 0; let n = 0
  for (let s = i * hop; s < Math.min((i + 1) * hop, nSamp); s++, n++) {
    const v = pcm.readInt16LE(s * 2) / 32768
    acc += v * v
  }
  volume[i] = n ? Math.sqrt(acc / n) : 0
}

// upstream's 0.08 threshold assumes normalized speech; our TTS is quieter.
// Scale the envelope so its 90th percentile sits at 0.12 (typical speech).
const sorted = [...volume].sort((a, b) => a - b)
const p90 = sorted[Math.floor(0.9 * sorted.length)] || 1
const scale = 0.12 / p90
for (let i = 0; i < N; i++) volume[i] *= scale

const exp = frames.frames // [N][52], brows are cols 0..4
const peakIdx = BROWS.map(c => c.reduce((bi, row, i, a) => row[2] > a[bi][2] ? i : bi, 0))
let events = 0

for (let segStart = 0; segStart < N; segStart += FRAME_SEGMENT) {
  const segEnd = Math.min(segStart + FRAME_SEGMENT, N)

  // contiguous runs of volume > threshold, length >= MIN_REGION_LENGTH
  const regions = []
  let runStart = -1
  for (let i = segStart; i <= segEnd; i++) {
    const hi = i < segEnd && volume[i] > VOLUME_THRESHOLD
    if (hi && runStart < 0) runStart = i
    if (!hi && runStart >= 0) {
      if (i - runStart >= MIN_REGION_LENGTH) regions.push([runStart, i - 1])
      runStart = -1
    }
  }
  if (!regions.length) continue

  const [rs, re] = regions[Math.floor(rand() * regions.length)]
  const regionLength = re - rs + 1
  const bi = Math.floor(rand() * 2)
  const curve = BROWS[bi]
  const pk = peakIdx[bi]
  const strength = STRENGTH_RANGE[0] + rand() * (STRENGTH_RANGE[1] - STRENGTH_RANGE[0])

  const add = (fi, row, s) => {
    if (fi < 0 || fi >= N) return
    for (let c = 0; c < 5; c++) exp[fi][c] = Math.min(1, Math.max(0, exp[fi][c] + row[c] * s))
  }

  let animEnd = -1
  if (regionLength > HOLD_THRESHOLD) {
    // align curve peak with the loudest frame of the region, hold to region end
    let lm = 0
    for (let i = rs; i <= re; i++) if (volume[i] > volume[rs + lm]) lm = i - rs
    const peakFrame = rs + lm
    const rise = curve.slice(0, pk + 1)
    const insertStart = Math.max(peakFrame - pk, segStart)
    const insertEnd = Math.min(peakFrame + (regionLength - lm), segEnd)
    if (insertStart + rise.length <= segEnd) {
      rise.forEach((row, i) => add(insertStart + i, row, strength))
      for (let i = insertStart + rise.length; i < insertEnd; i++) add(i, curve[pk], strength)
      animEnd = insertEnd
      events++
    }
  } else {
    const L = curve.length
    let pos = segStart + rs - segStart + Math.floor((regionLength - L) / 2)
    pos = Math.max(segStart, Math.min(pos, segEnd - L))
    if (pos + L <= segEnd) {
      curve.forEach((row, i) => add(pos + i, row, strength))
      animEnd = pos + L
      events++
    }
  }

  // fade-out: decay from the last written values to the pre-existing ones
  if (animEnd > 0 && animEnd < N) {
    const last = exp[animEnd - 1].slice(0, 5)
    for (let i = 0; i < FADE_FRAMES && animEnd + i < N; i++) {
      const w = 1 - (i + 1) / (FADE_FRAMES + 1)
      for (let c = 0; c < 5; c++) {
        exp[animEnd + i][c] = Math.min(1, Math.max(0, exp[animEnd + i][c] + last[c] * w))
      }
    }
  }
}

fs.writeFileSync(process.argv[4], JSON.stringify(frames))
console.log(`brow events inserted: ${events} over ${N} frames (${(N / frames.fps).toFixed(1)} s)`)
