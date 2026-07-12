// Compose final video from CDP screencast frames using exact media-clock
// retiming. Usage: node compose-cdp.cjs <wav> <out.mp4> [leadSec] [tailSec]
const fs = require('fs')
const { execFileSync } = require('child_process')

const SCRATCH = '/tmp/claude-1000/-home-olya-claude-folders-lipsync/cb27102d-ad5f-4aaa-8c5b-45f3122305ce/scratchpad'
const OUTDIR = SCRATCH + '/cdp-frames'
const WAV = process.argv[2]
const OUT = process.argv[3]
const LEAD = Number(process.argv[4] || 0.8)
const TAIL = Number(process.argv[5] || 0.8)
const FPS = 30
const CROP = process.env.CROP || '480:480:154:247'

const meta = JSON.parse(fs.readFileSync(OUTDIR + '/meta.json', 'utf8'))

// media time m corresponds to wall time playWall + startupLag + m.
// startupLag: median of (wall - playWall) - media over the steady samples.
const lags = meta.samples
  .map(([w, m]) => (w - meta.playWall) - m)
  .filter((l) => l >= 0 && l < 0.5)
  .sort((a, b) => a - b)
const lag = lags[Math.floor(lags.length / 2)]
console.log('startup lag (s):', lag.toFixed(3), 'of', meta.samples.length, 'samples')

// media duration = last steady sample's media time (clip end)
const mediaEnd = Math.max(...meta.samples.map(([, m]) => m))

// CDP delivery order is not strictly timestamp order (compositor jitter);
// an unsorted inversion strands the greedy pointer and freezes the video.
const frames = [...meta.frames].sort((a, b) => a.ts - b.ts)
const list = []
let fi = 0
const nOut = Math.round((LEAD + mediaEnd + TAIL) * FPS)
for (let k = 0; k < nOut; k++) {
  const m = k / FPS - LEAD // media time of this output frame (<0 = pre-play idle)
  const wall = meta.playWall + lag + m
  while (fi + 1 < frames.length && Math.abs(frames[fi + 1].ts - wall) <= Math.abs(frames[fi].ts - wall)) fi++
  list.push(`file '${OUTDIR}/${frames[fi].file}'`)
  list.push(`duration ${(1 / FPS).toFixed(6)}`)
}
fs.writeFileSync(SCRATCH + '/concat.txt', 'ffconcat version 1.0\n' + list.join('\n') + '\n')
console.log('output frames:', nOut, 'duration:', (nOut / FPS).toFixed(2), 's')

// sanity: every output frame should come from a fresh capture most of the time
const used = list.filter((l) => l.startsWith('file'))
console.log('unique capture frames used:', new Set(used).size, 'of', nOut)
let run = 1; let maxRun = 1
for (let i = 1; i < used.length; i++) {
  run = used[i] === used[i - 1] ? run + 1 : 1
  if (run > maxRun) maxRun = run
}
console.log('longest repeated-frame run:', maxRun, `(${(maxRun / FPS).toFixed(2)} s)`)
if (maxRun > FPS) throw new Error('video freezes for >1 s — mapping is broken, not encoding')

execFileSync('ffmpeg', [
  '-y', '-loglevel', 'error',
  '-f', 'concat', '-safe', '0', '-i', SCRATCH + '/concat.txt',
  '-itsoffset', String(LEAD), '-i', WAV,
  '-map', '0:v', '-map', '1:a',
  '-vf', `crop=${CROP}`, '-r', String(FPS),
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
  '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
  '-t', (nOut / FPS).toFixed(3),
  OUT
], { stdio: 'inherit' })
console.log('wrote', OUT)
