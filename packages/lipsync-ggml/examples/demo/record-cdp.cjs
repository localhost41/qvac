// Frame-accurate demo recorder: raw CDP screencast + wall-clock retiming.
// Each screencast frame carries a compositor timestamp; media.play() wall
// time is captured in-page, so every frame maps exactly to media time.
// Output: frames/NNNNN.jpg + meta.json {frames:[{ts,file}], playWall, samples}
const pw = require('/home/olya/claude_folders/qvac-workbench/node_modules/playwright-core')
const fs = require('fs')

const SCRATCH = '/tmp/claude-1000/-home-olya-claude-folders-lipsync/cb27102d-ad5f-4aaa-8c5b-45f3122305ce/scratchpad'
const CAM = process.env.CAM || '0,1.74,0.52,0,1.58,0'
const CLIP = process.env.CLIP || '0'
const OUTDIR = SCRATCH + '/cdp-frames'
const URL = `http://localhost:8080/photoreal.html?clip=${CLIP}&cam=${CAM}`

async function main () {
  fs.rmSync(OUTDIR, { recursive: true, force: true })
  fs.mkdirSync(OUTDIR, { recursive: true })

  const browser = await pw.chromium.launch({
    executablePath: '/usr/bin/google-chrome',
    headless: false,
    args: [
      '--autoplay-policy=no-user-gesture-required',
      '--disable-backgrounding-occluded-windows',
      '--window-position=40,40',
      '--mute-audio'
    ],
    env: { ...process.env, DISPLAY: ':0' }
  })
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } })
  await page.goto(URL)
  await page.waitForFunction(() => document.title.includes('READY'), null, { timeout: 60000 })
  console.log('READY; settling 2s')
  await page.waitForTimeout(2000)

  const cdp = await page.context().newCDPSession(page)
  const frames = []
  let n = 0
  cdp.on('Page.screencastFrame', (ev) => {
    const file = String(n).padStart(5, '0') + '.jpg'
    fs.writeFileSync(OUTDIR + '/' + file, Buffer.from(ev.data, 'base64'))
    frames.push({ ts: ev.metadata.timestamp, file })
    n++
    cdp.send('Page.screencastFrameAck', { sessionId: ev.sessionId }).catch(() => {})
  })
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 90, everyNthFrame: 1 })
  await page.waitForTimeout(1000)

  // start playback; record wall time of play + periodic (wall, media) samples
  await page.evaluate(() => {
    window.__samples = []
    window.__media.play()
    window.__playWall = performance.timeOrigin / 1000 + performance.now() / 1000
    window.__t0 = window.__media.currentTime
    window.__iv = setInterval(() => {
      window.__samples.push([
        performance.timeOrigin / 1000 + performance.now() / 1000,
        window.__media.currentTime
      ])
    }, 500)
  })
  await page.waitForFunction(() => window.__media.ended, null, { timeout: 60000 })
  await page.waitForTimeout(700)
  await cdp.send('Page.stopScreencast')
  const info = await page.evaluate(() => {
    clearInterval(window.__iv)
    return { playWall: window.__playWall, t0: window.__t0, samples: window.__samples }
  })
  await browser.close()

  fs.writeFileSync(OUTDIR + '/meta.json', JSON.stringify({ ...info, frames }))
  // report drift: media time vs wall time since play
  const drift = info.samples.map(([w, m]) => (m - (w - info.playWall)).toFixed(3))
  console.log('frames:', frames.length)
  console.log('media-minus-wall drift (s):', drift.join(' '))
}

main().catch((e) => { console.error(e); process.exit(1) })
