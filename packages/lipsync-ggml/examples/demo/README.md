# Lipsync browser demos

## photoreal.html — photorealistic Gaussian-splat avatar (LAM)

Feeds the ARKit-52 coefficient stream 1:1 into a photorealistic 3D
Gaussian-splat head reconstructed from a single photo by LAM (aigc3d,
SIGGRAPH 2025) — the companion project of the LAM_Audio2Expression model
this addon ports. Assets:

- `lam-render-libs.js` — `gaussian-splat-renderer-for-lam@0.0.9-alpha.1`
  (MIT) module build, verbatim from npm (self-contained, three.js bundled).
- `avatar-lam.zip` — sample avatar "p2-1" from the MIT-licensed
  `aigc3d/LAM_WebRender` repo (`asset/arkit/p2-1.zip`), repacked to strip
  macOS junk so the renderer finds the inner `p2-1/` folder. To use your
  own photo, generate an avatar with the LAM project (its Hugging Face /
  ModelScope demo exports this zip format) and drop it in as
  `avatar-lam.zip`.

Run: `python3 -m http.server -d . 8080` → http://localhost:8080/photoreal.html

The renderer polls `getExpressionData()` every frame for a
`{arkitName: weight}` map and applies per-avatar calibration
(`offset/scale`) internally; the page adds a light attack/release EMA and
procedural blinks on top of the raw model output, and clocks frames off
the `<audio>` element. Body gestures come from the asset's bundled
animation clips, switched by chat state (`Idle` when paused, `Responding`
while speaking). `?frame=N` pins a frame; `?cam=px,py,pz,tx,ty,tz`
re-frames the bust (the asset's splat cloud sits at head height around
y≈1.65).

**Needs a real GPU.** Under software WebGL (headless Chrome/SwiftShader)
the per-frame gaussian depth-sort produces garbage and the dense head
splats vanish — only blurry shoulder blobs render. On hardware GL it is
correct and fast. Keep that in mind for kiosk hardware and CI screenshots.

`?autoplay=1` starts playback ~2.5 s after the avatar reports READY (pair
with Chrome's `--autoplay-policy=no-user-gesture-required`); the page also
exposes `window.__media` for scripted control. `?avatar=<zip>` picks the
avatar asset: `avatar-lam.zip` (default, "p2-1" — stylized, weak lip
calibration: no `bsData.json`) or `james.zip` / `barbara.zip` (photoreal
LAM samples from Apache-2.0-licensed HumanAIGC-Engineering/OpenAvatarChat, with
per-blendshape calibration — far stronger articulation).

**Filling the non-mouth channels.** The model only drives ~21 of the 52
channels (all mouth/nose); eyes and brows come out as zeros and are meant
to be layered on afterwards, mirroring upstream:

- Blinks + gaze saccades: procedural in the page (`?gaze=0` disables gaze).
- Brows: `node apply-brows.cjs <frames.json> <wav> <out.json> [seed]` —
  a port of upstream's `apply_random_brow_movement` (volume-gated canned
  brow raises, curves in `brow-curves.json`), run it on the frames JSON
  before bundling into `data.js`. Two documented deviations: an 8-frame
  fade-out after each event (upstream relies on a Savitzky-Golay pass we
  don't run), and a boosted strength range (the OpenAvatarChat avatars'
  brow calibration is muted — upstream's 0.7–1.3 barely reads).

**Custom avatars from a photo:** LAM one-shot reconstruction runs on
`qvac-dev-linux-x64` (needs CUDA; see `~/olya-lam/` there):
`~/olya-lam/make-avatar.sh photo.jpg` → OpenAvatarChat zip in ~1 min.
Drop the zip here and load with `?avatar=<name>.zip`. Generated avatars
sit lower in world space than the bundled samples — translate the camera
rig down (e.g. `cam=0,1.72,0.62,0,1.42,0` vs james's
`0,1.86,0.62,0,1.56,0`). `avatar-tether-custom.mp4` was made this way.

**Lipsync chunk-size trap:** feed the model chunks that are a whole number
of 30 fps frames (multiples of 1600 samples @ 16 kHz). The model slices a
64-frame window with `int(64 - samples/sr*30)`, so a fractional chunk like
5120 samples (9.6 frames) emits 10 frames per chunk — ~4 % cumulative A/V
drift (lips ~1 s late by the end of a 30 s clip). `say.ts` now uses 4800.
The JFK / hello clips committed in `data.js` predate this fix and still
carry the drift; the Tether clips are clean.

Recording a video (`avatar-tether-short.mp4` was made this way):

```bash
CAM="0,1.72,0.78,0,1.55,0" CLIP=0 node record-cdp.cjs   # capture
CROP="520:480:130:247" node compose-cdp.cjs clip.wav out.mp4
```

`record-cdp.cjs` drives headed Chrome (hardware GL — the splat sort
degrades on software WebGL) and captures via CDP `Page.startScreencast`,
which reads page content directly, so other windows covering it don't
matter. Do NOT use Playwright's `recordVideo` — its webm timestamps drift
~3 % vs the media clock, which visibly desyncs lips over a 30 s clip.
The capture stores each frame's compositor wall-clock timestamp plus the
in-page `play()` wall time and (wall, currentTime) samples; `compose-cdp.cjs`
retimes to exact 30 fps against the media clock and muxes the original
WAV. Camera: keep the eye above head height (y ≳ 1.7) or the head reads
as filmed from below; the Responding gesture also sits the head lower
than the Idle pose you frame against, so pick the final crop from a
captured Responding frame, not a paused screenshot.

## vrm.html — 3D VRM avatar (the real evaluation target)

Drives a VRM avatar's expression presets (aa/ih/ou/ee/oh visemes, blinks,
brows) from the ARKit-52 coefficient stream — the proper way to judge the
model, since the rig consumes the coefficients natively. Assets:

- `vrm-libs.js` — three@0.164 + GLTFLoader + @pixiv/three-vrm@3.5.4 bundled
  to one script (`bun build entry.js --bundle --minify --format=iife`).
- `avatar.vrm` — "Seed-san" © VirtualCast, Inc. (VRM-Consortium sample,
  VRM Public License 1.0: redistribution + modification allowed, credit
  required). Textures downscaled for size; regenerate with
  `python3 fetch-avatar.py`.

Run: `python3 -m http.server -d . 8080` → http://localhost:8080/vrm.html
(a server is needed because the page fetches `avatar.vrm`).

The mapping is the plan's interim path: openness/rounding/wideness
compounds → VRM viseme mix. The production mascot VRM will expose all 52
ARKit blendshapes ("PerfectSync") and consume the stream 1:1.

## index.html — photo-puppet portrait

A real photograph animated by the LAM Audio2Expression coefficients,
audio-synced in the browser: JFK's official White House portrait (public
domain, `portrait.jpg`) speaking his own inaugural line as a 2D photo
puppet — the lower lip/jaw region is warped down by the mouth-openness
coefficients while the photo's upper teeth stay static, with a painted
oral cavity in the gap. `index.html` + `data.js` are fully static — open
`index.html` directly (no server needed).

This is a stand-in for the production renderer (3D VRM head consuming the
same ARKit-52 stream); the point here is judging lip-sync timing and
articulation against real speech.

## Create a clip from text (QVAC TTS → lipsync)

```bash
cd packages/sdk
bun run examples/lipsync/say.ts \
  --text "Hello! Ask me anything." \
  --lipsync /abs/path/lam-audio2exp-f32.gguf \
  --out-wav /tmp/say.wav --out-frames /tmp/say-frames.json
```

Downloads Supertonic TTS via the model registry on first use, synthesizes
the text, resamples 44.1 kHz → 16 kHz, and runs lipsync — the full
text→speech→blendshapes pipeline through the SDK. Then bundle any number
of clips (the page shows a clip selector when there is more than one):

```bash
python3 examples/demo/make-demo-data.py --out examples/demo/data.js \
  --portrait examples/demo/portrait.jpg \
  --clip "JFK inaugural, 1961" jfk-frames.json jfk.wav \
  --clip "Hello! I am the QVAC avatar" /tmp/say-frames.json /tmp/say.wav
```

## Regenerate for any WAV (16 kHz mono s16)

```bash
# 1. frames via the full QVAC SDK stack (loadModel → bare worker → addon)
cd packages/sdk
bun run examples/lipsync/lipsync-frames.ts \
  /abs/path/lam-audio2exp-f32.gguf /abs/path/speech.wav /tmp/frames.json

#    …or directly against the addon, no SDK:
cd packages/lipsync-ggml
bare examples/wav-to-frames-json.js models/lam-audio2exp-f32.gguf speech.wav /tmp/frames.json

# 2. bundle frames + audio (+ portrait) for the page
python3 examples/demo/make-demo-data.py /tmp/frames.json speech.wav examples/demo/data.js examples/demo/portrait.jpg
```

To swap in a different portrait, update the mouth/eye coordinates at the
top of the renderer section in `index.html` (`LIP`, `jawPath`, the blink
ellipses) — they are in the image's native pixel space.

The committed `data.js` was generated through the SDK path from
`packages/tts-ggml/test/reference-audio/jfk.wav` (11 s, 344 frames). Both
generation paths produce bit-identical coefficients.

Renderer notes: the page maps the ARKit-52 coefficients onto a 2D canvas
mascot (jaw/mouth-shape compound for the lips, brows, squint/wide eyes,
cheek puff). Coefficients are shown raw — no smoothing or symmetrization;
eye blinks are procedural in the renderer, mirroring the upstream design
where blinks are post-processing, not model output.
