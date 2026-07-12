# @qvac/lipsync-ggml

Audio → ARKit-52 blendshape lipsync addon for QVAC, running
[LAM Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression)
(Apache-2.0) on ggml. 16 kHz mono PCM in, 52 sigmoid blendshape
coefficients per frame out at 30 fps — drives a VRM/ARKit avatar's mouth
(and brows/eyes) from any speech audio, e.g. streaming TTS output.

## Architecture

```
f32 PCM 16 kHz
  → wav2vec2-base feature extractor   (7× conv1d, group-norm, GELU-erf)
  → linear interpolation 50 Hz → 30 fps (align_corners, exact matmul)
  → feature projection                (LayerNorm + Linear 512→768)
  → 12-layer transformer encoder      (768d, 12 heads, conv pos-embedding)
  → LAM head                          (Linear 768→512, identity-conditioned
                                       conv stack, 3× ConvNormRelu decoder)
  → Linear 512→52 + sigmoid           (ARKit-52 coefficients per frame)
```

~102 M params. CPU-first: multiple× realtime on desktop CPU; Vulkan/Metal
backend plugins ship with the prebuild for future opt-in.

## Numerical parity vs PyTorch

The ggml implementation is verified tensor-by-tensor against the upstream
PyTorch model:

- `scripts/dump_lam_reference.py` runs the PyTorch reference (HF
  transformers wav2vec2 + faithfully mirrored LAM head) and dumps every
  stage's activations for two deterministic audio cases.
- `test/parity/lam_parity_main.cpp` runs the same inputs through the ggml
  graph and compares **every intermediate** (feature extractor, each of the
  12 encoder layers, head convs, final coefficients).

Measured (F32 GGUF, CPU): max abs diff ≤ 6.5e-5 on every stage; final
coefficients ≤ 3e-6 — far inside the 1e-3 gate. The F16 variant stays
within ~1.1e-3 on the final output (use F32 where parity matters; F16 for
half the download size).

## Model conversion

```bash
# 1. Fetch the upstream checkpoint (Apache-2.0)
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar
tar -xf LAM_audio2exp_streaming.tar   # → pretrained_models/lam_audio2exp_streaming.tar

# 2. Convert to GGUF
python3 scripts/convert-lam-to-gguf.py \
  --checkpoint pretrained_models/lam_audio2exp_streaming.tar \
  --out models/lam-audio2exp-f32.gguf --dtype f32
```

The GGUF is self-describing: sample rate, fps, coefficient count, window
config and the ARKit-52 name table are stored as metadata.

## Build

Same toolchain as the other native addons (see repo root CLAUDE.md):

```bash
npm install
bare-make generate
bare-make build
bare-make install
```

## Usage

```js
const LipsyncModel = require('@qvac/lipsync-ggml')

const model = new LipsyncModel({ files: { model: ['/abs/path/lam-audio2exp-f32.gguf'] } })
await model.load()

// One-shot
const { frames, frameCount, fps } = await model.run({ pcm, idIdx: 0 })
// frames: Float32Array, frameCount × 52, frame-major, values in [0, 1]
// model.hparams.coeffNames — ARKit-52 blendshape order

// Streaming (e.g. teeing TTS PCM): 320 ms chunks recommended
for await (const batch of model.runStreaming(pcmChunks)) {
  render(batch.frames, batch.startTimeMs)
}

await model.unload()
```

Streaming follows the upstream LAM policy: each chunk is right-aligned in a
rolling 64-frame (~2.13 s) context window and only the chunk's new frames
are emitted. Latency = chunk length + one window inference. Delay audio
playback by one chunk for exact sync.

Raw model output is unsmoothed. The upstream post-processing (Savitzky-
Golay smoothing, blendshape symmetrization, procedural blinks) is
renderer-side policy and intentionally not part of the addon.

## Tests

- `npm run test:unit` — pure-JS helpers (no native build needed).
- `npm run test:integration` — golden-file parity through the full addon
  (needs the converted GGUF at `models/lam-audio2exp-f32.gguf` or
  `LIPSYNC_GGUF_PATH`).
- `test/parity/` — C++ tensor-level parity harness (see file header for
  the standalone build line).
