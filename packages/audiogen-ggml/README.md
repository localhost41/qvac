# @qvac/audiogen-ggml

Audio generation (music) addon for qvac, ggml backend. Text prompt in, stereo
48 kHz audio out. Powered by the [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
pipeline (text-encoder → LM → DiT → VAE), compiled natively per-platform and
linked against `tts-cpp` via vcpkg — the same shape as `@qvac/tts-ggml`.

> **Status: v0 scaffold.** The native addon structure, build wiring and stable
> JS API are in place. Generation is gated on the `tts_cpp::acestep::Engine`
> port (DiT/LM/text-encoder onto our ggml-speech fork, CPU-first). Only the VAE
> stage is ported so far. See
> `tts-cpp/docs/QVAC-21921-audiogen-plan.md` for the roadmap.
>
> No binaries: the addon compiles the C++ for every platform tts-cpp supports.

## Architecture

```
index.js ── binding (BARE_MODULE) ── AcestepModel ── tts_cpp::acestep::Engine
                                                          │
                       text-enc ─► LM ─► DiT ─► VAE   (ggml graphs, CPU-first)
```

- `addon/src/js-interface/binding.cpp` — `BARE_MODULE` exports.
- `addon/src/addon/AddonJs.hpp` — `createInstance` / `activate` / `runJob`.
- `addon/src/model-interface/acestep/` — `AcestepModel` implements the
  `qvac_lib_inference_addon_cpp` model interface, wrapping the engine.
- Build: `cmake-bare` + `cmake-vcpkg`, `find_package(tts-cpp)`,
  `add_bare_module`. `vcpkg.json` depends on `tts-cpp`.

## Build

```bash
npm install
npm run build      # bare-make generate && build && install
```

## Usage

```js
const { AudioGen } = require('@qvac/audiogen-ggml')

const gen = new AudioGen({ modelDir: '/path/to/acestep/models' })
await gen.activate()

const { outputArray, sampleRate, channels } = await gen.generate(
  'lo-fi hip hop, mellow piano, rainy night',
  { lyrics: '[Instrumental]' }
)
// outputArray: interleaved stereo Int16 @ sampleRate

await gen.destroy()
```

## Models (smallest working set, ~3.8 GB)

| Stage | GGUF | Size |
|------|------|------|
| text-enc | Qwen3-Embedding-0.6B-Q8_0 | 748 MB |
| LM | acestep-5Hz-lm-0.6B-Q8_0 | 710 MB |
| DiT | acestep-v15-turbo-Q8_0 | 2.4 GB |
| VAE | vae-BF16 | 322 MB |

## Roadmap

- [x] Native package scaffold + stable JS API + build wiring
- [x] Custom ggml ops (`col2im_1d`, `snake`) + VAE stage in tts-cpp
- [ ] `tts_cpp::acestep::Engine`: DiT → LM → text-encoder port (CPU)
- [ ] Flip `AcestepModel` to real engine calls; bump tts-cpp vcpkg REF
- [ ] GPU acceleration (Metal / CUDA / Vulkan); streaming + cancel

## License

Apache-2.0. Model weights belong to ACE Studio and StepFun.
