# Tensor-level parity harness

Compares every stage of the ggml LAM Audio2Expression graph against the
PyTorch golden dumps in `test/fixtures/reference/` (produced by
`scripts/dump_lam_reference.py`) and prints a per-tap max-abs-diff table.
Exit code 0 = all taps within tolerance.

Standalone build (no bare/vcpkg needed — link any ggml with the standard
CPU backend, e.g. the one installed by the qvac-fabric vcpkg port):

```bash
GGML=/path/to/vcpkg-installed/x64-linux   # has include/ggml.h + lib/libggml-*.a
clang++ -O2 -std=c++20 \
  -I "$GGML/include" -I addon/src \
  test/parity/lam_parity_main.cpp \
  addon/src/model-interface/lam_audio2expression.cpp \
  -o /tmp/lam_parity \
  "$GGML/lib/libggml.a" "$GGML/lib/libggml-vulkan.a" \
  "$GGML/lib/libggml-cpu.a" "$GGML/lib/libggml-base.a" \
  -lvulkan -lm -lpthread
# (drop libggml-vulkan.a / -lvulkan if your ggml was built CPU-only)

/tmp/lam_parity models/lam-audio2exp-f32.gguf test/fixtures/reference 1e-3
```

Reference results (CPU):

| variant | per-stage max abs diff | final coefficients | tolerance |
|---|---|---|---|
| F32 | ≤ 6.5e-5 | ≤ 3e-6 | 1e-3 PASS |
| F16 | ≤ 1.8e-2 (large-magnitude pre-norm stages) | ≤ 1.1e-3 | 2e-2 PASS |

The same end-to-end check runs as a gtest in `test/unit/test_parity.cpp`
(via `npm run test:cpp`), skipping when the GGUF is absent.
