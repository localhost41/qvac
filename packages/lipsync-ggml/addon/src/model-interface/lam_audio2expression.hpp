#pragma once

// LAM Audio2Expression on ggml — audio (16 kHz f32 PCM) → ARKit-52
// blendshape coefficients at 30 fps.
//
// Architecture (upstream: aigc3d/LAM_Audio2Expression, Apache-2.0):
//   wav2vec2-base feature extractor (7× conv1d, group-norm on layer 0, GELU)
//   → linear interpolation 50 Hz → 30 fps (align_corners=true)
//   → feature projection (LayerNorm + Linear 512→768)
//   → 12-layer post-norm transformer encoder (768d, 12 heads, conv pos-emb)
//   → LAM head: Linear 768→512, identity-conditioned conv stack
//     (3× ConvNormRelu, LayerNorm, residual), 3× ConvNormRelu decoder,
//     Linear 512→52, sigmoid.
//
// Numerical parity with the PyTorch reference is enforced by
// test/parity (tolerance 1e-3 on every stage, f32 weights).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <ggml-backend.h>
#include <ggml.h>

namespace qvac_lib_infer_lipsync_ggml {

struct LamHParams {
  uint32_t sampleRate = 16000;
  uint32_t fps = 30;
  uint32_t nCoeffs = 52;
  uint32_t nIdentity = 12;
  uint32_t identityFeatDim = 64;
  uint32_t hiddenDim = 512;
  uint32_t windowFrames = 64;
  float layerNormEps = 1e-5F;
  uint32_t encLayers = 12;
  uint32_t encHeads = 12;
  uint32_t encHidden = 768;
  uint32_t encFfn = 3072;
  uint32_t posConvKernel = 128;
  uint32_t posConvGroups = 16;
  std::vector<int32_t> feKernels{10, 3, 3, 3, 3, 2, 2};
  std::vector<int32_t> feStrides{5, 2, 2, 2, 2, 2, 2};
  std::vector<std::string> coeffNames;
};

class LamAudio2Expression {
public:
  LamAudio2Expression() = default;
  ~LamAudio2Expression();

  LamAudio2Expression(const LamAudio2Expression&) = delete;
  LamAudio2Expression& operator=(const LamAudio2Expression&) = delete;
  LamAudio2Expression(LamAudio2Expression&&) = delete;
  LamAudio2Expression& operator=(LamAudio2Expression&&) = delete;

  // Load a lam-audio2exp GGUF onto the given backend (CPU if null).
  // Returns false and sets lastError() on failure.
  bool load(const std::string& ggufPath, ggml_backend_t backend = nullptr);

  // Number of output frames for a given PCM length: ceil(n / sr * fps).
  [[nodiscard]] int64_t frameCount(int64_t nSamples) const;

  // Feature-extractor output length for a given PCM length (7 conv stages,
  // floor((len - k) / s) + 1 each).
  [[nodiscard]] int64_t convOutLen(int64_t nSamples) const;

  // Run the full model. pcm is 16 kHz mono f32; idIdx selects the identity
  // one-hot (0..nIdentity-1). Output: frames*nCoeffs floats, frame-major.
  // When taps is non-null, every named intermediate tensor is copied into
  // it (name → row-major values) for parity testing.
  bool run(const std::vector<float>& pcm, uint32_t idIdx,
           std::vector<float>& framesOut,
           std::map<std::string, std::vector<float>>* taps = nullptr);

  [[nodiscard]] const LamHParams& hparams() const { return hparams_; }
  [[nodiscard]] const std::string& lastError() const { return lastError_; }

private:
  struct ggml_tensor* weight(const std::string& name);

  LamHParams hparams_;
  std::string lastError_;

  ggml_backend_t backend_ = nullptr;
  bool ownsBackend_ = false;
  ggml_backend_buffer_t weightBuffer_ = nullptr;
  struct ggml_context* weightCtx_ = nullptr;
  std::map<std::string, struct ggml_tensor*> weights_;
};

} // namespace qvac_lib_infer_lipsync_ggml
