#pragma once

#include <any>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/RuntimeStats.hpp>

#include "model-interface/lam_audio2expression.hpp"
#include "model-interface/model_factory.hpp"

namespace qvac_lib_infer_lipsync_ggml {

// One inference call's worth of input. The framework's JobRunner runs
// process() on a worker thread after the JS callback has already returned,
// so the PCM buffer must be an owned copy.
struct LipsyncInput {
  std::vector<float> pcm; // 16 kHz mono f32
  uint32_t idIdx = 0;     // identity one-hot index, 0..nIdentity-1
};

// Owns a loaded LAM Audio2Expression model. Implements the canonical IModel
// interface so it plugs into the framework's job runner / output dispatch
// (same pattern as VlaModel in vla-ggml).
class LipsyncModel : public qvac_lib_inference_addon_cpp::model::IModel {
public:
  explicit LipsyncModel(
      const std::string& ggufPath, bool forceCpu = false,
      std::string backendsDir = {})
      : model_(createLipsyncModelFromGguf(ggufPath, forceCpu, backendsDir)) {}

  LipsyncModel(const LipsyncModel&) = delete;
  LipsyncModel& operator=(const LipsyncModel&) = delete;

  ~LipsyncModel() override = default;

  const LamHParams& hparams() const { return model_->hparams(); }

  // CPU-first policy (see model_factory.hpp).
  std::string backendName() const { return "CPU"; }

  // ─── IModel interface ────────────────────────────────────────────────────

  [[nodiscard]] std::string getName() const final { return "LipsyncModel"; }

  // Invoked on the JobRunner worker thread. Returns a std::vector<float> of
  // frameCount * nCoeffs blendshape coefficients (frame-major, ARKit-52
  // order, sigmoid range [0,1]) that the framework's
  // JsTypedArrayOutputHandler<float> converts into a JS Float32Array.
  std::any process(const std::any& input) final {
    const LipsyncInput* in = std::any_cast<LipsyncInput>(&input);
    if (in == nullptr) {
      throw std::invalid_argument(
          "LipsyncModel::process: input is not a LipsyncInput");
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<float> frames;
    if (!model_->run(in->pcm, in->idIdx, frames, nullptr)) {
      throw std::runtime_error(
          "lipsync inference failed: " + model_->lastError());
    }
    const auto end = std::chrono::steady_clock::now();

    // Worker thread is the only writer; runtimeStats() is read on the same
    // thread right after process() returns (OutputQueue::queueJobEnded).
    lastTotalMs_ =
        std::chrono::duration<double, std::milli>(end - start).count();
    lastFrames_ = static_cast<int64_t>(frames.size() / hparams().nCoeffs);
    return std::any{std::move(frames)};
  }

  [[nodiscard]] qvac_lib_inference_addon_cpp::RuntimeStats
  runtimeStats() const final {
    return {
        {"total_ms", lastTotalMs_},
        {"frames", lastFrames_},
        {"backendDevice", int64_t{0}}}; // CPU-first (0=CPU, 1=GPU)
  }

private:
  std::unique_ptr<LamAudio2Expression> model_;
  double lastTotalMs_ = 0.0;
  int64_t lastFrames_ = 0;
};

} // namespace qvac_lib_infer_lipsync_ggml
