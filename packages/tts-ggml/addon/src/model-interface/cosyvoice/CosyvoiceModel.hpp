#pragma once

#include <any>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inference-addon-cpp/ModelInterfaces.hpp"
#include "inference-addon-cpp/RuntimeStats.hpp"

#include "model-interface/cosyvoice/CosyvoiceConfig.hpp"

namespace tts_cpp::cosyvoice {
class Engine;
}

namespace qvac::ttsggml::cosyvoice {

// Addon model wrapper around tts_cpp::cosyvoice::Engine (Fun-CosyVoice3-0.5B /
// 1.5B).  Mirrors SupertonicModel: deferred async load, single in-flight job,
// cooperative cancel, RuntimeStats.  CPU-only (iteration 1); no LavaSR
// enhancer/denoiser path.
class CosyvoiceModel
    : public qvac_lib_inference_addon_cpp::model::IModel,
      public qvac_lib_inference_addon_cpp::model::IModelCancel,
      public qvac_lib_inference_addon_cpp::model::IModelAsyncLoad {
public:
  using Input = std::string;
  using Output = std::vector<int16_t>;

  struct AnyInput {
    std::string text;
  };

  explicit CosyvoiceModel(CosyvoiceConfig config);
  ~CosyvoiceModel() noexcept override;

  std::string getName() const override { return "CosyvoiceModel"; }
  std::any process(const std::any& input) override;
  qvac_lib_inference_addon_cpp::RuntimeStats runtimeStats() const override;

  void cancel() const override;

  void load();
  void unload();
  void reload();
  bool isLoaded() const {
    std::lock_guard lk(engineMu_);
    return static_cast<bool>(engine_);
  }

  // IModelAsyncLoad — AddonCpp::activate() (via JsAsyncTask::run) calls this on
  // a worker thread so the GGUF parse runs off the JS event loop; load() is
  // idempotent.
  void waitForLoadInitialization() override { load(); }
  void setWeightsForFile(
      const std::string&,
      std::unique_ptr<std::basic_streambuf<char>>&&) override {}

  void setConfig(CosyvoiceConfig config) { cfg_ = std::move(config); }
  const CosyvoiceConfig& config() const { return cfg_; }

  int sampleRate() const { return sampleRate_; }

private:
  Output synthesize(const std::string& text);
  static void validateConfig(const CosyvoiceConfig& cfg);

  void loadLocked();
  void unloadLocked();

  CosyvoiceConfig cfg_;

  mutable std::mutex engineMu_;
  std::shared_ptr<tts_cpp::cosyvoice::Engine> engine_;

  std::atomic_bool jobInProgress_{false};

  // A JS-side cancel issued between run() calls sets this; process() consumes
  // it on entry so a stale cancel doesn't poison the next synthesis.  cancel()
  // also forwards to the engine.
  mutable std::atomic_bool cancelRequested_{false};

  double totalTime_ = 0.0;
  double audioDurationMs_ = 0.0;
  int64_t totalSamples_ = 0;
  double realTimeFactor_ = 0.0;
  double tokensPerSecond_ = 0.0;
  size_t textLength_ = 0;
  int sampleRate_ = 24000;

  int backendDevice_ = 0;
  int backendId_ = 0;
  bool gpuUnsupported_ = false;
  std::string backendName_ = "CPU";
};

}
