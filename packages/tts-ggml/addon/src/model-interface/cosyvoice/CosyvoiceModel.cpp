#include "model-interface/cosyvoice/CosyvoiceModel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tts-cpp/cosyvoice/engine.h>

#include "addon/TTSErrors.hpp"
#include "inference-addon-cpp/Errors.hpp"
#include "model-interface/BackendUtils.hpp"

namespace qvac::ttsggml::cosyvoice {

namespace {

using qvac_errors::createTTSError;
using qvac_errors::StatusError;
using qvac_errors::tts_error::TTSErrorCode;
namespace general_error = qvac_errors::general_error;

tts_cpp::cosyvoice::EngineOptions toEngineOptions(const CosyvoiceConfig& cfg) {
  tts_cpp::cosyvoice::EngineOptions opts;
  opts.model_dir       = cfg.modelDir;
  opts.llm_gguf_path   = cfg.llmGgufPath;
  opts.flow_gguf_path  = cfg.flowGgufPath;
  opts.hift_gguf_path  = cfg.hiftGgufPath;
  opts.voice_gguf_path = cfg.voiceGgufPath;
  opts.vocab_path      = cfg.vocabPath;
  opts.merges_path     = cfg.mergesPath;
  if (!cfg.promptText.empty()) opts.prompt_text = cfg.promptText;
  if (!cfg.language.empty())   opts.language = cfg.language;
  if (cfg.steps.has_value())   opts.cfm_steps = *cfg.steps;
  if (cfg.seed.has_value())    opts.seed = *cfg.seed;
  if (cfg.threads.has_value()) opts.n_threads = *cfg.threads;
  opts.output_sample_rate = cfg.outputSampleRate.value_or(0);

  // Mirrors SupertonicModel::toEngineOptions: compose
  // `cfg.backendsDir / BACKENDS_SUBDIR` so a host passing
  // path.join(__dirname, 'prebuilds') gets the expected per-arch scan dir.
  if (!cfg.backendsDir.empty()) {
    std::filesystem::path backendsDirPath(cfg.backendsDir);
#ifdef BACKENDS_SUBDIR
    backendsDirPath =
        (backendsDirPath / std::filesystem::path(BACKENDS_SUBDIR)).lexically_normal();
#endif
    opts.backends_dir = backendsDirPath.string();
  }
  return opts;
}

std::vector<int16_t> pcmFloatToInt16(const float* pcm, size_t samples) {
  std::vector<int16_t> out;
  out.resize(samples);
  for (size_t i = 0; i < samples; ++i) {
    float s = std::clamp(pcm[i], -1.0f, 1.0f);
    out[i] = static_cast<int16_t>(std::lround(s * 32767.0f));
  }
  return out;
}

}

CosyvoiceModel::CosyvoiceModel(CosyvoiceConfig config)
    : cfg_(std::move(config)) {
  validateConfig(cfg_);
  // load() is deferred to waitForLoadInitialization() so the GGUF parse runs
  // off the JS event loop via JsAsyncTask::run-driven addon.activate().
}

CosyvoiceModel::~CosyvoiceModel() noexcept = default;

void CosyvoiceModel::validateConfig(const CosyvoiceConfig& cfg) {
  // Either a model_dir (component discovery) or an explicit LM path is required;
  // the engine constructor validates the full set of components and throws a
  // precise message for whatever is still missing.
  if (cfg.modelDir.empty() && cfg.llmGgufPath.empty()) {
    throw StatusError(general_error::InvalidArgument,
                      "cosyvoice: modelDir (or explicit component paths) is required");
  }
  if (!cfg.modelDir.empty() && !std::filesystem::is_directory(cfg.modelDir)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound,
                         "cosyvoice modelDir not found: " + cfg.modelDir);
  }
  if (cfg.steps.has_value() && *cfg.steps < 0) {
    throw StatusError(general_error::InvalidArgument, "steps must be >= 0");
  }
}

void CosyvoiceModel::load() {
  std::lock_guard lk(engineMu_);
  loadLocked();
}

void CosyvoiceModel::unload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
}

void CosyvoiceModel::reload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
  loadLocked();
}

void CosyvoiceModel::loadLocked() {
  if (engine_) return;

  try {
    engine_ = std::make_shared<tts_cpp::cosyvoice::Engine>(toEngineOptions(cfg_));
  } catch (const std::exception& e) {
    engine_.reset();
    throw createTTSError(TTSErrorCode::InitializationFailed,
                         std::string("CosyvoiceModel::load: ") + e.what());
  }

  backendName_    = engine_->backend_name();
  backendDevice_  = backendDeviceCode(engine_->backend_device());
  backendId_      = backendIdFromName(backendName_);
  gpuUnsupported_ = engine_->gpu_unsupported();
}

void CosyvoiceModel::unloadLocked() {
  engine_.reset();
}

void CosyvoiceModel::cancel() const {
  cancelRequested_.store(true, std::memory_order_relaxed);
  std::shared_ptr<tts_cpp::cosyvoice::Engine> e;
  {
    std::lock_guard lk(engineMu_);
    e = engine_;
  }
  if (e) e->cancel();
}

CosyvoiceModel::Output CosyvoiceModel::synthesize(const std::string& text) {
  std::shared_ptr<tts_cpp::cosyvoice::Engine> engine;
  {
    std::lock_guard lk(engineMu_);
    engine = engine_;
  }
  if (!engine) {
    throw createTTSError(TTSErrorCode::InitializationFailed,
                         "CosyvoiceModel::synthesize: engine not loaded");
  }
  if (cancelRequested_.load(std::memory_order_relaxed)) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         "synthesis cancelled before it started");
  }

  textLength_ = text.size();

  const auto t0 = std::chrono::steady_clock::now();
  tts_cpp::cosyvoice::SynthesisResult result;
  try {
    result = engine->synthesize(text);
  } catch (const std::exception& e) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         std::string("cosyvoice.synthesize: ") + e.what());
  }
  const auto t1 = std::chrono::steady_clock::now();

  sampleRate_ = result.sample_rate;
  totalSamples_ = static_cast<int64_t>(result.pcm.size());
  audioDurationMs_ = result.duration_s > 0.0f
      ? result.duration_s * 1000.0
      : (sampleRate_ > 0 ? (static_cast<double>(totalSamples_) * 1000.0 /
                            static_cast<double>(sampleRate_))
                         : 0.0);
  totalTime_ = std::chrono::duration<double>(t1 - t0).count();
  realTimeFactor_ = audioDurationMs_ > 0.0
      ? (totalTime_ * 1000.0) / audioDurationMs_
      : 0.0;
  tokensPerSecond_ = totalTime_ > 0.0
      ? static_cast<double>(textLength_) / totalTime_
      : 0.0;

  return pcmFloatToInt16(result.pcm.data(), result.pcm.size());
}

std::any CosyvoiceModel::process(const std::any& input) {
  const auto* anyInput = std::any_cast<AnyInput>(&input);
  if (!anyInput) {
    throw StatusError(general_error::InvalidArgument,
                      "CosyvoiceModel::process: input must be AnyInput");
  }

  bool expected = false;
  if (!jobInProgress_.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel)) {
    throw StatusError(general_error::InternalError,
                      "CosyvoiceModel::process: job already in progress");
  }
  struct InProgressGuard {
    std::atomic_bool& flag;
    ~InProgressGuard() { flag.store(false, std::memory_order_release); }
  } guard{jobInProgress_};

  cancelRequested_.store(false, std::memory_order_relaxed);
  return std::any(synthesize(anyInput->text));
}

qvac_lib_inference_addon_cpp::RuntimeStats CosyvoiceModel::runtimeStats() const {
  qvac_lib_inference_addon_cpp::RuntimeStats stats;
  stats.emplace_back("totalTime", totalTime_);
  stats.emplace_back("tokensPerSecond", tokensPerSecond_);
  stats.emplace_back("realTimeFactor", realTimeFactor_);
  stats.emplace_back("audioDurationMs", audioDurationMs_);
  stats.emplace_back("totalSamples", totalSamples_);
  stats.emplace_back("backendDevice", static_cast<int64_t>(backendDevice_));
  stats.emplace_back("backendId",     static_cast<int64_t>(backendId_));
  stats.emplace_back("gpuUnsupported", static_cast<int64_t>(gpuUnsupported_));
  return stats;
}

}
