#include "model-interface/parler/ParlerModel.hpp"

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

#include <tts-cpp/parler/description.h>
#include <tts-cpp/parler/engine.h>

#include "addon/TTSErrors.hpp"
#include "inference-addon-cpp/Errors.hpp"
#include "model-interface/BackendUtils.hpp"
#include "model-interface/OutputResampler.hpp"

namespace qvac::ttsggml::parler {

namespace {

using qvac_errors::createTTSError;
using qvac_errors::StatusError;
using qvac_errors::tts_error::TTSErrorCode;
namespace general_error = qvac_errors::general_error;

bool hasTemplateField(const ParlerDescriptionFields& d) {
  return !d.voice.empty() || !d.emotion.empty() || !d.pitch.empty() ||
         !d.pace.empty() || !d.expressivity.empty() || !d.noise.empty() ||
         !d.reverb.empty() || !d.quality.empty();
}

tts_cpp::parler::DescriptionSpec toSpec(const ParlerDescriptionFields& d) {
  tts_cpp::parler::DescriptionSpec spec;
  spec.voice = d.voice;
  spec.emotion = d.emotion;
  spec.pitch = d.pitch;
  spec.pace = d.pace;
  spec.expressivity = d.expressivity;
  spec.noise = d.noise;
  spec.reverb = d.reverb;
  spec.quality = d.quality;
  return spec;
}

tts_cpp::parler::EngineOptions toEngineOptions(const ParlerConfig& cfg) {
  tts_cpp::parler::EngineOptions opts;
  opts.model_gguf_path = cfg.modelGgufPath;
  opts.default_description = ParlerModel::resolveDescription(cfg.desc);
  if (cfg.seed.has_value())
    opts.seed = *cfg.seed;
  if (cfg.threads.has_value())
    opts.n_threads = *cfg.threads;
  if (cfg.temperature.has_value())
    opts.temperature = *cfg.temperature;
  if (cfg.topK.has_value())
    opts.top_k = *cfg.topK;
  if (cfg.topP.has_value())
    opts.top_p = *cfg.topP;
  if (cfg.maxFrames.has_value())
    opts.max_frames = *cfg.maxFrames;
  if (cfg.minNewTokens.has_value())
    opts.min_new_tokens = *cfg.minNewTokens;
  if (cfg.normalizeNumbers.has_value())
    opts.normalize_numbers = *cfg.normalizeNumbers;

  // Mirrors SupertonicModel::toEngineOptions: compose
  // `cfg.backendsDir / BACKENDS_SUBDIR` before forwarding.
  if (!cfg.backendsDir.empty()) {
    std::filesystem::path backendsDirPath(cfg.backendsDir);
#ifdef BACKENDS_SUBDIR
    backendsDirPath = (backendsDirPath / std::filesystem::path(BACKENDS_SUBDIR))
                          .lexically_normal();
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

} // namespace

ParlerModel::ParlerModel(ParlerConfig config) : cfg_(std::move(config)) {
  validateConfig(cfg_);
  // load() is deferred to waitForLoadInitialization(); see ChatterboxModel.
}

ParlerModel::~ParlerModel() noexcept = default;

std::string
ParlerModel::resolveDescription(const ParlerDescriptionFields& desc) {
  if (!desc.description.empty()) {
    if (hasTemplateField(desc)) {
      throw StatusError(
          general_error::InvalidArgument,
          "description is mutually exclusive with the voice/emotion/pitch/"
          "pace/expressivity/noise/reverb/quality template options");
    }
    return desc.description;
  }
  try {
    return tts_cpp::parler::build_description(toSpec(desc));
  } catch (const std::invalid_argument& e) {
    throw StatusError(general_error::InvalidArgument, e.what());
  }
}

void ParlerModel::validateConfig(const ParlerConfig& cfg) {
  if (cfg.modelGgufPath.empty()) {
    throw StatusError(
        general_error::InvalidArgument, "parlerModelPath is required");
  }
  if (!std::filesystem::exists(cfg.modelGgufPath)) {
    throw createTTSError(
        TTSErrorCode::ModelFileNotFound,
        "parler model not found: " + cfg.modelGgufPath);
  }
  // Same-level description/template conflict + template value validation.
  (void)resolveDescription(cfg.desc);
  if (cfg.temperature.has_value() && *cfg.temperature < 0.0f) {
    throw StatusError(
        general_error::InvalidArgument,
        "temperature must be >= 0 (0 = model default)");
  }
  if (cfg.topK.has_value() && *cfg.topK < 0) {
    throw StatusError(
        general_error::InvalidArgument,
        "topK must be >= 0 (0 = model default)");
  }
  if (cfg.topP.has_value() && (*cfg.topP <= 0.0f || *cfg.topP > 1.0f)) {
    throw StatusError(general_error::InvalidArgument, "topP must be in (0, 1]");
  }
  // ~86 decoder steps/s: a cap under 10 cannot fit even the delay-pattern
  // warmup, so reject it instead of synthesizing silence.
  if (cfg.maxFrames.has_value() &&
      (*cfg.maxFrames < 0 || (*cfg.maxFrames > 0 && *cfg.maxFrames <= 9))) {
    throw StatusError(
        general_error::InvalidArgument,
        "maxFrames must be 0 (model default) or > 9");
  }
  if (cfg.minNewTokens.has_value() && *cfg.minNewTokens < -1) {
    throw StatusError(
        general_error::InvalidArgument,
        "minNewTokens must be >= -1 (-1 = model default)");
  }
  if (cfg.outputSampleRate.has_value() && *cfg.outputSampleRate != 0 &&
      (*cfg.outputSampleRate < 8000 || *cfg.outputSampleRate > 192000)) {
    throw StatusError(
        general_error::InvalidArgument,
        "outputSampleRate must be 0 or in [8000, 192000]");
  }
}

void ParlerModel::setConfig(ParlerConfig config) {
  validateConfig(config);
  cfg_ = std::move(config);
}

void ParlerModel::load() {
  std::lock_guard lk(engineMu_);
  loadLocked();
}

void ParlerModel::unload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
}

void ParlerModel::reload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
  loadLocked();
}

void ParlerModel::loadLocked() {
  if (engine_)
    return;

  try {
    engine_ = std::make_shared<tts_cpp::parler::Engine>(toEngineOptions(cfg_));
  } catch (const std::exception& e) {
    engine_.reset();
    throw createTTSError(
        TTSErrorCode::InitializationFailed,
        std::string("ParlerModel::load: ") + e.what());
  }

  backendName_ = engine_->backend_name();
  backendDevice_ = backendDeviceCode(engine_->backend_device());
  backendId_ = backendIdFromName(backendName_);
}

void ParlerModel::unloadLocked() { engine_.reset(); }

void ParlerModel::cancel() const {
  cancelRequested_.store(true, std::memory_order_relaxed);
  std::shared_ptr<tts_cpp::parler::Engine> e;
  {
    std::lock_guard lk(engineMu_);
    e = engine_;
  }
  if (e)
    e->cancel();
}

ParlerModel::Output ParlerModel::synthesize(const AnyInput& input) {
  std::shared_ptr<tts_cpp::parler::Engine> engine;
  {
    std::lock_guard lk(engineMu_);
    engine = engine_;
  }
  if (!engine) {
    throw createTTSError(
        TTSErrorCode::InitializationFailed,
        "ParlerModel::synthesize: engine not loaded");
  }
  if (cancelRequested_.load(std::memory_order_relaxed)) {
    throw createTTSError(
        TTSErrorCode::SynthesisFailed, "synthesis cancelled before it started");
  }

  // Per-call description resolution: explicit text wins for this call;
  // template fields merge over the constructor-level template fields.
  // A constructor-level free-text description cannot be merged with
  // per-call template fields — that combination is rejected.
  std::string description;
  if (!input.desc.description.empty()) {
    description = resolveDescription(input.desc);
  } else if (hasTemplateField(input.desc)) {
    if (!cfg_.desc.description.empty()) {
      throw StatusError(
          general_error::InvalidArgument,
          "per-call template options (voice/emotion/...) cannot be combined "
          "with a constructor-level description; pass a per-call description "
          "instead");
    }
    ParlerDescriptionFields merged = cfg_.desc;
    if (!input.desc.voice.empty())
      merged.voice = input.desc.voice;
    if (!input.desc.emotion.empty())
      merged.emotion = input.desc.emotion;
    if (!input.desc.pitch.empty())
      merged.pitch = input.desc.pitch;
    if (!input.desc.pace.empty())
      merged.pace = input.desc.pace;
    if (!input.desc.expressivity.empty())
      merged.expressivity = input.desc.expressivity;
    if (!input.desc.noise.empty())
      merged.noise = input.desc.noise;
    if (!input.desc.reverb.empty())
      merged.reverb = input.desc.reverb;
    if (!input.desc.quality.empty())
      merged.quality = input.desc.quality;
    description = resolveDescription(merged);
  }

  textLength_ = input.text.size();

  const auto t0 = std::chrono::steady_clock::now();
  tts_cpp::parler::SynthesisResult result;
  try {
    result = description.empty() ? engine->synthesize(input.text)
                                 : engine->synthesize(input.text, description);
  } catch (const std::exception& e) {
    throw createTTSError(
        TTSErrorCode::SynthesisFailed,
        std::string("parler.synthesize: ") + e.what());
  }

  // The parler engine has no output-rate knob; resample addon-side.
  if (cfg_.outputSampleRate.has_value() && *cfg_.outputSampleRate > 0 &&
      *cfg_.outputSampleRate != result.sample_rate) {
    result.pcm = OutputResampler::resample(
        result.pcm, result.sample_rate, *cfg_.outputSampleRate);
    result.sample_rate = *cfg_.outputSampleRate;
  }

  const auto t1 = std::chrono::steady_clock::now();

  sampleRate_ = result.sample_rate;
  totalSamples_ = static_cast<int64_t>(result.pcm.size());
  audioDurationMs_ = sampleRate_ > 0
                         ? (static_cast<double>(totalSamples_) * 1000.0 /
                            static_cast<double>(sampleRate_))
                         : 0.0;
  totalTime_ = std::chrono::duration<double>(t1 - t0).count();
  realTimeFactor_ =
      audioDurationMs_ > 0.0 ? (totalTime_ * 1000.0) / audioDurationMs_ : 0.0;
  tokensPerSecond_ =
      totalTime_ > 0.0 ? static_cast<double>(textLength_) / totalTime_ : 0.0;

  return pcmFloatToInt16(result.pcm.data(), result.pcm.size());
}

std::any ParlerModel::process(const std::any& input) {
  const auto* anyInput = std::any_cast<AnyInput>(&input);
  if (!anyInput) {
    throw StatusError(
        general_error::InvalidArgument,
        "ParlerModel::process: input must be AnyInput");
  }

  bool expected = false;
  if (!jobInProgress_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    throw StatusError(
        general_error::InternalError,
        "ParlerModel::process: job already in progress");
  }
  struct InProgressGuard {
    std::atomic_bool& flag;
    ~InProgressGuard() { flag.store(false, std::memory_order_release); }
  } guard{jobInProgress_};

  cancelRequested_.store(false, std::memory_order_relaxed);
  return std::any(synthesize(*anyInput));
}

qvac_lib_inference_addon_cpp::RuntimeStats ParlerModel::runtimeStats() const {
  qvac_lib_inference_addon_cpp::RuntimeStats stats;
  stats.emplace_back("totalTime", totalTime_);
  stats.emplace_back("tokensPerSecond", tokensPerSecond_);
  stats.emplace_back("realTimeFactor", realTimeFactor_);
  stats.emplace_back("audioDurationMs", audioDurationMs_);
  stats.emplace_back("totalSamples", totalSamples_);
  stats.emplace_back("backendDevice", static_cast<int64_t>(backendDevice_));
  stats.emplace_back("backendId", static_cast<int64_t>(backendId_));
  // CPU is the validated backend for parler; report the same key the GPU
  // engines do so hosts can branch uniformly.
  stats.emplace_back("gpuUnsupported", static_cast<int64_t>(0));
  return stats;
}

} // namespace qvac::ttsggml::parler
