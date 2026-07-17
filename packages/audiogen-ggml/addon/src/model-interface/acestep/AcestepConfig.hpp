#pragma once

#include <optional>
#include <string>

namespace qvac::audiogenggml::acestep {

// Mirrors tts_cpp::acestep::EngineOptions, populated by JSAdapter from the JS
// `configuration` object. Either set `modelDir` (auto-classify the four GGUFs
// by architecture) or the explicit per-stage paths (explicit wins).
struct AcestepConfig {
  std::string modelDir;

  std::string textEncModelPath;  // Qwen3-Embedding-*.gguf
  std::string lmModelPath;       // acestep-5Hz-lm-*.gguf
  std::string ditModelPath;      // acestep-v15-*.gguf
  std::string vaeModelPath;      // vae-*.gguf

  std::optional<int>  inferenceSteps;  // unset/0 = auto (turbo 8, base/sft 50)
  std::optional<float> shift;          // unset/0 = auto (turbo 3.0, base/sft 1.0)
  std::optional<int>  seed;
  std::optional<int>  threads;
  std::optional<bool> useGpu;
  std::optional<int>  nGpuLayers;

  std::string backendsDir;
};

}  // namespace qvac::audiogenggml::acestep
