#include "model_factory.hpp"

#include <stdexcept>

#include "../utils/BackendSelection.hpp"

namespace qvac_lib_infer_lipsync_ggml {

std::unique_ptr<LamAudio2Expression> createLipsyncModelFromGguf(
    const std::string& ggufPath, bool /*forceCpu*/,
    const std::string& backendsDir) {
  // Register backend plugins (Vulkan/Metal/…) for future opt-in GPU use;
  // harmless when only the built-in CPU backend exists.
  lipsync_backend_selection::loadBackendsOnce(backendsDir);

  auto model = std::make_unique<LamAudio2Expression>();
  // nullptr backend → the model initialises its own multithreaded CPU
  // backend (see CPU-first policy in model_factory.hpp).
  if (!model->load(ggufPath, nullptr)) {
    throw std::runtime_error(model->lastError());
  }
  return model;
}

} // namespace qvac_lib_infer_lipsync_ggml
