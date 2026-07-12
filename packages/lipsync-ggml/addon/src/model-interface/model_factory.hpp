#pragma once

#include <memory>
#include <string>

#include "lam_audio2expression.hpp"

namespace qvac_lib_infer_lipsync_ggml {

// Create and load a lipsync model from a GGUF file, dispatching on the
// GGUF `general.architecture` key (currently only "lam-audio2exp").
//
// CPU-first policy: the model is small enough that CPU runs multiple times
// realtime, and the graph is validated for exact PyTorch parity on the CPU
// backend. GPU backends (Vulkan/Metal) are registered via loadBackendsOnce
// so a future opt-in can pick them without further plumbing, but until the
// op coverage is validated per-backend we do not silently place the graph
// on a GPU.
//
// Throws std::runtime_error on failure.
std::unique_ptr<LamAudio2Expression> createLipsyncModelFromGguf(
    const std::string& ggufPath, bool forceCpu, const std::string& backendsDir);

} // namespace qvac_lib_infer_lipsync_ggml
