#pragma once

#include <atomic>

#include "inference-addon-cpp/Logger.hpp"

namespace qvac_lib_infer_lipsync_ggml {
namespace logging {

extern std::atomic<qvac_lib_inference_addon_cpp::logger::Priority>
    g_verbosityLevel;

} // namespace logging
} // namespace qvac_lib_infer_lipsync_ggml

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define QLOG_IF(priority, message)                                             \
  do {                                                                         \
    if (static_cast<int>(priority) <=                                          \
        static_cast<int>(                                                      \
            qvac_lib_infer_lipsync_ggml::logging::g_verbosityLevel.load(           \
                std::memory_order_relaxed))) {                                 \
      QLOG(priority, message);                                                 \
    }                                                                          \
  } while (0)
