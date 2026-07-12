// Parity harness: runs the ggml LAM Audio2Expression implementation against
// the PyTorch golden fixtures produced by scripts/dump_lam_reference.py and
// reports the max abs difference for every tapped stage.
//
// Usage: lam_parity <model.gguf> <fixtures-dir> [tolerance]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <picojson/picojson.h>

#include "model-interface/lam_audio2expression.hpp"

namespace {

std::vector<float>
readBin(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(1);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

struct CompareResult {
  double maxAbs = 0.0;
  double meanAbs = 0.0;
};

// mode "direct": flat element-by-element (ggml ne0 == PT fastest dim).
// mode "trans":  pt[i0*p1+i1] vs gg[i1*p0+i0].
CompareResult
compare(const std::vector<float>& pt, const std::vector<float>& gg,
        int64_t p0, int64_t p1, bool trans) {
  CompareResult res;
  double sum = 0.0;
  for (int64_t i0 = 0; i0 < p0; ++i0) {
    for (int64_t i1 = 0; i1 < p1; ++i1) {
      const double a = pt[i0 * p1 + i1];
      const double b = trans ? gg[i1 * p0 + i0] : gg[i0 * p1 + i1];
      const double d = a > b ? a - b : b - a;
      sum += d;
      if (d > res.maxAbs) {
        res.maxAbs = d;
      }
    }
  }
  res.meanAbs = sum / static_cast<double>(p0 * p1);
  return res;
}

// Taps whose ggml layout is the transpose-compare case (PT (C,T) dumps that
// the graph keeps feature-major, plus interp_out which it keeps time-major).
bool
isTransposeCompared(const std::string& name) {
  return name == "interp_out" || name == "lam_proj" ||
         name == "ident_out" || name.rfind("ident_cnr_", 0) == 0 ||
         name.rfind("dec_cnr_", 0) == 0;
}

} // namespace

int
main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model.gguf> <fixtures-dir> [tolerance]\n",
            argv[0]);
    return 1;
  }
  const std::string modelPath = argv[1];
  const std::string fixDir = argv[2];
  const double tolerance = argc > 3 ? atof(argv[3]) : 1e-3;

  qvac_lib_infer_lipsync_ggml::LamAudio2Expression model;
  if (!model.load(modelPath)) {
    fprintf(stderr, "load failed: %s\n", model.lastError().c_str());
    return 1;
  }
  printf("model loaded: %u coeffs, %u enc layers, window %u frames\n",
         model.hparams().nCoeffs, model.hparams().encLayers,
         model.hparams().windowFrames);

  std::ifstream mf(fixDir + "/manifest.json");
  if (!mf) {
    fprintf(stderr, "cannot open %s/manifest.json\n", fixDir.c_str());
    return 1;
  }
  picojson::value manifest;
  const std::string err = picojson::parse(manifest, mf);
  if (!err.empty()) {
    fprintf(stderr, "manifest parse error: %s\n", err.c_str());
    return 1;
  }

  int failures = 0;
  for (const auto& caseVal : manifest.get("cases").get<picojson::array>()) {
    const std::string caseName = caseVal.get("case").get<std::string>();
    const uint32_t idIdx =
        static_cast<uint32_t>(caseVal.get("id_idx").get<double>());
    const auto& tensors = caseVal.get("tensors").get<picojson::object>();

    const auto& pcmInfo = tensors.at("input_pcm");
    const std::vector<float> pcm =
        readBin(fixDir + "/" + pcmInfo.get("file").get<std::string>());

    printf("\n=== case %s: %zu samples, id_idx=%u ===\n", caseName.c_str(),
           pcm.size(), idIdx);

    std::vector<float> frames;
    std::map<std::string, std::vector<float>> taps;
    if (!model.run(pcm, idIdx, frames, &taps)) {
      fprintf(stderr, "run failed: %s\n", model.lastError().c_str());
      return 1;
    }

    for (const auto& [name, info] : tensors) {
      if (name == "input_pcm" || name == "id_onehot") {
        continue;
      }
      // ident_out is the same tensor as ident_cnr_2 in the reference model.
      const std::string tapName = name == "ident_out" ? "ident_cnr_2" : name;
      const auto it = taps.find(tapName);
      if (it == taps.end()) {
        printf("  %-14s MISSING TAP\n", name.c_str());
        ++failures;
        continue;
      }
      const std::vector<float> ref =
          readBin(fixDir + "/" + info.get("file").get<std::string>());
      const auto& shape = info.get("shape").get<picojson::array>();
      const int64_t p0 = static_cast<int64_t>(shape[0].get<double>());
      const int64_t p1 = shape.size() > 1
                             ? static_cast<int64_t>(shape[1].get<double>())
                             : 1;
      if (ref.size() != it->second.size()) {
        printf("  %-14s SIZE MISMATCH ref=%zu gg=%zu\n", name.c_str(),
               ref.size(), it->second.size());
        ++failures;
        continue;
      }
      const CompareResult res = compare(ref, it->second, p0, p1,
                                        isTransposeCompared(name));
      const bool pass = res.maxAbs <= tolerance;
      printf("  %-14s max|d|=%.3e mean|d|=%.3e %s\n", name.c_str(),
             res.maxAbs, res.meanAbs, pass ? "OK" : "FAIL");
      if (!pass) {
        ++failures;
      }
    }
  }

  printf("\n%s (%d failures, tolerance %.1e)\n",
         failures == 0 ? "PARITY PASS" : "PARITY FAIL", failures, tolerance);

  // Rough benchmark: repeated 64-frame windows (the streaming shape).
  {
    const int64_t winSamples = 16000LL * 64 / 30;
    const std::vector<float> pcm(static_cast<size_t>(winSamples), 0.01F);
    std::vector<float> frames;
    model.run(pcm, 0, frames, nullptr); // warm-up
    const int reps = 10;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
      model.run(pcm, 0, frames, nullptr);
    }
    const auto end = std::chrono::steady_clock::now();
    const double msPerRun =
        std::chrono::duration<double, std::milli>(end - start).count() / reps;
    const double windowSec =
        static_cast<double>(winSamples) / model.hparams().sampleRate;
    printf("bench: %.1f ms per %.2f s window → RTF %.1fx realtime\n",
           msPerRun, windowSec, windowSec * 1000.0 / msPerRun);
  }
  return failures == 0 ? 0 : 1;
}
