// Golden-file parity test vs the PyTorch reference dumps produced by
// scripts/dump_lam_reference.py. Skips (does not fail) when the converted
// GGUF is absent; the fixtures themselves are committed.
//
// Full per-stage tap comparison lives in test/parity/lam_parity_main.cpp;
// this gtest checks the end-to-end 52-coefficient output at the 1e-3 gate.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "model-interface/lam_audio2expression.hpp"

namespace {

std::string modelPath() {
  if (const char* env = std::getenv("LIPSYNC_GGUF_PATH")) {
    return env;
  }
#ifdef LIPSYNC_DEFAULT_GGUF
  return LIPSYNC_DEFAULT_GGUF;
#else
  return "models/lam-audio2exp-f32.gguf";
#endif
}

std::string fixturesDir() {
#ifdef LIPSYNC_FIXTURES_DIR
  return LIPSYNC_FIXTURES_DIR;
#else
  return "test/fixtures/reference";
#endif
}

std::vector<float> readBin(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(file.good()) << "missing fixture: " << path;
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<float> data(static_cast<size_t>(size) / sizeof(float));
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

} // namespace

TEST(LamParity, EndToEndMatchesPyTorchGolden) {
  if (!std::filesystem::exists(modelPath())) {
    GTEST_SKIP() << "GGUF not found at " << modelPath()
                 << " — run scripts/convert-lam-to-gguf.py";
  }

  qvac_lib_infer_lipsync_ggml::LamAudio2Expression model;
  ASSERT_TRUE(model.load(modelPath())) << model.lastError();

  const struct {
    const char* name;
    int64_t frames;
  } cases[] = {{"win64", 64}, {"sec2", 60}};

  for (const auto& c : cases) {
    const std::vector<float> pcm =
        readBin(fixturesDir() + "/" + c.name + "_input_pcm.bin");
    const std::vector<float> expected =
        readBin(fixturesDir() + "/" + c.name + "_expr.bin");
    ASSERT_FALSE(pcm.empty());

    std::vector<float> frames;
    ASSERT_TRUE(model.run(pcm, 0, frames, nullptr)) << model.lastError();
    ASSERT_EQ(frames.size(), expected.size()) << c.name;
    ASSERT_EQ(
        static_cast<int64_t>(frames.size()) / model.hparams().nCoeffs,
        c.frames);

    float maxDiff = 0.0F;
    for (size_t i = 0; i < frames.size(); ++i) {
      maxDiff = std::max(maxDiff, std::fabs(frames[i] - expected[i]));
    }
    EXPECT_LE(maxDiff, 1e-3F) << c.name << ": ggml vs PyTorch";
  }
}
