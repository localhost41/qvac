// Constructor-validation + scaffold-smoke tests for CosyvoiceModel.
//
// Same shape as test_supertonic_config.cpp: validateConfig is private so we
// drive it indirectly via the public constructor and assert the throw path.
//
// Unlike Supertonic (which needs a real GGUF to load), the iteration-1
// CosyVoice3 engine is a scaffold that tolerates missing weights and returns
// placeholder audio — so a valid model directory is enough to exercise the
// full construct -> load -> process path here.

#include <any>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inference-addon-cpp/Errors.hpp"
#include "model-interface/cosyvoice/CosyvoiceConfig.hpp"
#include "model-interface/cosyvoice/CosyvoiceModel.hpp"

using qvac::ttsggml::cosyvoice::CosyvoiceConfig;
using qvac::ttsggml::cosyvoice::CosyvoiceModel;
using qvac_errors::StatusError;

namespace {

std::filesystem::path testModelDir() {
  auto dir = std::filesystem::temp_directory_path() /
             "qvac-tts-ggml-cosyvoice-tests";
  std::filesystem::create_directories(dir);
  return dir;
}

CosyvoiceConfig minimallyValidStubConfig() {
  CosyvoiceConfig cfg;
  cfg.modelDir = testModelDir().string();  // exists; weights may be absent
  return cfg;
}

} // namespace

TEST(CosyvoiceValidate, EmptyConfigRejected) {
  CosyvoiceConfig cfg;
  EXPECT_THROW(CosyvoiceModel{cfg}, StatusError);
}

TEST(CosyvoiceValidate, NonexistentModelDirRejected) {
  CosyvoiceConfig cfg;
  cfg.modelDir = "/definitely/does/not/exist/cosyvoice3";
  EXPECT_THROW(CosyvoiceModel{cfg}, StatusError);
}

TEST(CosyvoiceValidate, NonexistentReferenceAudioRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.referenceAudio = "/definitely/does/not/exist/ref.wav";
  EXPECT_THROW(CosyvoiceModel{cfg}, StatusError);
}

TEST(CosyvoiceValidate, NegativeCfmStepsRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.cfmSteps = -1;
  EXPECT_THROW(CosyvoiceModel{cfg}, StatusError);
}

TEST(CosyvoiceValidate, UseGpuNGpuLayersConflictRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.useGpu = true;
  cfg.nGpuLayers = 0;
  EXPECT_THROW(CosyvoiceModel{cfg}, StatusError);
}

TEST(CosyvoiceValidate, ConfigDefaultsAreCpuFriendly) {
  CosyvoiceConfig cfg;
  EXPECT_EQ(cfg.language, "en");
  EXPECT_FALSE(cfg.useGpu.has_value());
  EXPECT_FALSE(cfg.nGpuLayers.has_value());
  EXPECT_FALSE(cfg.cfmSteps.has_value());
  EXPECT_FALSE(cfg.streamChunkTokens.has_value());
}

TEST(CosyvoiceScaffold, ConstructDefersLoad) {
  auto cfg = minimallyValidStubConfig();
  CosyvoiceModel m(cfg);
  EXPECT_EQ(m.getName(), "CosyvoiceModel");
  EXPECT_FALSE(m.isLoaded()) << "load is deferred until activate()/load()";
}

TEST(CosyvoiceScaffold, LoadAndSynthesizePlaceholder) {
  auto cfg = minimallyValidStubConfig();
  CosyvoiceModel m(cfg);
  EXPECT_NO_THROW(m.load());
  EXPECT_TRUE(m.isLoaded());
  EXPECT_EQ(m.sampleRate(), 24000);

  CosyvoiceModel::AnyInput input;
  input.text = "Hello world.";
  std::any out;
  EXPECT_NO_THROW(out = m.process(std::any(input)));
  const auto* pcm = std::any_cast<std::vector<int16_t>>(&out);
  ASSERT_NE(pcm, nullptr);
  EXPECT_GT(pcm->size(), 0u) << "placeholder waveform should be non-empty";

  EXPECT_NO_THROW(m.unload());
  EXPECT_FALSE(m.isLoaded());
}

TEST(CosyvoiceScaffold, ProcessRejectsWrongAnyInputType) {
  auto cfg = minimallyValidStubConfig();
  CosyvoiceModel m(cfg);
  m.load();
  EXPECT_THROW(m.process(std::any{int64_t{42}}), StatusError);
}

TEST(CosyvoiceScaffold, StreamingDeliversChunks) {
  auto cfg = minimallyValidStubConfig();
  cfg.streamChunkTokens = 25;       // ~1 s hops
  cfg.streamFirstChunkTokens = 10;  // smaller first chunk
  CosyvoiceModel m(cfg);
  m.load();

  int chunks = 0;
  bool sawLast = false;
  size_t streamedSamples = 0;
  CosyvoiceModel::AnyInput input;
  input.text = "Streaming placeholder synthesis over several chunks.";
  input.chunkCallback = [&](std::vector<int16_t>&& pcm, int idx, bool isLast) {
    EXPECT_EQ(idx, chunks);
    streamedSamples += pcm.size();
    ++chunks;
    if (isLast) sawLast = true;
  };

  std::any out;
  EXPECT_NO_THROW(out = m.process(std::any(std::move(input))));
  EXPECT_GT(chunks, 1) << "expected multiple streaming chunks";
  EXPECT_TRUE(sawLast);

  // The returned batch PCM must equal the concatenation of streamed chunks.
  const auto* full = std::any_cast<std::vector<int16_t>>(&out);
  ASSERT_NE(full, nullptr);
  EXPECT_EQ(full->size(), streamedSamples);
}
