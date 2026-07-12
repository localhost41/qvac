// Shape/geometry unit tests — no model file required.

#include <gtest/gtest.h>

#include "model-interface/lam_audio2expression.hpp"

using qvac_lib_infer_lipsync_ggml::LamAudio2Expression;

TEST(LamShapes, FrameCountIsCeilSamplesOverSrTimesFps) {
  LamAudio2Expression model; // hparams defaults: 16 kHz, 30 fps
  EXPECT_EQ(model.frameCount(16000), 30);
  EXPECT_EQ(model.frameCount(32000), 60);
  EXPECT_EQ(model.frameCount(16000 * 64 / 30), 64); // 34133-sample window
  EXPECT_EQ(model.frameCount(1), 1);
  EXPECT_EQ(model.frameCount(16001), 31);
}

TEST(LamShapes, ConvOutLenMatchesWav2vec2Geometry) {
  LamAudio2Expression model;
  // Verified against the PyTorch feature extractor:
  // 34133 samples → 106 frames @ 50 Hz, 32000 → 99.
  EXPECT_EQ(model.convOutLen(34133), 106);
  EXPECT_EQ(model.convOutLen(32000), 99);
  EXPECT_EQ(model.convOutLen(16000), 49);
}

TEST(LamShapes, HParamDefaultsMatchLamConfig) {
  LamAudio2Expression model;
  const auto& hp = model.hparams();
  EXPECT_EQ(hp.sampleRate, 16000U);
  EXPECT_EQ(hp.fps, 30U);
  EXPECT_EQ(hp.nCoeffs, 52U);
  EXPECT_EQ(hp.nIdentity, 12U);
  EXPECT_EQ(hp.hiddenDim, 512U);
  EXPECT_EQ(hp.encLayers, 12U);
  EXPECT_EQ(hp.encHeads, 12U);
  EXPECT_EQ(hp.encHidden, 768U);
  EXPECT_EQ(hp.windowFrames, 64U);
}
