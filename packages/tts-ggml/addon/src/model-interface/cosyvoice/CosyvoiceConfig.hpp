#pragma once

#include <optional>
#include <string>

namespace qvac::ttsggml::cosyvoice {

// Configuration for the CosyVoice3 (Fun-CosyVoice3-0.5B / 1.5B) engine.
//
// CosyVoice3 ships as a small set of GGUFs plus a Qwen2 BPE tokenizer and a
// baked default voice.  Point `modelDir` at the folder holding them (the
// layout produced by scripts/assemble-cosyvoice3-model.py in
// qvac-ext-lib-whisper.cpp/tts-cpp):
//
//   <modelDir>/
//     cosyvoice3-llm-f32.gguf
//     cosyvoice3-flow-f32.gguf
//     cosyvoice3-hift-f32.gguf
//     voice.gguf
//     vocab.json
//     merges.txt
//
// or set the per-component paths explicitly (they win over modelDir discovery).
//
// Iteration 1 is CPU-only, so there are no GPU knobs here (unlike Supertonic /
// Chatterbox); they will be added when a GPU path lands.
struct CosyvoiceConfig {
  std::string modelDir;

  // Explicit component paths (optional; override modelDir discovery).
  std::string llmGgufPath;
  std::string flowGgufPath;
  std::string hiftGgufPath;
  std::string voiceGgufPath;
  std::string vocabPath;
  std::string mergesPath;

  // LM prompt transcript for the baked voice.  Empty -> the engine's built-in
  // default (matches the shipped voice.gguf).
  std::string promptText;

  std::string language = "en";

  // Flow-matching Euler step count (0/unset -> model default, 10).
  std::optional<int> steps;
  std::optional<int> seed;
  std::optional<int> threads;

  // Desired output sample rate in Hz (8000–192000), or unset/0 to keep the
  // engine's native 24 kHz.  Forwarded to EngineOptions::output_sample_rate.
  std::optional<int> outputSampleRate;

  // Directory to scan for dynamically-loaded ggml backends (forwarded to
  // EngineOptions::backends_dir).  No-op on the CPU-only iteration-1 path but
  // plumbed for parity with the sibling engines.
  std::string backendsDir;
};

}
