# Repository layout

Monorepo structure overview. All QVAC components live under `/packages`, including the SDK, libraries, and tooling. Not every component is published to npm.

Legend:
* **Core:** foundational building blocks shared across the ecosystem.
* **Addon:** capability packages — each QVAC capability is implemented by one or more addons.
* **SDK:** primary entry point for consumers.
* **Tool:** user-facing tools and services that support the ecosystem.

| Package | Description | Category |
| :--- | :--- | :--- |
| sdk | Main entry point to develop AI applications with QVAC | SDK |
| lib-decoder-audio | Audio decoder library leveraging FFmpeg for efficient audio decoding as preprocessing step for other addons | Addon |
| lib-infer-llamacpp-embed | Native C++ addon for running text embedding models to generate high-quality contextual embeddings via `qvac-fabric-llm.cpp` | Addon |
| lib-infer-llamacpp-llm | Native C++ addon for running Large Language Models (LLMs) via `qvac-fabric-llm.cpp` | Addon |
| diffusion-cpp | Native C++ addon for text-to-image generation via `qvac-ext-stable-diffusion.cpp` | Addon |
| lib-infer-nmtcpp | Native C++ addon for translation using either `qvac-fabric-llm.cpp` or [Bergamot](https://browser.mt) | Addon |
| lib-infer-onnx | Bare addon for ONNX Runtime session management | Addon |
| tts-ggml | Text-to-Speech (TTS) library using Chatterbox and Supertonic neural TTS models via the GGML backend | Addon |
| lib-infer-parakeet | High-performance speech-to-text inference addon using via NVIDIA/Parakeet | Addon |
| transcription-whispercpp | Library for running Whisper transcription model for audio transcription via `qvac-ext-lib-whisper.cpp` | Addon |
| inference-addon-cpp | Header-only C++ library providing common abstractions and infrastructure for building high-performance inference addons | Addon |
| langdetect-text | Language detection library providing interface for detecting language of given text | Addon |
| langdetect-text-cld2 | Language detection using CLD2 with same API as @qvac/langdetect-text | Addon |
| ocr-onnx | Optical Character Recognition (OCR) addon using ONNX Runtime | Addon |
| rag | JavaScript library for Retrieval-Augmented Generation (RAG) with document ingestion, vector search, and LLM integration | Addon |
| dl-base | Base class for QVAC dataloader libraries providing common interface for loading data from various sources | Core |
| dl-filesystem | Data loading library for loading model weights and resources from local filesystem | Core |
| dl-hyperdrive | Data loading library for loading model weights and resources from Hyperdrive distributed file system | Core |
| error | Standardized error handling capabilities for all QVAC libraries | Core |
| infer-base | Base class for inference addon clients defining common lifecycle and generic methods for model interaction | Core |
| logging | Logger wrapper that normalizes logging interface across QVAC libraries | Core |
| cli | Command-line interface for the QVAC ecosystem with tooling for building, bundling, and managing QVAC-powered applications | Tool |
| diagnostics | Diagnostic report generation library for QVAC | Tool |
| lib-registry-server | Distributed model registry for downloading AI models for local inference and contributing new models | Tool |
| lint-cpp | Configuration files for formatting and linting C++ source files with pre-commit hooks | Tool |