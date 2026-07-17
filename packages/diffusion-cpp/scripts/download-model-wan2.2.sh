#!/usr/bin/env bash
set -euo pipefail

# Wan 2.2 video model downloader
#
# Downloads a complete inference layout for either:
#   --ti2v-5b  Dense text/image-to-video model (default)
#   --t2v-a14b Two-expert text-to-video MoE model
#
# The A14B path needs both high- and low-noise experts. The dense TI2V-5B
# checkpoint is not an MoE substitute and must not be combined with A14B
# high-noise parameters.
#
# Usage:
#   ./scripts/download-model-wan2.2.sh
#   ./scripts/download-model-wan2.2.sh --t2v-a14b
#   ./scripts/download-model-wan2.2.sh --t2v-a14b --fp8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$SCRIPT_DIR/.." && pwd)/models"
HF="https://huggingface.co"
REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
REPO_REV="fb1388adc906ab39ffc26ee40e96b22886b56bc4"

VARIANT="ti2v-5b"
PRECISION="fp16"

usage() {
  cat <<'EOF'
Usage: download-model-wan2.2.sh [--ti2v-5b | --t2v-a14b] [--fp16 | --fp8]

  --ti2v-5b  Dense 5B T2V/I2V model; fp16 only (default).
  --t2v-a14b Two-expert A14B text-to-video MoE model.
  --fp16      Download fp16 weights (default).
  --fp8       Download A14B scaled-FP8 experts and UMT5 encoder.
  --help      Show this message.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --ti2v-5b) VARIANT="ti2v-5b" ;;
    --t2v-a14b) VARIANT="t2v-a14b" ;;
    --fp16) PRECISION="fp16" ;;
    --fp8) PRECISION="fp8" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ "$VARIANT" == "ti2v-5b" && "$PRECISION" != "fp16" ]]; then
  echo "--ti2v-5b is published only as fp16 in this repository; remove --fp8." >&2
  exit 1
fi

mkdir -p "$OUT"
source "$SCRIPT_DIR/dl-functions.sh"

download() {
  local relative_path="$1"
  local filename="$2"
  dl "$HF/$REPO/resolve/$REPO_REV/$relative_path" "$OUT/$filename"
}

if [[ "$VARIANT" == "ti2v-5b" ]]; then
  echo "Downloading Wan 2.2 TI2V-5B (dense, fp16) to $OUT"
  download "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors" \
    "wan2.2_ti2v_5B_fp16.safetensors"
  download "split_files/vae/wan2.2_vae.safetensors" "wan2.2_vae.safetensors"
  download "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
    "umt5_xxl_fp16.safetensors"
else
  echo "Downloading Wan 2.2 T2V-A14B MoE ($PRECISION) to $OUT"
  if [[ "$PRECISION" == "fp16" ]]; then
    EXPERT_SUFFIX="fp16"
    TEXT_ENCODER="umt5_xxl_fp16.safetensors"
  else
    EXPERT_SUFFIX="fp8_scaled"
    TEXT_ENCODER="umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  fi

  download "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_${EXPERT_SUFFIX}.safetensors" \
    "wan2.2_t2v_low_noise_14B_${EXPERT_SUFFIX}.safetensors"
  download "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_${EXPERT_SUFFIX}.safetensors" \
    "wan2.2_t2v_high_noise_14B_${EXPERT_SUFFIX}.safetensors"
  download "split_files/vae/wan_2.1_vae.safetensors" "wan_2.1_vae.safetensors"
  download "split_files/text_encoders/$TEXT_ENCODER" "$TEXT_ENCODER"
fi

echo "done → $OUT"
