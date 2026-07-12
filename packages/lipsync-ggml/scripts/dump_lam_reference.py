#!/usr/bin/env python3
"""Dump PyTorch reference activations for LAM Audio2Expression.

Standalone reference implementation of the LAM Audio2Expression forward pass
(upstream: https://github.com/aigc3d/LAM_Audio2Expression, Apache-2.0). Uses
the HuggingFace transformers Wav2Vec2Model submodules exactly the way the
upstream custom forward does (feature_extractor -> 50->30fps linear
interpolation -> feature_projection -> encoder), and reimplements the small
LAM head (feature projection, identity-conditioned conv stack, ConvNormRelu
decoder, sigmoid output) functionally, mirroring models/network.py.

Runs deterministic audio and dumps the input, every stage's intermediate
activations, and the final 52-coefficient blendshape frames as raw
little-endian f32 .bin files plus a manifest.json with shapes. These are the
golden fixtures the ggml port is verified against (tolerance 1e-3).

Usage:
  python3 dump_lam_reference.py \
      --wav2vec2-config /path/to/LAM_Audio2Expression/configs/wav2vec2_config.json \
      --checkpoint /path/to/pretrained_models/lam_audio2exp_streaming.tar \
      --out-dir ../test/fixtures/reference
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


def make_test_audio(num_samples, seed=1234):
    """Deterministic speech-like waveform: harmonic voiced segments with an
    amplitude envelope, pauses, and low-level noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(num_samples, dtype=np.float64) / 16000.0
    # f0 glides around 90-220 Hz
    f0 = 150.0 + 60.0 * np.sin(2 * np.pi * 0.7 * t) + 10.0 * np.sin(2 * np.pi * 2.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / 16000.0
    voiced = (
        0.6 * np.sin(phase)
        + 0.3 * np.sin(2 * phase)
        + 0.15 * np.sin(3 * phase)
        + 0.08 * np.sin(5 * phase)
    )
    # syllable-rate envelope (~4 Hz) with pauses
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.7 * t - 0.5))
    env = env ** 1.5
    gate = (np.sin(2 * np.pi * 0.9 * t) > -0.6).astype(np.float64)
    noise = 0.01 * rng.randn(num_samples)
    audio = 0.5 * voiced * env * gate + noise
    return audio.astype(np.float32)


def load_weights(checkpoint_path, wav2vec2_config_path):
    """Returns (hf_wav2vec2_model, head_tensors_dict)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        state[key] = value

    config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
    encoder = Wav2Vec2Model(config)

    enc_state = {}
    for key, value in state.items():
        if key.startswith("audio_encoder."):
            key = key[len("audio_encoder."):]
            if key.startswith("lm_head."):
                continue  # LAM subclass artifact, unused at inference
            enc_state[key] = value

    # Old checkpoints store weight-norm as weight_g/weight_v; newer torch
    # parametrizes them as parametrizations.weight.original0/original1.
    model_keys = set(encoder.state_dict().keys())
    remapped = {}
    for key, value in enc_state.items():
        if key not in model_keys and key.endswith("weight_g"):
            key = key.replace("weight_g", "parametrizations.weight.original0")
        elif key not in model_keys and key.endswith("weight_v"):
            key = key.replace("weight_v", "parametrizations.weight.original1")
        remapped[key] = value

    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(f"missing wav2vec2 keys: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected wav2vec2 keys: {unexpected}")
    encoder.eval()

    head = {k: v for k, v in state.items() if not k.startswith("audio_encoder.")}
    return encoder, head


def conv_norm_relu(x, head, prefix, residual=None):
    """ConvNormRelu with norm='ln' from LAM models/network.py.

    conv(k3,s1,p1) -> LayerNorm over channels -> (+ residual) -> ReLU.
    residual: None (no residual), 'identity' (add input),
    'conv' (add residual_layer conv of input).
    """
    out = F.conv1d(x, head[f"{prefix}.conv.weight"], head[f"{prefix}.conv.bias"], padding=1)
    out = F.layer_norm(
        out.transpose(1, 2), (out.shape[1],),
        head[f"{prefix}.norm.weight"], head[f"{prefix}.norm.bias"],
    ).transpose(1, 2)
    if residual == "identity":
        out = out + x
    elif residual == "conv":
        out = out + F.conv1d(
            x,
            head[f"{prefix}.residual_layer.0.weight"],
            head[f"{prefix}.residual_layer.0.bias"],
            padding=1,
        )
    return F.relu(out)


def run_reference(encoder, head, audio, id_onehot, captures):
    """Full LAM Audio2Expression forward, mirroring models/network.py."""
    num_samples = audio.shape[-1]
    time_steps = math.ceil(num_samples / 16000 * 30)

    # --- custom Wav2Vec2Model.forward (models/encoder/wav2vec.py) ---
    hs = encoder.feature_extractor(audio)          # (B, 512, T_conv)
    captures["fe_out"] = hs
    hs = hs.transpose(1, 2)                        # (B, T_conv, 512)
    # linear_interpolation(features, 50, 30, output_len=frame_num)
    hs = F.interpolate(
        hs.transpose(1, 2), size=time_steps, align_corners=True, mode="linear"
    ).transpose(1, 2)                              # (B, time_steps, 512)
    captures["interp_out"] = hs
    hs = encoder.feature_projection(hs)[0]         # (B, time_steps, 768)
    captures["fp_out"] = hs

    pos_hook_out = {}
    hook = encoder.encoder.pos_conv_embed.register_forward_hook(
        lambda _m, _i, out: pos_hook_out.__setitem__("v", out)
    )
    enc_out = encoder.encoder(
        hs, attention_mask=None, output_hidden_states=True, return_dict=True
    )
    hook.remove()
    captures["pos_conv_out"] = pos_hook_out["v"]
    all_hs = enc_out.hidden_states                 # [pre-layer0, post-layer0, ..., post-layer11]
    captures["enc_pre_ln"] = all_hs[0]
    for i in range(1, len(all_hs)):
        captures[f"enc_layer_{i - 1}"] = all_hs[i]
    last_hidden = enc_out.last_hidden_state        # == all_hs[-1]

    # --- LAM head (models/network.py Audio2Expression.forward) ---
    af = F.linear(last_hidden, head["feature_projection.weight"],
                  head["feature_projection.bias"]).transpose(1, 2)  # (B, 512, T)
    captures["lam_proj"] = af

    # AudioIdentityEncoder.forward (dropout is a no-op at eval)
    identity = id_onehot.reshape(id_onehot.shape[0], -1, 1).repeat(
        1, 1, af.shape[2]).to(torch.float32)
    identity = F.conv1d(identity, head["identity_encoder.id_mlp.weight"],
                        head["identity_encoder.id_mlp.bias"])       # (B, 64, T)
    x = torch.cat([af, identity], dim=1)                            # (B, 576, T)
    x = conv_norm_relu(x, head, "identity_encoder.first_net.conv_layers.0", residual="conv")
    captures["ident_cnr_0"] = x
    x = conv_norm_relu(x, head, "identity_encoder.first_net.conv_layers.1", residual="identity")
    captures["ident_cnr_1"] = x
    x = conv_norm_relu(x, head, "identity_encoder.first_net.conv_layers.2", residual="identity")
    captures["ident_cnr_2"] = x
    captures["ident_out"] = x

    # decoder: 3x ConvNormRelu without residual
    for i in range(3):
        x = conv_norm_relu(x, head, f"decoder.0.{i}", residual=None)
        captures[f"dec_cnr_{i}"] = x

    logits = F.linear(x.permute(0, 2, 1), head["output_proj.weight"],
                      head["output_proj.bias"])                     # (B, T, 52)
    captures["expr_logits"] = logits
    return torch.sigmoid(logits)


def dump_case(encoder, head, out_dir, case_name, num_samples, id_idx=0):
    os.makedirs(out_dir, exist_ok=True)
    audio = make_test_audio(num_samples)
    time_steps = math.ceil(num_samples / 16000 * 30)

    id_onehot = F.one_hot(torch.tensor(id_idx), 12)[None, ...]
    captures = {}
    with torch.no_grad():
        expr = run_reference(
            encoder, head, torch.from_numpy(audio)[None, ...], id_onehot, captures
        )

    captures["input_pcm"] = torch.from_numpy(audio)
    captures["id_onehot"] = id_onehot.float()
    captures["expr"] = expr

    manifest = {"case": case_name, "num_samples": num_samples,
                "time_steps": time_steps, "id_idx": id_idx, "tensors": {}}
    for name, tensor in captures.items():
        arr = tensor.detach().float().cpu().numpy()
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        fname = f"{case_name}_{name}.bin"
        arr.tofile(os.path.join(out_dir, fname))
        manifest["tensors"][name] = {"file": fname, "shape": list(arr.shape)}
        print(f"  {case_name}/{name}: shape={list(arr.shape)} "
              f"min={arr.min():.5f} max={arr.max():.5f} mean={arr.mean():.5f}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec2-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    encoder, head = load_weights(args.checkpoint, args.wav2vec2_config)

    manifests = []
    # 64-frame streaming window: 16000*64//30 samples, the addon's native shape
    manifests.append(dump_case(encoder, head, args.out_dir, "win64", 16000 * 64 // 30))
    # 2 s clip -> 60 frames, exercises the non-64 frame path
    manifests.append(dump_case(encoder, head, args.out_dir, "sec2", 32000))

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"cases": manifests}, f, indent=2)
    print(f"wrote {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
