#!/usr/bin/env python3
"""Bundle lipsync demo data (one or more clips + portrait) into data.js.

Multi-clip form:
  python3 make-demo-data.py --out data.js --portrait portrait.jpg \
      --clip "JFK 1961" jfk-frames.json jfk.wav \
      --clip "QVAC TTS" say-frames.json say.wav

Legacy single-clip form (kept for compatibility):
  python3 make-demo-data.py <frames.json> <audio.wav> <out-data.js> [portrait.jpg]

Output shape (consumed by index.html and vrm.html):
  window.LIPSYNC_DEMO_DATA = {
    fps, coeffNames, portraitB64?,
    clips: [{ name, frames, audioB64 }, ...]
  }
"""

import base64
import json
import sys


def b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def write_out(out_path, payload):
    with open(out_path, "w") as f:
        f.write("window.LIPSYNC_DEMO_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {out_path}")


def main():
    argv = sys.argv[1:]
    if "--out" not in argv:
        # legacy positional form
        frames_path, wav_path, out_path = argv[0:3]
        with open(frames_path) as f:
            data = json.load(f)
        payload = {
            "fps": data["fps"],
            "coeffNames": data["coeffNames"],
            "clips": [{"name": "clip", "frames": data["frames"],
                       "audioB64": b64_file(wav_path)}],
        }
        if len(argv) > 3:
            payload["portraitB64"] = b64_file(argv[3])
        write_out(out_path, payload)
        return

    out_path = argv[argv.index("--out") + 1]
    portrait = None
    if "--portrait" in argv:
        portrait = argv[argv.index("--portrait") + 1]

    clips = []
    fps = None
    names = None
    i = 0
    while i < len(argv):
        if argv[i] == "--clip":
            label, frames_path, wav_path = argv[i + 1: i + 4]
            with open(frames_path) as f:
                data = json.load(f)
            fps = data["fps"]
            names = data["coeffNames"]
            clips.append({"name": label, "frames": data["frames"],
                          "audioB64": b64_file(wav_path)})
            i += 4
        else:
            i += 1

    payload = {"fps": fps, "coeffNames": names, "clips": clips}
    if portrait:
        payload["portraitB64"] = b64_file(portrait)
    write_out(out_path, payload)


if __name__ == "__main__":
    main()
