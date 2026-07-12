#!/usr/bin/env python3
"""Fetch and shrink the demo VRM avatar (Seed-san).

Downloads the VRM-Consortium sample model Seed-san (© VirtualCast, Inc.,
VRM Public License 1.0 — redistribution and modification permitted, credit
required) and downscales its textures to <=512 px (JPEG for opaque, PNG for
alpha) so the demo page stays small. Writes ./avatar.vrm.

Usage: python3 fetch-avatar.py [out.vrm]
"""

import json
import os
import struct
import sys
import urllib.request
from io import BytesIO

from PIL import Image

URL = ("https://raw.githubusercontent.com/vrm-c/vrm-specification/"
       "master/samples/Seed-san/vrm/Seed-san.vrm")


def shrink(raw):
    magic, version, length = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "not a glb/vrm"
    off = 12
    chunks = []
    while off < length:
        clen, ctype = struct.unpack("<II", raw[off:off+8])
        chunks.append([ctype, raw[off+8:off+8+clen]])
        off += 8 + clen
    gltf = json.loads(chunks[0][1])
    binbuf = chunks[1][1]

    image_bv = {}
    for img in gltf.get("images", []):
        bvi = img["bufferView"]
        bv = gltf["bufferViews"][bvi]
        data = binbuf[bv.get("byteOffset", 0):
                      bv.get("byteOffset", 0) + bv["byteLength"]]
        im = Image.open(BytesIO(data))
        w, h = im.size
        scale = 512 / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                           Image.LANCZOS)
        out = BytesIO()
        if im.mode == "RGBA":
            im.save(out, "PNG", optimize=True)
            img["mimeType"] = "image/png"
        else:
            im.convert("RGB").save(out, "JPEG", quality=82)
            img["mimeType"] = "image/jpeg"
        image_bv[bvi] = out.getvalue()

    newbin = bytearray()
    for i, bv in enumerate(gltf["bufferViews"]):
        data = image_bv.get(i)
        if data is None:
            data = binbuf[bv.get("byteOffset", 0):
                          bv.get("byteOffset", 0) + bv["byteLength"]]
        while len(newbin) % 4:
            newbin.append(0)
        bv["byteOffset"] = len(newbin)
        bv["byteLength"] = len(data)
        newbin.extend(data)
    gltf["buffers"][0]["byteLength"] = len(newbin)

    jout = json.dumps(gltf, separators=(",", ":")).encode()
    while len(jout) % 4:
        jout += b" "
    while len(newbin) % 4:
        newbin.append(0)
    total = 12 + 8 + len(jout) + 8 + len(newbin)
    out = BytesIO()
    out.write(struct.pack("<III", 0x46546C67, 2, total))
    out.write(struct.pack("<II", len(jout), 0x4E4F534A))
    out.write(jout)
    out.write(struct.pack("<II", len(newbin), 0x004E4942))
    out.write(bytes(newbin))
    return out.getvalue()


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatar.vrm")
    print(f"downloading {URL} ...")
    raw = urllib.request.urlopen(URL).read()
    print(f"  {len(raw)/1e6:.1f} MB; shrinking textures ...")
    small = shrink(raw)
    with open(dst, "wb") as f:
        f.write(small)
    print(f"wrote {dst} ({len(small)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
