# One-off evidence for qvac#4337, not a proposed audit implementation.
import csv, hashlib, json, os, pathlib, re, struct, subprocess, sys
root = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
baseline = set(re.findall(r"^\s*(\S+)\s+\(", (out / "baseline-ldconfig.txt").read_text(), re.M))
rows = []
for base, dirs, files in os.walk(root):
    for name in files:
        p = pathlib.Path(base) / name
        rel = p.relative_to(root).as_posix()
        if any(x in rel for x in ["android-", "linux-arm", "linux-ia32", "linux-riscv", "linux-mips", "linux-x64-musl"]):
            continue
        try:
            with p.open("rb") as f:
                header = f.read(64)
            if header[:4] != b"\x7fELF" or len(header) < 20 or header[4] != 2 or header[5] != 1 or struct.unpack_from("<H", header, 18)[0] != 62:
                continue
            dynamic = subprocess.run(["readelf", "-d", str(p)], capture_output=True, text=True, check=True).stdout
            needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
            if not needed:
                continue
            sonames = re.findall(r"\(SONAME\).*?\[(.*?)\]", dynamic)
            runpath = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", dynamic)
            package_root = p.parent
            while not (package_root / "package.json").is_file() and package_root != root:
                package_root = package_root.parent
            package = json.loads((package_root / "package.json").read_text())
            rows.append(dict(package=package.get("name"), version=package.get("version"), binary=rel,
                sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(), needed=needed,
                soname=sonames[0] if sonames else None, runpath=runpath,
                absent_from_image=[n for n in needed if n not in baseline]))
        except Exception as e:
            raise RuntimeError(f"Could not inspect {p}: {e}") from e
bundled = {}
for row in rows:
    if row["soname"]:
        bundled.setdefault(row["soname"], []).append(row["binary"])
for row in rows:
    row["bundled_candidates"] = {n: bundled[n] for n in row["absent_from_image"] if n in bundled}
    row["absent_from_image_and_tree"] = [n for n in row["absent_from_image"] if n not in bundled]
rows.sort(key=lambda x: (x["package"], x["binary"]))
(out / "native-needed.json").write_text(json.dumps(rows, indent=2) + "\n")
with (out / "native-needed.csv").open("w") as f:
    w=csv.writer(f); w.writerow(["package", "version", "binary", "DT_NEEDED", "absent from pristine image", "absent from image and scanned tree", "RPATH/RUNPATH"])
    for row in rows:
        w.writerow([row["package"],row["version"],row["binary"],"; ".join(row["needed"]),"; ".join(row["absent_from_image"]),"; ".join(row["absent_from_image_and_tree"]),"; ".join(row["runpath"])])
print(json.dumps(dict(binaries=len(rows), packages=len(set(row["package"] for row in rows)), missing=sorted(set(n for row in rows for n in row["absent_from_image_and_tree"]))), indent=2))
