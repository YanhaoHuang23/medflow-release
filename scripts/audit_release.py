#!/usr/bin/env python
"""Fail closed if restricted artifacts or local identity leak into a release tree."""

import argparse
from pathlib import Path


FORBIDDEN_SUFFIXES = {".pkl", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".h5", ".hdf5", ".wandb"}
FORBIDDEN_PARTS = {"runs", "results", "output", "output_flow", "output_vqfinal", "logs", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".json", ".sh"}
IDENTITY_MARKERS = ("/home" + "/Yanhao", "/mnt" + "/DATA-2", "C:\\Users\\")


def main():
    parser = argparse.ArgumentParser(description="Audit a MedFlow repository before public upload.")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    violations = []
    scanned = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        scanned += 1
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append("restricted artifact: {}".format(relative))
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            violations.append("generated-output path: {}".format(relative))
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in IDENTITY_MARKERS:
                if marker in text:
                    violations.append("local path marker {} in {}".format(marker, relative))
    if violations:
        print("RELEASE AUDIT FAILED")
        for violation in sorted(set(violations)):
            print("- {}".format(violation))
        raise SystemExit(1)
    print("RELEASE AUDIT PASSED: {} files checked under {}".format(scanned, root))


if __name__ == "__main__":
    main()
