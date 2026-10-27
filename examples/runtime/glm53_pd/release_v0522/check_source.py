#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that Python will load the source tree targeted by the startup patch."""

import argparse
import importlib.util
import os
from pathlib import Path


def check_source(source_dir):
    if os.environ.get("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND") != "python":
        raise RuntimeError(
            "This full migration profile requires "
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python in the worker environment; "
            "the official Rust tree does not execute the migrated Python tree fixes"
        )
    expected = Path(source_dir).resolve() / "python/sglang/__init__.py"
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        actual = None if spec is None else spec.origin
        raise RuntimeError(
            f"Python loads sglang from {actual!r}, expected {str(expected)!r}"
        )
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    print(check_source(args.source_dir))


if __name__ == "__main__":
    main()
