"""Execute a public Kaggle notebook with GB10-local data and output paths.

The notebook source is read at runtime and kept out of the repository. Only
the generated artifacts and this reproducible path adapter are project-owned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def notebook_source(path: Path) -> str:
    document = json.loads(path.read_text())
    chunks: list[str] = []
    for cell in document["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else source)
    return "\n\n".join(chunks)


def replace_exact(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one occurrence of {old!r}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    source = notebook_source(args.notebook)
    source = replace_exact(
        source,
        'OUTPUT_DIR = Path("/kaggle/working")',
        f"OUTPUT_DIR = Path({str(args.output_dir)!r})",
    )
    source = replace_exact(
        source,
        'PREP_DIR = OUTPUT_DIR / "prepared_140926_s4"',
        'PREP_DIR = OUTPUT_DIR / "prepared"',
    )
    source = replace_exact(
        source,
        "data_dir = find_data_dir()",
        f"data_dir = Path({str(args.data_dir)!r})",
    )
    print(f"Executing public notebook source: {args.notebook}", flush=True)
    print(f"Competition data: {args.data_dir}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    exec(compile(source, str(args.notebook), "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
