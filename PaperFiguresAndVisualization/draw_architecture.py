"""Publish the canonical editable LaneGameFormer architecture SVG.

The diagram is authored directly as SVG primitives. This module replaces the
old Matplotlib approximation and copies the canonical source byte-for-byte,
after validating its canvas, editability, and element IDs.

Examples:
    python draw_architecture.py
    python draw_architecture.py --output Architecture.svg
    python draw_architecture.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
EXPECTED_CANVAS = {"width": "1149", "height": "541", "viewBox": "0 0 1149 541"}
MINIMUM_EDITABLE_ELEMENTS = {"rect": 38, "path": 29, "text": 65}


def canonical_svg_path() -> Path:
    """Return canonical hand-authored SVG source."""
    script = Path(__file__).resolve()
    candidates = [
        *(parent / "SVG" / "Architecture.svg" for parent in script.parents),
        *(parent / "ResearchPaper" / "PythonCodesAndSVG" / "SVG" / "Architecture.svg" for parent in script.parents),
        *(parent / "ResearchPaper" / "PythonCodesAndSVG" / "LaneGameFormer_Architecture.svg" for parent in script.parents),
        script.parent / "Architecture.svg",
        script.parent / "LaneGameFormer_Architecture.svg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate canonical Architecture.svg from script path")


def validate_svg(svg_path: Path) -> dict[str, int]:
    """Validate dimensions, vector-only content, IDs, and editable elements."""
    root = ET.parse(svg_path).getroot()
    namespace = f"{{{SVG_NAMESPACE}}}"
    if root.tag != f"{namespace}svg":
        raise ValueError(f"Expected SVG root, got {root.tag!r}")

    for attribute, expected in EXPECTED_CANVAS.items():
        actual = root.attrib.get(attribute)
        if actual != expected:
            raise ValueError(f"Invalid {attribute}: expected {expected!r}, got {actual!r}")

    counts = {
        name: len(root.findall(f".//{namespace}{name}"))
        for name in ("rect", "path", "text", "tspan", "image", "g")
    }
    if counts["image"]:
        raise ValueError("SVG must remain vector-only; embedded raster image found")
    for name, minimum in MINIMUM_EDITABLE_ELEMENTS.items():
        if counts[name] < minimum:
            raise ValueError(f"Expected at least {minimum} editable {name} elements")

    ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
    duplicate_ids = sorted({element_id for element_id in ids if ids.count(element_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate SVG IDs: {', '.join(duplicate_ids)}")
    return counts


def sha256(path: Path) -> str:
    """Return file SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_output_path() -> Path:
    """Use historical output name appropriate for current directory."""
    script_dir = Path(__file__).resolve().parent
    filename = (
        "LaneGameFormer_Architecture.svg"
        if script_dir.name == "PythonCodesAndSVG"
        else "Architecture.svg"
    )
    return script_dir / filename


def draw_diagram(output_path: str | Path | None = None) -> Path:
    """Publish exact canonical SVG and return resolved output path."""
    source = canonical_svg_path()
    validate_svg(source)
    output = Path(output_path) if output_path is not None else default_output_path()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output != source.resolve():
        shutil.copyfile(source, output)
    validate_svg(output)
    if sha256(output) != sha256(source):
        raise RuntimeError("Published SVG differs from canonical source")
    return output


def drawDiagram(outputPath: str | Path) -> Path:
    """Backward-compatible alias for previous callers."""
    return draw_diagram(outputPath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or validate the editable LaneGameFormer architecture SVG."
    )
    parser.add_argument("--output", type=Path, help="Destination SVG path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate canonical SVG without writing output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = canonical_svg_path()
    if args.check:
        counts = validate_svg(source)
        print(f"OK: {source}")
        print(f"SHA256: {sha256(source)}")
        print(f"Editable: {counts['rect']} rect, {counts['path']} path, {counts['text']} text")
        return 0
    output = draw_diagram(args.output)
    print(f"Wrote: {output}")
    print(f"SHA256: {sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
