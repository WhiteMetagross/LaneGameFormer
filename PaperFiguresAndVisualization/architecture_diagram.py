"""Compatibility entry point for the canonical architecture SVG publisher.

New code should import from ``draw_architecture`` directly. Existing imports
of ``architecture_diagram`` continue to work through re-exported functions.
"""

from draw_architecture import (
    canonical_svg_path,
    draw_diagram,
    drawDiagram,
    main,
    sha256,
    validate_svg,
)


__all__ = [
    "canonical_svg_path",
    "draw_diagram",
    "drawDiagram",
    "sha256",
    "validate_svg",
]


if __name__ == "__main__":
    raise SystemExit(main())
