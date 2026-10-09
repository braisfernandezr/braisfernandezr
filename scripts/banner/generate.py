#!/usr/bin/env python3
"""Generate the animated GitHub profile banners (Portrait only).

Run from the repository root:
    python scripts/banner/generate.py

Requires: pip install -r scripts/banner/requirements.txt
Requires: a portrait photo at assets/source/portrait.png
"""

from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/source/portrait.png"
ASSETS = ROOT / "assets"

W, H = 1180, 610
SEED = 314159

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Brais Fernandez"),
    (1, "role", "Computer Engineering Student"),
    (1, "origin", "Galicia, Spain"),
    (1, "focus", "Infrastructure · Backend · Cloud"),
    (1, "status", "Automation · Development · Observability"),
    (1, "toolchain", "Java · Ansible · Grafana LGTM"),

    (0, "stack", ""),
    (1, "systems", "Linux · Debian · Rocky Linux"),
    (1, "virtualization", "Vagrant"),
    (1, "automation", "Ansible · Python"),
    (1, "backend", "Java"),
    (1, "storage", "MinIO · Distributed Storage"),
    (1, "observability", "Grafana · Loki · Tempo · Mimir · Alloy"),

    (0, "projects", ""),
    (1, "featured", "Distributed Storage & Cloud-Native Observability"),
    (1, "infrastructure", "Linux · Ansible · MinIO"),
    (1, "monitoring", "Metrics · Logs · Distributed Tracing"),

    (0, "contact", ""),
    (1, "github", "braisfernandezr"),
]

THEMES = {
    "dark": {
        "bg":      "#0A0F1E",
        "panel":   "#0D1628",
        "panel2":  "#101B30",
        "line":    "#25344C",
        "muted":   "#8291A8",
        "text":    "#F0E6F0",
        "portrait":"#F78CA0",
        "chrome":  "#C9B1D9",
        "accent":  "#F78CA0",
        "shadow":  "#02050B",
    },
    "light": {
        "bg":      "#FDF0F3",
        "panel":   "#FFFFFF",
        "panel2":  "#FDE8EE",
        "line":    "#F0C0CE",
        "muted":   "#9B7B8A",
        "text":    "#2D1A24",
        "portrait":"#E05F80",
        "chrome":  "#7B5EA7",
        "accent":  "#E05F80",
        "shadow":  "#D4A0B0",
    },
}

def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit Floyd-Steinberg diffusion; True means a lit pixel."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    height, width = work.shape
    for y in range(height):
        left_to_right = y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += err * 7 / 16
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < width:
                    work[y + 1, nx] += err * 1 / 16
    return out

def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Return sampled x/y banner coordinates from a 300x340 dither grid."""
    source = Image.open(SOURCE).convert("RGBA")
    w, h = source.size
    crop_w = int(w * 0.60)
    crop_h = int(crop_w * (340 / 300))
    left = (w - crop_w) // 2
    top = int(h * 0.08)
    crop = source.crop((left, top, left + crop_w, top + crop_h)).resize((300, 340), Image.Resampling.LANCZOS)
    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    if theme == "dark":
        bg = Image.new("RGBA", crop.size, "black")
        bg.alpha_composite(crop)
        prepared = ImageOps.grayscale(bg.convert("RGB"))
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
        prepared = ImageEnhance.Brightness(prepared).enhance(1.05)
        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=1))
        select_lit = True
    else:
        bg = Image.new("RGBA", crop.size, "white")
        bg.alpha_composite(crop)
        prepared = ImageOps.grayscale(bg.convert("RGB"))
        prepared = ImageOps.autocontrast(prepared, cutoff=1)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.25)
        prepared = ImageEnhance.Brightness(prepared).enhance(1.05)
        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150, threshold=1))
        select_lit = False

    bits = floyd_steinberg(np.asarray(prepared))
    active = bits if select_lit else ~bits
    if theme == "dark":
        active &= alpha > 0.08

    ys, xs = np.where(active)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack((74 + xs, 154 + ys)).astype(np.float32)
    if len(points) > 18000:
        points = points[rng.choice(len(points), 18000, replace=False)]
    return points

def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")

def point_path(points: np.ndarray) -> str:
    """Aggregate adjacent horizontal one-pixel dots into compact SVG path runs."""
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)

def render_svg(theme_name: str, portrait: np.ndarray, rng: np.random.Generator) -> str:
    t = THEMES[theme_name]
    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-labelledby="title desc">',
        "<title id=\"title\">Brais Fernandez's live system profile</title>",
        '<desc id="desc">Animated terminal profile with a dithered portrait.</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" '
        'flood-opacity=".28"/></filter>',
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">vim profile.yml</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        '<g opacity="1">',
    ]

    # Gentle breathing/drifting animation for the portrait (No logos)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 2, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        delta = noise[band]
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["portrait"]}" stroke-width="1" opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'dur="8s" repeatCount="indefinite" calcMode="ease-in-out" '
            f'values="0 0; {num(delta[0])} {num(delta[1])}; 0 0"/></path>'
        )

    parts.append("</g>")

    # One-shot scattered intro
    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["portrait"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            "</path>"
        )

    parts.extend(
        [
            "</g>",
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · FS/SERPENTINE</text>',
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing=".5">profile.yml</text>',
            f'<text x="580" y="111" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[YAML]</text>',
            f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1062" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700">@braisfernandezr</text>',
        ]
    )

    row_y = 148.0
    for idx, (indent, key, value) in enumerate(YAML_ROWS, 1):
        line_num = f"{idx:2d}"
        if indent == 0:
            content = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            text_x = 525.0
        else:
            content = (
                f'<tspan fill="{t["portrait"]}">{html.escape(key)}: </tspan>'
                f'<tspan fill="{t["text"]}">{html.escape(value)}</tspan>'
            )
            text_x = 542.0

        parts.extend(
            [
                f'<text x="506" y="{num(row_y)}" text-anchor="end" fill="{t["muted"]}" opacity=".45" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
                f"{line_num}</text>",
                f'<text x="{num(text_x)}" y="{num(row_y)}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
                f"{content}</text>",
            ]
        )
        row_y += 21.5

    parts.extend(
        [
            f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
            f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}" rx="0 0 5 5"/>',
            f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["portrait"]}"/>',
            f'<text x="521" y="547" text-anchor="middle" fill="{t["bg"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11" font-weight="700">NORMAL</text>',
            f'<text x="569" y="547" fill="{t["text"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="600">profile.yml</text>',
            f'<text x="740" y="547" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[utf-8]</text>',
            f'<text x="1134" y="547" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">17L, 482B  100%  17:1</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source portrait: {SOURCE}")
    ASSETS.mkdir(parents=True, exist_ok=True)

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        points = portrait_points(theme, rng)
        svg = render_svg(theme, points, rng)
        output = ASSETS / f"banner-{theme}.v9.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"{output.relative_to(ROOT)}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(points):,} portrait dots"
        )

if __name__ == "__main__":
    main()