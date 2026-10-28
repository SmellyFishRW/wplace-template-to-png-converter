from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

PALETTE_HEX = [
    "#000000", "#3c3c3c", "#787878", "#d2d2d2", "#ffffff", "#600018", "#ed1c24", "#ff7f27",
    "#f6aa09", "#f9dd3b", "#fffabc", "#0eb968", "#13e67b", "#87ff5e", "#0c816e", "#10aea6",
    "#13e1be", "#28509e", "#4093e4", "#60f7f2", "#6b50f6", "#99b1fb", "#780c99", "#aa38b9",
    "#e09ff9", "#cb007a", "#ec1f80", "#f38da9", "#684634", "#95682a", "#f8b277", "#aaaaaa",
    "#a50e1e", "#fa8072", "#e45c1a", "#d6b594", "#9c8431", "#c5ad31", "#e8d45f", "#4a6b3a",
    "#5a944a", "#84c573", "#0f799f", "#bbfaf2", "#7dc7ff", "#4d31b8", "#4a4284", "#7a71c4",
    "#b5aef1", "#dba463", "#d18051", "#ffc5a5", "#9b5249", "#d18078", "#fab6a4", "#7b6352",
    "#9c846b", "#333941", "#6d758d", "#b3b9d1", "#6d643f", "#948c6b", "#cdc59e"
]

PALETTE_RGB = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                        for h in PALETTE_HEX], dtype=np.float64)
SRGB_LINEAR = np.array([((v / 255 + .055) / 1.055) ** 2.4
                        if v / 255 > .04045 else v / 255 / 12.92
                        for v in range(256)])


def rgb_to_lab(rgb):
    r, g, b = SRGB_LINEAR[np.asarray(rgb, dtype=np.uint8)].T
    xyz = np.array([(r * .4124 + g * .3576 + b * .1805) / .95047,
                    r * .2126 + g * .7152 + b * .0722,
                    (r * .0193 + g * .1192 + b * .9505) / 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]),
                     200 * (f[1] - f[2])]).T


PALETTE_LAB = rgb_to_lab(PALETTE_RGB)


def calc_compuphase(pixel_rgb, palette_rgb):
    r, g, b = np.asarray(pixel_rgb, dtype=np.float64)
    pr, pg, pb = np.asarray(palette_rgb, dtype=np.float64).T
    mean = (r + pr) / 2
    return (2 + mean / 256) * (r - pr) ** 2 + 4 * (g - pg) ** 2 + (2 + (255 - mean) / 256) * (b - pb) ** 2


def delta94_squared(pixel, palette):
    dl, da, db = (pixel - palette).T
    c1 = math.sqrt(pixel[1] * pixel[1] + pixel[2] * pixel[2])
    c2 = np.sqrt(palette[:, 1] ** 2 + palette[:, 2] ** 2)
    dc = c1 - c2
    dh = np.sqrt(np.maximum(0, da * da + db * db - dc * dc))
    return dl * dl + (dc / (1 + .045 * c1)) ** 2 + (dh / (1 + .015 * c1)) ** 2


def delta2000_squared(pixel, palette):
    l1, a1, b1 = pixel
    l2, a2, b2 = palette.T
    mean_c = (np.sqrt(a1 * a1 + b1 * b1) + np.sqrt(a2 * a2 + b2 * b2)) * .5
    def pow7(v):
        squared = v * v
        return squared * squared * squared * v
    seventh = pow7(mean_c)
    g = .5 * (1 - np.sqrt(seventh / (seventh + 6103515625)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    c1, c2 = np.sqrt(ap1 * ap1 + b1 * b1), np.sqrt(ap2 * ap2 + b2 * b2)
    product = c1 * c2
    h1 = np.where(c1 == 0, 0, np.mod(np.arctan2(b1, ap1), 2 * np.pi))
    h2 = np.where(c2 == 0, 0, np.mod(np.arctan2(b2, ap2), 2 * np.pi))
    dh = h2 - h1
    dh = np.where(dh > np.pi, dh - 2 * np.pi, np.where(dh < -np.pi, dh + 2 * np.pi, dh))
    dh = np.where(product == 0, 0, dh)
    delta_h = np.where(product == 0, 0, 2 * np.sqrt(product) * np.sin(dh * .5))
    mean_l, mean_c = (l1 + l2) * .5, (c1 + c2) * .5
    hsum = h1 + h2
    mean_h = np.where(product == 0, hsum,
                      np.where(np.abs(h1 - h2) > np.pi,
                               np.where(hsum < 2 * np.pi, (hsum + 2 * np.pi) * .5,
                                        (hsum - 2 * np.pi) * .5), hsum * .5))
    t = (1 - .17 * np.cos(mean_h - .5235987755982988) + .24 * np.cos(2 * mean_h)
         + .32 * np.cos(3 * mean_h + .10471975511965977)
         - .2 * np.cos(4 * mean_h - 1.0995574287564276))
    angle = (mean_h - 4.799655442984406) / .4363323129985824
    theta = .5235987755982988 * np.exp(-(angle * angle))
    seventh = pow7(mean_c)
    rc = 2 * np.sqrt(seventh / (seventh + 6103515625))
    ll = (mean_l - 50) ** 2
    sl = 1 + .015 * ll / np.sqrt(20 + ll)
    sc, sh = 1 + .045 * mean_c, 1 + .015 * mean_c * t
    dl, dc, dh = (l2 - l1) / sl, (c2 - c1) / sc, delta_h / sh
    return np.maximum(0, dl * dl + dc * dc + dh * dh - np.sin(2 * theta) * rc * dc * dh)


def normalize_metric(metric):
    return metric if metric in ("compuphase", "ciede2000") else "lab"


def color_indices(values):
    if not isinstance(values, (list, tuple)):
        return None
    return sorted({v for v in values if type(v) is int and v > 0}) or None


def quantize_rgba(source, metric="lab", dithering=False, allowed_color_idxs=None):
    pixels = np.array(source, dtype=np.uint8, copy=True)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("Expected an H x W x 4 RGBA array")
    metric = normalize_metric(metric)
    indices = color_indices(allowed_color_idxs)
    indices = list(range(1, 64)) if not indices else [v for v in indices if v <= 63]
    indices = indices or [1]
    palette = PALETTE_RGB[np.array(indices) - 1]
    labs = PALETTE_LAB[np.array(indices) - 1]
    cache = {}

    def nearest(rgb):
        key = tuple(int(v) for v in rgb)
        result = cache.get(key)
        if result is None:
            if metric == "compuphase":
                distances = calc_compuphase(rgb, palette)
            else:
                lab = rgb_to_lab(rgb)
                distances = (delta2000_squared(lab, labs) if metric == "ciede2000"
                             else delta94_squared(lab, labs))
            result = palette[np.argmin(distances)].astype(np.uint8)
            cache[key] = result
        return result

    if not dithering:
        visible = pixels[:, :, 3] >= 16
        unique, inverse = np.unique(pixels[:, :, :3][visible], axis=0, return_inverse=True)
        if len(unique):
            mapped = np.array([nearest(rgb) for rgb in unique], dtype=np.uint8)
            pixels[:, :, :3][visible] = mapped[inverse]
        pixels[:, :, 3] = np.where(visible, 255, 0)
        return pixels

    height, width = pixels.shape[:2]
    current = np.zeros((width, 3), dtype=np.float32)
    following = np.zeros_like(current)
    for y in range(height):
        current, following = following, current
        following.fill(0)
        for x in range(width):
            if pixels[y, x, 3] < 16:
                pixels[y, x, 3] = 0
                continue
            rgb = np.clip(np.floor(pixels[y, x, :3].astype(np.float64)
                                   + current[x].astype(np.float64) + .5), 0, 255)
            chosen = nearest(rgb)
            pixels[y, x, :3] = chosen
            pixels[y, x, 3] = 255
            error = rgb - chosen
            if x + 1 < width:
                current[x + 1] = current[x + 1].astype(np.float64) + 7 / 16 * error
            if y + 1 < height:
                if x > 0:
                    following[x - 1] = following[x - 1].astype(np.float64) + 3 / 16 * error
                following[x] = following[x].astype(np.float64) + 5 / 16 * error
                if x + 1 < width:
                    following[x + 1] = following[x + 1].astype(np.float64) + 1 / 16 * error
    return pixels


def js_round(value):
    return math.floor(value + .5)


def get_template_dimensions(bounds):
    half_world = 2 * math.pi * 6378137 / 2
    resolution = (2 * half_world / 1000) / 2 ** 11
    def project(lat, lon):
        x = lon / 180 * half_world
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * half_world / 180
        return js_round((x + half_world) / resolution), js_round((half_world - y) / resolution)
    def unproject(x, y):
        mx, my = x * resolution - half_world, half_world - y * resolution
        lon = mx / half_world * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp((my / half_world * 180) * math.pi / 180)) - math.pi / 2)
        return lat, lon
    values = [bounds[k] for k in ("north", "south", "west", "east")]
    if not all(type(v) in (float, int) and math.isfinite(v) for v in values):
        raise ValueError("Template bounds must contain finite numbers")
    north, south, west, east = values
    if not (-90 < north < 90 and -90 < south < 90):
        raise ValueError("Template latitude must be between -90 and 90")
    north, west = unproject(*project(north, west))
    south, east = unproject(*project(south, east))
    north, south = min(85, max(north, south)), max(-85, min(north, south))
    north, south = max(-85, north), min(85, south)
    west, east = max(-180, min(west, east)), min(180, max(west, east))
    west, east = min(180, west), max(-180, east)
    x0, y0 = project(north, west)
    x1, y1 = project(south, east)
    return max(1, abs(x1 - x0)), max(1, abs(y1 - y0))


def resize_rgba(source, width, height):
    source_h, source_w = source.shape[:2]
    xs = np.minimum(np.arange(width, dtype=np.int64) * source_w // width, source_w - 1)
    ys = np.minimum(np.arange(height, dtype=np.int64) * source_h // height, source_h - 1)
    return source[ys[:, None], xs].copy()


def resolve_palette(data, unlocked_colors=None):
    mode = data.get("colorPaletteMode", "all")
    if mode == "free":
        return list(range(1, 32))
    if mode == "template":
        return color_indices(data.get("templateColorIdxs"))
    if mode == "unlocked":
        if unlocked_colors is None:
            raise ValueError("Unlocked palette needs account color IDs: use --unlocked-colors 32,33,... (free colors are included automatically). The .wplace export does not contain account unlocks.")
        return sorted(set(range(1, 32)) | set(unlocked_colors))
    return None


def find_browser(explicit=None):
    candidates = [explicit, os.environ.get("WPLACE_BROWSER")]
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if root:
            candidates.extend([str(Path(root) / "Microsoft/Edge/Application/msedge.exe"),
                               str(Path(root) / "Google/Chrome/Application/chrome.exe")])
    candidates.extend(shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "microsoft-edge"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise ValueError("Legacy colors and non-PNG images need Chrome/Edge/Chromium for exact Canvas behavior. Supply --browser PATH or WPLACE_BROWSER.")


def browser_pixels(payload, browser=None):
    executable = find_browser(browser)
    script = r"""
    (async () => {
      try {
        const p = PAYLOAD;
        const make = (w,h) => { const c = document.createElement('canvas'); c.width=w; c.height=h; return c; };
        let source, image;
        if (p.raw) {
          const bytes = Uint8ClampedArray.from(atob(p.raw), c=>c.charCodeAt(0));
          source = make(p.width,p.height);
          source.getContext('2d',{colorSpace:'srgb'}).putImageData(new ImageData(bytes,p.width,p.height),0,0);
        } else {
          const bytes = Uint8Array.from(atob(p.encoded), c=>c.charCodeAt(0));
          image = await createImageBitmap(new Blob([bytes]),{colorSpaceConversion:'default'});
          if (image.width>16384 || image.height>16384 || image.width*image.height>40000000) throw Error('Source image exceeds Wplace limits');
          const scale = Math.min(1,4096/image.width,4096/image.height);
          source = make(Math.max(1,Math.round(image.width*scale)),Math.max(1,Math.round(image.height*scale)));
          const ctx=source.getContext('2d',{colorSpace:'srgb'}); ctx.imageSmoothingEnabled=false;
          ctx.drawImage(image,0,0,source.width,source.height); image.close();
          // Site stores getImageData as raw pixels and recreates its source canvas.
          const raw=ctx.getImageData(0,0,source.width,source.height);
          ctx.putImageData(raw,0,0);
          if (!p.legacy) return finish(raw);
        }
        const target=make(p.targetWidth||source.width,p.targetHeight||source.height);
        const ctx=target.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});
        ctx.imageSmoothingEnabled=false; ctx.drawImage(source,0,0,target.width,target.height);
        finish(ctx.getImageData(0,0,target.width,target.height));
        function finish(result) {
          let binary=''; for(let i=0;i<result.data.length;i+=32768) binary+=String.fromCharCode(...result.data.subarray(i,i+32768));
          document.body.textContent='WPLACE_RESULT:'+JSON.stringify({width:result.width,height:result.height,raw:btoa(binary)});
        }
      } catch(e) {document.body.textContent='WPLACE_ERROR:'+String(e);}
    })();
    """.replace("PAYLOAD", json.dumps(payload))
    with tempfile.TemporaryDirectory(prefix="wplace-canvas-") as temp:
        page = Path(temp) / "canvas.html"
        page.write_text("<!doctype html><meta charset='utf-8'><body><script>" + script + "</script>", encoding="utf-8")
        command = [executable, "--headless=new", "--disable-gpu", "--no-first-run",
                   "--disable-extensions", "--disable-background-networking",
                   "--no-proxy-server", "--host-resolver-rules=MAP * ~NOTFOUND",
                   "--user-data-dir=" + str(Path(temp) / "profile"),
                   "--dump-dom", "--virtual-time-budget=10000", page.as_uri()]
        result = subprocess.run(command, capture_output=True, timeout=120,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = result.stdout.decode("utf-8", errors="replace")
        match = re.search(r'<body>WPLACE_RESULT:(\{.*?\})</body>', output, re.DOTALL)
        if not match:
            error = re.search(r'<body>WPLACE_ERROR:([^<]*)', output)
            raise ValueError("Browser image processing failed: " + (error.group(1) if error else "no pixel result returned"))
        decoded = json.loads(match.group(1))
        return np.frombuffer(base64.b64decode(decoded["raw"]), dtype=np.uint8).reshape(decoded["height"], decoded["width"], 4).copy()


def decode_template_image(data, target=None, browser=None):
    value = data["image"]["dataUrl"]
    if not isinstance(value, str):
        raise ValueError("image.dataUrl must be a base64 string")
    value = value.strip()
    if value.startswith("data:"):
        header, value = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("Expected a base64 image data URL")
    encoded = base64.b64decode(value, validate=True)
    legacy = data.get("useLegacyColors") is True
    if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return browser_pixels({"encoded": base64.b64encode(encoded).decode("ascii"),
                               "legacy": legacy, "targetWidth": target[0] if target else None,
                               "targetHeight": target[1] if target else None}, browser)
    with Image.open(io.BytesIO(encoded)) as image:
        if max(image.size) > 16384 or image.width * image.height > 40_000_000:
            raise ValueError("Source image exceeds Wplace limits")
        if image.mode in ("I;16", "I;16L", "I;16B"):
            gray16 = np.asarray(image, dtype=np.uint16)
            gray8 = (gray16 >> 8).astype(np.uint8)
            alpha = np.full(gray8.shape, 255, dtype=np.uint8)
            source = np.stack((gray8, gray8, gray8, alpha), axis=-1)
        else:
            source = np.array(image.convert("RGBA"))
    h, w = source.shape[:2]
    if max(w, h) > 4096:
        scale = 4096 / max(w, h)
        source = resize_rgba(source, max(1, js_round(w * scale)), max(1, js_round(h * scale)))
    if legacy:
        h, w = source.shape[:2]
        return browser_pixels({"raw": base64.b64encode(source.tobytes()).decode("ascii"),
                               "width": w, "height": h,
                               "targetWidth": target[0] if target else w,
                               "targetHeight": target[1] if target else h}, browser)
    return source


def convert_template(data, *, unlocked_colors=None, browser=None):
    metric = normalize_metric(data.get("colorMetric"))
    allowed = resolve_palette(data, unlocked_colors)
    target = get_template_dimensions(data["bounds"]) if "bounds" in data else None
    if target and max(target) > 4096:
        raise ValueError("Template dimensions exceed Wplace's 4096 x 4096 limit")
    source = decode_template_image(data, target, browser)
    if target:
        source = resize_rgba(source, *target)
    pixels = quantize_rgba(source, metric, data.get("dithering") is True, allowed)
    return Image.fromarray(pixels)


def process_wplace(file_path, *, output=None, unlocked_colors=None, browser=None):
    path = Path(file_path)
    print(f"Reading: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    metric = normalize_metric(data.get("colorMetric"))
    label = {"compuphase": "weighted", "lab": "perceptual", "ciede2000": "ciede2000"}[metric]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(data.get("name") or "output"))
    name = Path(name).stem.rstrip(" .") or "output"
    destination = Path(output) if output else path.with_name(f"{name}_{label}.png")
    if destination.resolve() == path.resolve():
        raise ValueError("Output must not overwrite the input template")
    print(f"Converting: {metric}, dithering={data.get('dithering') is True}, palette={data.get('colorPaletteMode', 'all')}")
    image = convert_template(data, unlocked_colors=unlocked_colors, browser=browser)
    image.save(destination, format="PNG")
    print(f"Saved {image.width} x {image.height}: {destination}")
    return destination


def parse_color_ids(value):
    try:
        values = [] if not value.strip() else [int(v.strip()) for v in value.split(",")]
        if any(v < 1 or v > 63 for v in values):
            raise ValueError()
        return values
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated color IDs from 1 to 63") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help=".wplace file(s), including drag and drop")
    parser.add_argument("-o", "--output", help="Output PNG path (one input only)")
    parser.add_argument("--unlocked-colors", type=parse_color_ids, help="Purchased color IDs for unlocked mode; free colors are always included")
    parser.add_argument("--browser", help="Chrome/Edge/Chromium executable for legacy colors or non-PNG sources")
    parser.add_argument("--no-pause", action="store_true", help="Never wait for Enter (automation)")
    args = parser.parse_args(argv)
    if args.output and len(args.files) != 1:
        parser.error("--output requires exactly one input")
    status = 0
    if not args.files:
        print("Drag and drop a .wplace file onto this script, or run python wplace_converter.py template.wplace")
    for file_path in args.files:
        try:
            process_wplace(file_path, output=args.output, unlocked_colors=args.unlocked_colors, browser=args.browser)
        except Exception as exc:
            print(f"Error converting {file_path}: {exc}", file=sys.stderr)
            status = 1
    if not args.no_pause and sys.stdin.isatty():
        input("\nPress Enter to exit...")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
