# WPlace Template to PNG Converter

A lightning-fast Python script that accurately converts exported `.wplace` files into ready-to-use PNG images, mapping raw pixel data with 1:1 parity to the official Wplace canvas.

## Features
- **Drag-and-Drop & Batch Processing:** No terminal commands needed. Drag a single `.wplace` file—or select ten at once—and drop them right onto the script.
- **100% Website Parity:** Replicates the exact math, map coordinate snapping, and dithering used by the official Wplace worker.
- **Palette Modes:** Respects all website palette restrictions, including **All Colors**, **Free Colors only**, and **Template-locked** palettes.
- **Preserves Transparency:** Alpha channels and transparent pixels are kept intact.
- **Auto-Naming & Sanitization:** Reads template names safely and labels the metric used (e.g., `image_ciede2000.png`).
- **No Heavy Dependencies:** Pure NumPy implementation—no need for bulky scientific libraries.

## Supported Color Metrics
The script flawlessly replicates the color algorithms used by the website:
* `Weighted RGB (CompuPhase)` - Dynamic channel weighting based on red-luma.
* `Perceptual (CIELAB ΔE94)` - Perceptual difference mapping.
* `CIEDE2000` - The industry standard, most accurate color distance formula.

## Prerequisites
You just need Python installed, along with two lightweight image libraries.

Open your terminal or command prompt and run:
```bash
pip install Pillow numpy
```
*(Note: A Chromium-based browser like Google Chrome or Microsoft Edge is already installed on most PCs and is used automatically in the background if your template uses legacy colors or non-PNG images).*

## How to Use

### The Easy Way (Drag-and-Drop)
1. Grab your exported `.wplace` file(s).
2. Drag and drop them directly onto `wplace_converter.py`.
3. The converted `.png` image will instantly appear right next to the original file!

### Command Line Usage (Optional)

You can also run the script from your terminal:

```bash
python wplace_converter.py template.wplace [options]
```

#### All Available Options:

| Option | What it does | Example |
| :--- | :--- | :--- |
| `files` | One or more `.wplace` files to convert. | `python wplace_converter.py art.wplace` |
| `-o`, `--output` | Choose a custom output name/location (one file only). | `-o my_art.png` |
| `--unlocked-colors` | Comma-separated IDs (1–63) for colors you purchased. | `--unlocked-colors 32,33,35` |
| `--browser` | Custom path to Chrome/Edge (if the script can't find it). | `--browser "C:\Chrome\chrome.exe"` |
| `--no-pause` | Skips the *"Press Enter to exit..."* prompt. | `--no-pause` |
| `-h`, `--help` | Shows the built-in help message. | `python wplace_converter.py -h` |

## License
MIT License - Free to use, modify, and distribute.