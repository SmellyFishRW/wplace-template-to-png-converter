# WPlace Template to PNG Converter

A lightning-fast, drag-and-drop Python script that accurately converts exported `.wplace` files into ready-to-use PNG images, while mapping the raw image data perfectly to the official color palette.

## Features
- **Drag-and-Drop:** No need to type terminal commands. Just drag your `.wplace` file onto the script.
- **Accurate Color Matching:** Reads the internal JSON config to map colors exactly how the website does.
- **Preserves Transparency:** Alpha channels are kept completely intact.
- **Auto-Naming:** Automatically reads the intended output name and appends the metric used (e.g., `image_ciede2000.png`).

## Supported Color Metrics
The script flawlessly replicates the following algorithms used by the website:
* `Weighted RGB (CompuPhase)` - Dynamic channel weighting based on red-luma.
* `Perceptual (CIELAB ΔE94)` - Standard CIE94 perceptual mapping.
* `CIEDE2000` - The most accurate color distance formula available.

## Prerequisites
You need Python installed, along with a few image-processing libraries. 
Open your terminal or command prompt and run:
```bash
pip install Pillow numpy scikit-image
```

## How to Use
1. Download `wplace_converter.py`.
2. Grab your exported `.wplace` file.
3. Drag and drop it directly onto the `wplace_converter.py` file!
4. The converted `.png` file will instantly appear in the same folder.

## License
MIT License - Free to use, modify, and distribute.
