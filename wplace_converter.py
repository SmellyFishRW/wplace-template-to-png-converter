import sys
import os
import json
import base64
import io
from PIL import Image
import numpy as np
from skimage import color

# raw hex wplace palette
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

def calc_compuphase(pixel_rgb, palette_rgb):
    # cast to float to dodge uint8 overflow
    c1 = pixel_rgb.astype(float)
    palette = palette_rgb.astype(float)
    
    r_mean = (c1[0] + palette[:, 0]) / 2.0
    r_diff = c1[0] - palette[:, 0]
    g_diff = c1[1] - palette[:, 1]
    b_diff = c1[2] - palette[:, 2]
    
    # compuphase sliding scale
    weight_r = 2.0 + (r_mean / 256.0)
    weight_g = 4.0
    weight_b = 2.0 + ((255.0 - r_mean) / 256.0)
    
    return (weight_r * (r_diff ** 2)) + (weight_g * (g_diff ** 2)) + (weight_b * (b_diff ** 2))

def process_wplace(file_path):
    print(f"reading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    orig_name = data.get("name", "output.png")
    metric = data.get("colorMetric", "ciede2000").lower()
    
    # map json keys to readable filenames
    metrics_map = {
        "compuphase": "weighted",
        "lab": "perceptual",
        "ciede2000": "ciede2000"
    }
    display_metric = metrics_map.get(metric, metric)
    
    # setup paths
    folder = os.path.dirname(file_path)
    name, ext = os.path.splitext(orig_name)
    ext = ext if ext else ".png"
    
    out_name = f"{name}_{display_metric}{ext}"
    out_path = os.path.join(folder, out_name)
    print(f"using {display_metric} mode -> {out_name}")

    # decode base64 payload
    b64_str = data["image"]["dataUrl"].split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGBA")
    img_arr = np.array(img)
    
    rgb_arr = img_arr[:, :, :3]
    alpha_arr = img_arr[:, :, 3]
    
    # skip empty/transparent pixels to save cpu time
    mask = alpha_arr > 0
    valid_pixels = rgb_arr[mask]
    
    if len(valid_pixels) == 0:
        print("image is fully transparent. skipping.")
        img.save(out_path)
        return

    # find unique colors
    uniques, inverses = np.unique(valid_pixels, axis=0, return_inverse=True)
    print(f"processing {len(uniques)} unique colors...")
    
    # convert hex array to rgb array dynamically
    palette_rgb = np.array([
        [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] 
        for h in PALETTE_HEX
    ])
    
    if metric in ["ciede2000", "lab"]:
        # pre-calc lab conversions
        unique_lab = color.rgb2lab((uniques / 255.0).reshape(1, -1, 3)).reshape(-1, 3)
        palette_lab = color.rgb2lab((palette_rgb / 255.0).reshape(1, -1, 3)).reshape(-1, 3)
    
    mapped = np.zeros_like(uniques)
    
    # match each color
    for i in range(len(uniques)):
        if metric == "compuphase":
            dists = calc_compuphase(uniques[i], palette_rgb)
        elif metric == "lab":
            p_arr = np.tile(unique_lab[i], (len(palette_lab), 1))
            dists = color.deltaE_ciede94(p_arr, palette_lab)
        else:
            p_arr = np.tile(unique_lab[i], (len(palette_lab), 1))
            dists = color.deltaE_ciede2000(p_arr, palette_lab)
            
        mapped[i] = palette_rgb[np.argmin(dists)]
        
    # rebuild image array
    new_rgb = rgb_arr.copy()
    new_rgb[mask] = mapped[inverses]
    
    # stitch alpha back on
    new_img_arr = np.dstack((new_rgb, alpha_arr))
    Image.fromarray(new_img_arr.astype(np.uint8), "RGBA").save(out_path)
    print("done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("no file provided. drag and drop a .wplace file onto this script.")
    else:
        try:
            process_wplace(sys.argv[1])
        except Exception as e:
            print(f"error: {e}")
            
    input("\npress enter to exit...")