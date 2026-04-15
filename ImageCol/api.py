"""
Image Restoration & Colorization - Flask API
Run: python api.py
Then open frontend/index.html in your browser.
"""

import os
import io
import base64
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import sys
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from realesrgan_arch import RRDBNet

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
IMAGE_SIZE     = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "models"   # looks for .pth files in this directory

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

# ─────────────────────────────────────────────────────
# MODEL DEFINITIONS (mirrors train_colab.py)
# ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_norm=True, activation="leaky"):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
        else:
            layers.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
        if use_norm:
            layers.append(nn.BatchNorm2d(out_c))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.e1 = ConvBlock(input_channels, 64,  down=True,  use_norm=False)
        self.e2 = ConvBlock(64,  128, down=True)
        self.e3 = ConvBlock(128, 256, down=True)
        self.e4 = ConvBlock(256, 512, down=True)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True))
        self.d1 = ConvBlock(512,       512, down=False, activation="relu")
        self.d2 = ConvBlock(512 + 512, 256, down=False, activation="relu")
        self.d3 = ConvBlock(256 + 256, 128, down=False, activation="relu")
        self.d4 = ConvBlock(128 + 128,  64, down=False, activation="relu")
        self.final = nn.Sequential(nn.ConvTranspose2d(64 + 64, output_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        b  = self.bottleneck(e4)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        return self.final(torch.cat([d4, e1], dim=1))

# ─────────────────────────────────────────────────────
# GLOBAL MODEL STATE
# ─────────────────────────────────────────────────────
restoration_model  = None
colorization_model = None
esrgan_model       = None
loaded_checkpoint  = None
loaded_epoch       = None

def find_checkpoints():
    """Returns sorted list of .pth checkpoint files."""
    return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")))

def load_checkpoint(checkpoint_path):
    global restoration_model, colorization_model, loaded_checkpoint, loaded_epoch
    print(f"Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location=DEVICE)

    restoration_model  = CustomUNet(1, 1).to(DEVICE)
    colorization_model = CustomUNet(1, 2).to(DEVICE)
    restoration_model.load_state_dict(cp["restore_model_state"])
    colorization_model.load_state_dict(cp["color_model_state"])
    restoration_model.eval()
    colorization_model.eval()

    loaded_checkpoint = os.path.basename(checkpoint_path)
    loaded_epoch      = cp.get("epoch", "?")
    print(f"Loaded epoch {loaded_epoch} from {loaded_checkpoint}")

    # Load ESRGAN
    global esrgan_model
    esrgan_path = os.path.join(CHECKPOINT_DIR, "RealESRGAN_x4plus.pth")
    if os.path.exists(esrgan_path) and esrgan_model is None:
        try:
            print(f"Loading Real-ESRGAN Upscaler from {esrgan_path}...")
            esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
            esrgan_cp = torch.load(esrgan_path, map_location=DEVICE)
            
            # Support different potential checkpoint formats
            if "params_ema" in esrgan_cp:
                state_dict = esrgan_cp["params_ema"]
            elif "params" in esrgan_cp:
                state_dict = esrgan_cp["params"]
            else:
                state_dict = esrgan_cp
                
            esrgan_model.load_state_dict(state_dict)
            esrgan_model.to(DEVICE)
            esrgan_model.eval()
            print("Real-ESRGAN Upscaler ready.")
            sys.stdout.flush()
        except Exception as e:
            print(f"WARNING: Real-ESRGAN failed to load: {e}")
            esrgan_model = None
            sys.stdout.flush()

# Auto-load the latest checkpoint on startup
_checkpoints = find_checkpoints()
if _checkpoints:
    load_checkpoint(_checkpoints[-1])
else:
    print("No checkpoint found — call POST /load_checkpoint first.")

# ─────────────────────────────────────────────────────
# HELPER: PIL Image → base64 string
# ─────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def apply_sharpening(pil_img: Image.Image, amount: float) -> Image.Image:
    """
    Apply Unsharp Mask sharpening.
    amount=0.0  → no change
    amount=1.0  → moderate sharpening (good default)
    amount=2.0  → strong, photography-grade sharpening
    Works in LAB space so only the Luminance (detail) channel is sharpened,
    leaving the color channels untouched to avoid color fringing.
    """
    if amount <= 0:
        return pil_img
    from PIL import ImageFilter
    img_np = np.array(pil_img, dtype=np.float32)
    lab    = rgb2lab(img_np / 255.0).astype(np.float32)

    # Sharpen only the L channel
    L_pil      = Image.fromarray(np.clip(lab[:, :, 0], 0, 100).astype(np.uint8))
    L_sharp    = L_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(amount * 120), threshold=2))
    lab[:, :, 0] = np.clip(np.array(L_sharp, dtype=np.float32), 0, 100)

    rgb_sharp  = (lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_sharp)

# ─────────────────────────────────────────────────────
# HELPER: Run full pipeline on a given PIL image
# ─────────────────────────────────────────────────────
def run_pipeline(pil_img, add_degradation=True, scratch_intensity=6, sharpen_amount=1.0, upscale=False):
    """
    Runs the full restoration → colorization pipeline.
    Returns dict with:
      - original_gray_b64
      - degraded_b64  (if add_degradation)
      - restored_b64
      - colorized_b64
    """
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img, dtype=np.uint8)

    lab = rgb2lab(img_np).astype(np.float32)
    L   = lab[:, :, 0]

    # Optionally degrade
    if add_degradation:
        L_input = L.copy()
        h, w = L_input.shape
        noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
        L_input = np.clip(L_input + noise, 0, 100)
        for _ in range(scratch_intensity):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            length = max(int(np.hypot(x2-x1, y2-y1)), 1)
            xs = np.linspace(x1, x2, length).astype(int).clip(0, w-1)
            ys = np.linspace(y1, y2, length).astype(int).clip(0, h-1)
            L_input[ys, xs] = 95.0 if random.random() > 0.5 else 5.0
    else:
        L_input = L

    # Tensor prep
    L_t = torch.tensor((L_input / 50.0) - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        L_clean_pred = restoration_model(L_t)
        ab_pred      = colorization_model(L_clean_pred)

    # Decode outputs
    L_out  = ((L_clean_pred.squeeze().cpu().numpy() + 1.0) * 50.0)
    ab_out = (ab_pred.squeeze().cpu().numpy().transpose(1, 2, 0)) * 128.0

    lab_out          = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab_out[:, :, 0] = np.clip(L_out, 0, 100)
    lab_out[:, :, 1:]= np.clip(ab_out, -128, 127)

    # Build individual images
    def L_to_pil(l_arr):
        lab_tmp = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        lab_tmp[:, :, 0] = np.clip(l_arr, 0, 100)
        rgb = (lab2rgb(lab_tmp) * 255).astype(np.uint8)
        return Image.fromarray(rgb)

    original_gray_pil = L_to_pil(L)
    degraded_pil      = L_to_pil(L_input) if add_degradation else original_gray_pil
    restored_pil      = L_to_pil(L_out)
    colorized_raw     = Image.fromarray((lab2rgb(lab_out) * 255).astype(np.uint8))
    colorized_pil     = colorized_raw

    # Upscale outputs if requested and loaded
    if upscale and esrgan_model is not None:
        print("Applying 4x Upscaling via Real-ESRGAN...")
        with torch.no_grad():
            def upscale_pil(img_p):
                print(f"  > Processing {img_p.size} image...")
                # preprocess for ESRGAN: model expects BGR [0,1], BCHW
                img_arr = np.array(img_p)
                img_bgr = img_arr[:, :, ::-1].copy() # Convert RGB to BGR
                tensor_img = torch.from_numpy(img_bgr.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                out = esrgan_model(tensor_img)
                out = out.data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
                out_bgr = (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                out_rgb = out_bgr[:, :, ::-1] # Convert BGR to RGB
                out_img = Image.fromarray(out_rgb)
                print(f"  > Upscale complete: {out_img.size}")
                sys.stdout.flush()
                return out_img
            
            restored_pil  = upscale_pil(restored_pil)
            colorized_pil = upscale_pil(colorized_pil)

    # Always sharpen *after* upscaling to prevent ESRGAN from amplifying sharpening halos into blocky artifacts
    colorized_pil = apply_sharpening(colorized_pil, sharpen_amount)

    return {
        "original_gray": pil_to_b64(original_gray_pil),
        "degraded":      pil_to_b64(degraded_pil),
        "restored":      pil_to_b64(restored_pil),
        "colorized":     pil_to_b64(colorized_pil),
    }

# ─────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns current checkpoint info and available checkpoints."""
    checkpoints = [os.path.basename(c) for c in find_checkpoints()]
    return jsonify({
        "loaded":           loaded_checkpoint,
        "epoch":            loaded_epoch,
        "device":           str(DEVICE),
        "available":        checkpoints,
        "model_ready":      restoration_model is not None,
    })

@app.route("/api/load_checkpoint", methods=["POST"])
def api_load_checkpoint():
    """Load a specific checkpoint by filename."""
    data = request.get_json()
    filename = data.get("filename")
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {filename}"}), 404
    try:
        load_checkpoint(path)
        return jsonify({"ok": True, "epoch": loaded_epoch, "file": loaded_checkpoint})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/process", methods=["POST"])
def api_process():
    """
    Accepts a multipart form image upload and runs full pipeline.
    Form fields:
      - image: the uploaded image file
      - mode: 'restore' | 'colorize' | 'both' (default: 'both')
      - add_degradation: 'true' | 'false' (default: 'true')
      - scratch_intensity: integer 1-15 (default: 6)
    """
    if restoration_model is None:
        return jsonify({"error": "No model loaded. Upload a checkpoint first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file      = request.files["image"]
    add_deg   = request.form.get("add_degradation", "true").lower() == "true"
    intensity = int(request.form.get("scratch_intensity", 6))
    sharpen   = float(request.form.get("sharpen_amount", 1.0))
    upscale   = request.form.get("upscale", "false").lower() == "true"

    try:
        pil_img = Image.open(file.stream)
        results = run_pipeline(pil_img, add_degradation=add_deg,
                               scratch_intensity=intensity, sharpen_amount=sharpen, upscale=upscale)
        return jsonify({
            "ok":          True,
            "checkpoint":  loaded_checkpoint,
            "epoch":       loaded_epoch,
            **results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"\nImage Restoration & Colorization API")
    print(f"Device  : {DEVICE}")
    print(f"Checkpoint: {loaded_checkpoint} (Epoch {loaded_epoch})")
    print(f"Running at http://127.0.0.1:6060\n")
    app.run(host="0.0.0.0", port=6060, debug=False)
