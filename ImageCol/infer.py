import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

# Import the model architecture from the previously created train_colab script
# We define them here to keep it self-contained in case it's run separately.
import torch.nn as nn

IMAGE_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.d1 = ConvBlock(512,       512, down=False, activation="relu")
        self.d2 = ConvBlock(512 + 512, 256, down=False, activation="relu")
        self.d3 = ConvBlock(256 + 256, 128, down=False, activation="relu")
        self.d4 = ConvBlock(128 + 128,  64, down=False, activation="relu")

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.bottleneck(e4)

        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        out = self.final(torch.cat([d4, e1], dim=1))
        return out


def load_checkpoint(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location=DEVICE)
    
    restoration_model = CustomUNet(1, 1).to(DEVICE)
    colorization_model = CustomUNet(1, 2).to(DEVICE)
    
    restoration_model.load_state_dict(cp["restore_model_state"])
    colorization_model.load_state_dict(cp["color_model_state"])
    
    restoration_model.eval()
    colorization_model.eval()
    print(f"Models loaded successfully from Epoch {cp.get('epoch', '?')}!")
    return restoration_model, colorization_model


def test_image(image_path, restoration_model, colorization_model, out_path="result.png", degrade_input=True):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found.")
        return
        
    print(f"Processing image: {image_path}")
    # 1. Load and prepare the image
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img)
    
    # 2. Convert to LAB and extract Grayscale/Luminance
    lab = rgb2lab(img_np).astype(np.float32)
    L = lab[:, :, 0] # Real Luminance
    
    # Add fake degradation just to test the restoration works
    import random
    L_degraded = L.copy()
    if degrade_input:
        h, w = L.shape
        noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
        L_degraded += noise
        for _ in range(5):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
            length = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
            xs = np.linspace(x1, x2, length).astype(int).clip(0, w - 1)
            ys = np.linspace(y1, y2, length).astype(int).clip(0, h - 1)
            L_degraded[ys, xs] = 95.0

    L_t = torch.tensor((L_degraded / 50.0) - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print("Running models...")
    # 3. Model Inference
    with torch.no_grad():
        L_clean_pred = restoration_model(L_t)
        ab_pred = colorization_model(L_clean_pred)

    # 4. Post-processing
    L_out  = ((L_clean_pred.squeeze().cpu().numpy() + 1.0) * 50.0)
    ab_out = (ab_pred.squeeze().cpu().numpy().transpose(1, 2, 0)) * 128.0

    lab_out = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab_out[:, :, 0]  = np.clip(L_out, 0, 100)
    lab_out[:, :, 1:] = np.clip(ab_out, -128, 127)

    rgb_out = (lab2rgb(lab_out) * 255).astype(np.uint8)
    
    # Also save the degraded grayscale for comparison
    lab_deg = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab_deg[:, :, 0] = np.clip(L_degraded, 0, 100)
    rgb_deg = (lab2rgb(lab_deg) * 255).astype(np.uint8)

    # 5. Display/Save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_deg); axes[0].set_title("1. Scratched Grayscale Input"); axes[0].axis('off')
    
    lab_restored = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab_restored[:,:,0] = np.clip(L_out, 0, 100)
    axes[1].imshow((lab2rgb(lab_restored) * 255).astype(np.uint8)); axes[1].set_title("2. AI Restored Grayscale"); axes[1].axis('off')
    
    axes[2].imshow(rgb_out); axes[2].set_title("3. Final Colorized Output"); axes[2].axis('off')
    
    # Save the standalone degraded input image
    base_name, ext = os.path.splitext(out_path)
    deg_out_path = f"{base_name}_degraded{ext}"
    Image.fromarray(rgb_deg).save(deg_out_path)
    print(f"Saved standalone degraded input to: {deg_out_path}")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Success! View your result at: {out_path}")


if __name__ == "__main__":
    CHECKPOINT = "models/checkpoint_epoch_0150.pth"
    restoration_model, colorization_model = load_checkpoint(CHECKPOINT)

    # Test all 3 sample images
    samples = [
        ("SampleImages/sample_portrait_bw.jpg",  "SampleImages/result_portrait.png"),
        ("SampleImages/sample_landscape.jpg",     "SampleImages/result_landscape.png"),
        ("SampleImages/sample_woman.jpg",         "SampleImages/result_woman.png"),
        ("SampleImages/test_input1.jpg",          "SampleImages/test_result.png"),   # your own image
    ]

    for img_path, out_path in samples:
        if os.path.exists(img_path):
            print(f"\nProcessing: {img_path}")
            # Try it with NO DEGRADATION (degrade_input=False) so we only restore/colorize real input!
            test_image(img_path, restoration_model, colorization_model, out_path=out_path, degrade_input=False)
        else:
            print(f"Skipping {img_path} — file not found.")
