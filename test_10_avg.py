import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.MIMOUNet import build_net

# ============================================================
# CONFIG (MATCHES pipeline.py)
# ============================================================
MIMO_ORIGINAL = "weights/MIMO-UNet.pkl"
MIMO_ENHANCED = "results/MIMO-UNet-Enhanced/weights/model.pkl"

DATASET_ROOT = "dataset/GOPRO/test"   # automatically works with your folder
BLUR_DIR = os.path.join(DATASET_ROOT, "blur")
SHARP_DIR = os.path.join(DATASET_ROOT, "sharp")

DEVICE = torch.device("cpu")

# ============================================================
# Load Models (same code as pipeline.py)
# ============================================================
def load_mimo(path, model_type):
    print(f"\n📦 Loading {model_type} from {path}")
    model = build_net(model_type)
    state = torch.load(path, map_location="cpu")

    if "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded {model_type}")
    return model


mimo_baseline = load_mimo(MIMO_ORIGINAL, "MIMO-UNet")
mimo_enhanced = load_mimo(MIMO_ENHANCED, "MIMO-UNet-Enhanced")

# ============================================================
# Load blur/sharp pair
# ============================================================
def load_pair(name):
    blur = Image.open(os.path.join(BLUR_DIR, name)).convert("RGB")
    sharp = Image.open(os.path.join(SHARP_DIR, name)).convert("RGB")
    return to_tensor(blur).unsqueeze(0), to_tensor(sharp).unsqueeze(0)

# ============================================================
# Compute PSNR & SSIM
# ============================================================
def compute_metrics(pred, gt):
    p = pred.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    g = gt.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    psnr = peak_signal_noise_ratio(g, p, data_range=1.0)
    ssim = structural_similarity(g, p, multichannel=True, data_range=1.0)
    return psnr, ssim

# ============================================================
# Evaluate 10 random images
# ============================================================
all_imgs = sorted(os.listdir(BLUR_DIR))
random.shuffle(all_imgs)
test_imgs = all_imgs[:10]

baseline_psnr = []
enhanced_psnr = []

baseline_ssim = []
enhanced_ssim = []

print("\n================= TESTING 10 RANDOM IMAGES =================\n")

for img in test_imgs:
    print(f"🔍 Processing: {img}")
    blur, sharp = load_pair(img)

    with torch.no_grad():
        out_baseline = mimo_baseline(blur)[-1].clamp(0, 1)
        out_enhanced = mimo_enhanced(blur)[-1].clamp(0, 1)

    psnr_b, ssim_b = compute_metrics(out_baseline, sharp)
    psnr_e, ssim_e = compute_metrics(out_enhanced, sharp)

    baseline_psnr.append(psnr_b)
    enhanced_psnr.append(psnr_e)
    baseline_ssim.append(ssim_b)
    enhanced_ssim.append(ssim_e)

# ============================================================
# Print results
# ============================================================
avg_b_psnr = np.mean(baseline_psnr)
avg_e_psnr = np.mean(enhanced_psnr)

avg_b_ssim = np.mean(baseline_ssim)
avg_e_ssim = np.mean(enhanced_ssim)

print("\n==================== FINAL RESULTS ====================")
print(f"Baseline PSNR:        {avg_b_psnr:.3f} dB")
print(f"Enhanced PSNR:        {avg_e_psnr:.3f} dB")
print(f"Improvement:         +{(avg_e_psnr - avg_b_psnr):.3f} dB")

print("--------------------------------------------------------")
print(f"Baseline SSIM:        {avg_b_ssim:.4f}")
print(f"Enhanced SSIM:        {avg_e_ssim:.4f}")
print(f"Improvement:         +{(avg_e_ssim - avg_b_ssim):.4f}")

print("========================================================\n")
