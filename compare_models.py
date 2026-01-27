"""
CPU-only comparison of MIMO-UNet and MIMO-UNet-Enhanced
Displays: Blurry Input | MIMO-UNet | MIMO-UNet-Enhanced | Ground Truth
Also prints PSNR values
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from models.MIMOUNet import build_net

# ----------------------------- PSNR Function -----------------------------
def calculate_psnr(img1, img2, max_val=255.0):
    """Compute PSNR between two images (NumPy or PIL)."""
    if isinstance(img1, Image.Image):
        img1 = np.array(img1).astype(np.float32)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

# ----------------------------- Image Loader ------------------------------
def load_and_preprocess_image(image_path, target_size=(1280, 720)):
    """Load and resize image if necessary."""
    img = Image.open(image_path).convert('RGB')
    if img.size != target_size:
        print(f"Resizing {os.path.basename(image_path)} from {img.size} to {target_size}")
        img = T.Resize((target_size[1], target_size[0]))(img)
    return img

# ----------------------------- Deblur Function ---------------------------
def deblur_with_model(img, model):
    """Run inference with given model on CPU."""
    to_tensor = T.Compose([T.ToTensor()])
    input_tensor = to_tensor(img).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)[-1]

    output_tensor = output_tensor.squeeze(0).clamp(0, 1).cpu()
    return F.to_pil_image(output_tensor, 'RGB')

# ----------------------------- Compare Models ----------------------------
def compare_models(input_path, gt_path, model1_path, model2_path, output_path):
    """Compare MIMO-UNet and MIMO-UNet-Enhanced on CPU."""
    device = torch.device("cpu")
    print(f"🧠 Running on device: {device}\n")

    # Load models
    print("📦 Loading MIMO-UNet ...")
    model1 = build_net("MIMO-UNet")
    state_dict1 = torch.load(model1_path, map_location="cpu")
    if "model" in state_dict1:
        state_dict1 = state_dict1["model"]
    model1.load_state_dict(state_dict1, strict=False)
    model1.to(device).eval()
    print("✅ MIMO-UNet loaded.\n")

    print("📦 Loading MIMO-UNet-Enhanced ...")
    model2 = build_net("MIMO-UNet-Enhanced")
    state_dict2 = torch.load(model2_path, map_location="cpu")
    if "model" in state_dict2:
        state_dict2 = state_dict2["model"]
    model2.load_state_dict(state_dict2, strict=False)
    model2.to(device).eval()
    print("✅ MIMO-UNet-Enhanced loaded.\n")

    # Load images
    blurry_img = load_and_preprocess_image(input_path)
    ground_truth = load_and_preprocess_image(gt_path) if gt_path else None

    # Run inference
    print("🚀 Running MIMO-UNet...")
    output1 = deblur_with_model(blurry_img, model1)
    print("✅ MIMO-UNet done.\n")

    print("🚀 Running MIMO-UNet-Enhanced...")
    output2 = deblur_with_model(blurry_img, model2)
    print("✅ MIMO-UNet-Enhanced done.\n")

    # Calculate PSNR
    if ground_truth:
        psnr_blur = calculate_psnr(blurry_img, ground_truth)
        psnr_mimo = calculate_psnr(output1, ground_truth)
        psnr_enh = calculate_psnr(output2, ground_truth)

        print("📈 PSNR Scores:")
        print(f"  Blurry Input:         {psnr_blur:.2f} dB")
        print(f"  MIMO-UNet:            {psnr_mimo:.2f} dB (+{psnr_mimo - psnr_blur:.2f} dB)")
        print(f"  MIMO-UNet-Enhanced:   {psnr_enh:.2f} dB (+{psnr_enh - psnr_blur:.2f} dB)")
        print(f"  🔹 Improvement (Enhanced vs Base): {psnr_enh - psnr_mimo:.2f} dB\n")
    else:
        print("⚠️ No ground truth provided — cannot calculate absolute PSNR.\n")

    # ------------------ Visualization ------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    images = [blurry_img, output1, output2, ground_truth]
    titles = [
        f"Blurry Input\nPSNR: {psnr_blur:.2f} dB" if ground_truth else "Blurry Input",
        f"MIMO-UNet\nPSNR: {psnr_mimo:.2f} dB" if ground_truth else "MIMO-UNet",
        f"MIMO-UNet-Enhanced\nPSNR: {psnr_enh:.2f} dB" if ground_truth else "MIMO-UNet-Enhanced",
        "Ground Truth" if ground_truth else "—"
    ]

    for ax, img, title in zip(axes, images, titles):
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"💾 Comparison figure saved to: {output_path}\n")
    plt.close()

# ----------------------------- Main -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MIMO-UNet and MIMO-UNet-Enhanced on CPU.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to blurry input image")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to sharp ground truth image")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to MIMO-UNet model weights")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to MIMO-UNet-Enhanced model weights")
    parser.add_argument("--output", type=str, required=True, help="Path to save comparison figure")

    args = parser.parse_args()
    compare_models(
        input_path=args.input_image,
        gt_path=args.ground_truth,
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        output_path=args.output
    )
