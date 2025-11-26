import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ------------------------
# CONFIG
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_unet_rgbd.pth"

IMG_SIZE = 320
DEPTH_MAX_MM = 1000.0     # same normalization used during training
N_CLASSES = 4

# Colors for visualization
CLASS_COLORS = {
    0: (0, 0, 0),         # background   - black
    1: (255, 0, 0),       # flesh        - red
    2: (0, 255, 0),       # dissection   - green
    3: (0, 0, 255),       # tool         - blue
}

# ------------------------
# Model builder
# ------------------------
def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,         # RGB (3) + depth (1)
        classes=N_CLASSES,
    )

# ------------------------
# Input loader
# ------------------------
def load_rgbd(rgb_path, depth_path):
    # Load & normalize RGB
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    # Load depth (.npy)
    depth = np.load(depth_path).astype(np.float32)
    depth = np.clip(depth, 0, DEPTH_MAX_MM) / DEPTH_MAX_MM
    depth = depth[..., None]  # shape (H, W, 1)

    # Stack to 4 channels
    x = np.concatenate([rgb, depth], axis=-1)

    # Resize to model resolution
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Convert to tensor
    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)  # (1,4,H,W)
    return x

# ------------------------
# Mask colorizer
# ------------------------
def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in CLASS_COLORS.items():
        color[mask == cls] = col
    return color

# ------------------------
# Prediction function
# ------------------------
def predict(rgb_path, depth_path, save_prefix="result"):
    print(f"Loading model: {MODEL_PATH}")

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load the image pair
    x = load_rgbd(rgb_path, depth_path).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # Load RGB for overlay
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    # Color mask + overlay
    mask_color = colorize_mask(pred_mask)
    overlay = cv2.addWeighted(rgb_small, 0.6, mask_color, 0.4, 0)

    # Save results
    os.makedirs("predictions", exist_ok=True)

    cv2.imwrite(f"predictions/{save_prefix}_rgb.png", cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"predictions/{save_prefix}_mask.png", mask_color)
    cv2.imwrite(f"predictions/{save_prefix}_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved predictions:")
    print(f"  predictions/{save_prefix}_rgb.png")
    print(f"  predictions/{save_prefix}_mask.png")
    print(f"  predictions/{save_prefix}_overlay.png")

    # Save depth visualization
    depth = np.load(depth_path)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype("uint8")
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    cv2.imwrite(f"predictions/{save_prefix}_depth.png", depth_color)


    return pred_mask, mask_color, overlay


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    fname = "session_01_frame_00018_crop.png"

    rgb = f"/workspace/final_dataset_split/val/images/{fname}"
    depth = f"/workspace/final_dataset_split/val/depth/{fname.replace('.png', '.npy')}"

    predict(rgb, depth, save_prefix="test_prediction")

