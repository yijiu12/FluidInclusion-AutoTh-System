"""
U-Net++ Gas-Liquid Segmentation - Quick Test Script
Usage: python test_unet_infer.py --input cropped_inclusion.jpg
"""
import os
import argparse
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# Default paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "unetpp_res34_best.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "unet_segmentation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "resnet34"
NUM_CLASSES = 3

# Color map: background=black, liquid=red, gas=blue
COLOR_MAP = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [255, 0, 0],
}


def load_model():
    """Load U-Net++ model"""
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model


def segment(image_path, model):
    """Run segmentation on a single cropped inclusion image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None, None, None

    orig_h, orig_w = image.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = cv2.resize(img_rgb, (256, 256))
    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    input_tensor = (input_tensor - mean) / std
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    pred_orig = cv2.resize(pred_256.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Generate visualization
    vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        vis[pred_orig == cls_id] = color

    # Overlay on original image
    overlay = cv2.addWeighted(image, 0.6, vis, 0.4, 0)

    # Calculate G/L ratio
    liquid = int(np.sum(pred_orig == 1))
    gas = int(np.sum(pred_orig == 2))
    ratio = gas / liquid if liquid > 0 else float('inf')

    return overlay, pred_orig, ratio


def main():
    parser = argparse.ArgumentParser(description="U-Net++ Gas-Liquid Segmentation Test")
    parser.add_argument("--input", type=str, required=True, help="Input cropped inclusion image path")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading U-Net++ model on {DEVICE}...")
    model = load_model()
    print("Model loaded successfully.")

    overlay, mask, ratio = segment(args.input, model)
    if overlay is None:
        return

    # Save outputs
    img_name = os.path.basename(args.input)
    name_no_ext = os.path.splitext(img_name)[0]

    overlay_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_overlay.png")
    mask_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_mask.png")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask)

    print(f"\nSegmentation complete:")
    print(f"  Liquid pixels: {liquid}")
    print(f"  Gas pixels: {gas}")
    print(f"  Gas/Liquid ratio: {ratio:.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  Overlay: {overlay_path}")
    print(f"  Mask: {mask_path}")


if __name__ == "__main__":
    main()
