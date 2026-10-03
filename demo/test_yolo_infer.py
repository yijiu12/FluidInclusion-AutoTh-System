"""
YOLO26n Fluid Inclusion Detection - Quick Test Script
Usage: python test_yolo_infer.py --input your_image.jpg
"""
import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Default paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "yolo26n_best.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "yolo_detection")


def detect(image_path, model, conf_threshold=0.5):
    """Run YOLO detection on a single image"""
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # Predict
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    result = results[0]
    boxes = result.boxes

    # Draw boxes
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    detection_count = 0
    if boxes is not None and len(boxes) > 0:
        detection_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            label = f"{model.names[cls_id]} {conf:.2f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1-text_h-4, x1+text_w+4, y1], fill='red')
            draw.text((x1+2, y1-text_h-2), label, fill='white', font=font)

    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result_img, detection_count


def main():
    parser = argparse.ArgumentParser(description="YOLO Fluid Inclusion Detection Test")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    result_img, count = detect(args.input, model, args.conf)
    if result_img is None:
        return

    # Save output
    img_name = os.path.basename(args.input)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{img_name}")
    cv2.imwrite(output_path, result_img)

    print(f"\nDetection complete: {count} inclusions found")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
