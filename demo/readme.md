# Demo Scripts
Quick test scripts to verify individual modules without running the full system.

## Scripts
- `test_yolo_infer.py`: Test YOLO26n fluid inclusion detection
  ```bash
  python demo/test_yolo_infer.py --input sample_image.jpg
  ```

- `test_unet_infer.py`: Test U-Net++ gas-liquid segmentation
  ```bash
  python demo/test_unet_infer.py --input cropped_inclusion.jpg
  ```

All outputs are saved to `test_output/` folder.
