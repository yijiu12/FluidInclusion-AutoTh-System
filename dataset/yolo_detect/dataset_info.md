# YOLO Detection Dataset

## Overview
- **Total images**: 4362 (full dataset on Zenodo)
- **Annotation tool**: LabelImg
- **Classes**: 1 (fluid_inclusion)
- **Input size**: 1024×1024
- **Train/Val/Test split**: 8:1:1

## Preprocessing
1. Grayscale world white balance (color correction)
2. Laplacian sharpening (image enhancement)
3. Resize to 1024×1024

## Data Augmentation
- Horizontal/vertical flip
- Rotation (±15°)
- Random scaling
- Mosaic augmentation
- Color jitter

## Sample Folder
Example images with corresponding XML annotation files for demonstration.

## Tools
Dataset processing scripts are at `src/dl_model/detection_model/dataset_tools/`
