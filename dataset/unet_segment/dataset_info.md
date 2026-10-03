# U-Net++ Segmentation Dataset

## Overview
- **Total images**: 14057 (full dataset on Zenodo)
- **Annotation tool**: LabelMe
- **Classes**: 3 (background, liquid, gas)
- **Input size**: 256×256
- **Train/Val/Test split**: 8:1:1

## Preprocessing
1. Crop single inclusion ROIs using YOLO detection bounding boxes
2. Grayscale world white balance
3. Laplacian sharpening
4. Resize to 256×256
5. Convert JSON annotations to binary masks

## Sample Folder
Example cropped inclusion images with:
- Original cropped image
- LabelMe JSON annotation
- Generated mask image (PNG)

## Tools
Dataset processing scripts are at `src/dl_model/segmentation_model/dataset_tools/`
