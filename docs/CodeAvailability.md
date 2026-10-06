# Code Availability

## Software Information
- **Software name**: FluidInclusion_AutoTh_System
- **Programming language**: Python 3.11, C# (.NET Framework 4.8)
- **Operating system**: Windows 11
- **Hardware requirements**: 
  - Linkam MDSG600 heating/freezing stage
  - Olympus CX40M microscope
  - Blackfly BFLY-U3-23S6C industrial camera
  - Z-axis stepping motor (DTStageDriver)
- **GPU**: Not required for inference; CPU-only real-time inference supported. Training was performed on NVIDIA RTX 3050 (4GB).

## Third-party Dependencies
### Python libraries
- ultralytics (YOLO26n)
- torch, torchvision
- segmentation-models-pytorch
- pyautogui
- pythonnet (.NET interop)
- pyserial
- opencv-python
- numpy, pandas, matplotlib, scipy, scikit-image

### Hardware libraries
- LinkamIpc.dll (custom-built from Linkam IPC SDK, source provided in `csharp_ipc_builder/`)
- DTStageDriver.dll (provided by the Z-axis motor manufacturer; not redistributed in this repository)

## Data Availability
- **Complete dataset**: 4,362 detection annotations and 14,057 segmentation masks. Archived on Zenodo with a permanent DOI: https://doi.org/10.5281/zenodo.23114354
- **Sample dataset**: a demonstration subset is included in the `dataset/` folder for quick testing.
- **Model weights**: the best-performing YOLO26n weight is provided in the `weights/` folder of the source-code repository; the U-Net++ weight is attached to the repository's `v1.0.0` release.
- **Raw experimental data**: full-resolution image sequences and measurement records (global scan, batch run, repeatability test) are archived on Zenodo alongside the dataset.

## Repository Contents
The source-code repository contains the complete runtime code, the trained model weights,
a demonstration subset of the dataset, the training artefacts and all numerical results
behind the figures of the paper. The following are deliberately not included:

- `docs/vendor_sdk/` — third-party SDK bundles redistributed by their vendors under their
  own terms.
- Two files under `training_output/` are byte-identical duplicates of the weights shipped
  in `weights/` (`unetpp_res34/unet_train_results/best_model.pth` and
  `yolo26n/yolo_train_results/weights/best.pt`) and are therefore not stored twice.

## Repository Structure
```
├── src/hardware/          # Hardware automation control
├── src/dl_model/          # YOLO detection + U-Net++ segmentation
├── src/tracking_autofocus/ # Tracking and auto-focus algorithms
├── src/measurement/       # Homogenization temperature measurement workflow
├── src/utils/             # Common utilities
├── csharp_ipc_builder/    # C# project for LinkamIpc.dll compilation
├── dataset/               # Sample datasets and documentation
├── training_output/       # Training logs, metrics, curves and checkpoints
├── weights/               # Trained model weights
├── docs/                  # Documentation and vendor SDKs
└── main.py                # Main program entry
```

## Code Functionality
1. **Global scanning**: Full slide scanning and fluid inclusion database construction
2. **3D relocation**: Target repositioning via 3D spatial search
3. **Auto-focus**: Joint sharpness-confidence auto-focus algorithm
4. **Real-time target locking**: automatic target re-localization during heating
5. **Homogenization detection**: Three-frame stable window homogenization temperature identification
6. **Batch measurement**: Automated batch heating/cooling measurement protocol

## Output Formats
- CSV temperature measurement data
- PNG segmentation masks and detection visualizations
- Temperature curve plots
- Batch statistical histograms

## Contact
- Corresponding author: Weiqiang Li (School of Earth Sciences and Engineering, Nanjing University)
- Email: liweiqiang@nju.edu.cn
- GitHub repository: https://github.com/yijiu12/FluidInclusion-AutoTh-System
