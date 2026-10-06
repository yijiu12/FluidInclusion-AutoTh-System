# FluidInclusion AutoTh System

Automated fluid inclusion homogenization temperature measurement system based on deep learning and hardware automation.

## Overview
This system implements fully automatic homogenization temperature (Th) measurement of fluid inclusions without requiring open API access to the commercial heating stage. It combines YOLO object detection, U-Net++ semantic segmentation, GUI automation, and real-time temperature reading to achieve batch measurement of fluid inclusions.

### Key Features
- **No-API hardware control**: GUI automation + IPC temperature reading for Linkam MDSG600
- **Deep learning pipeline**: YOLO26n detection + U-Net++ gas-liquid segmentation
- **Global scanning**: Full-slide scanning and inclusion database construction
- **3D relocation**: Accurate target repositioning via spatial coordinates
- **Auto-focus**: Joint sharpness-confidence auto-focus algorithm
- **Real-time target locking**: automatic target re-localization during heating
- **Batch measurement**: Automated batch heating/cooling measurement protocol

## Paper Information
- **Title**: Automated fluid inclusion homogenization temperature measurement powered by machine learning and GUI automation
- **Journal**: Computers & Geosciences
- **DOI**: to be added upon publication
- **Authors**: Shijie Li, Weiqiang Li, Guoguang Wang, and Rubing Shao

## Hardware Requirements
- Linkam MDSG600 heating/freezing stage
- Olympus CX40M microscope (200×/500× magnification)
- Blackfly BFLY-U3-23S6C industrial camera
- Z-axis stepping motor (DTStageDriver)
- Windows 11 PC (x64)

## Installation

### 1. Python Environment
```bash
pip install -r requirements.txt
```

### 2. Hardware Setup
Refer to `docs/hardware_guide.md` for detailed hardware connection and calibration instructions.

### 3. Third-party hardware libraries (not bundled)
Two proprietary runtime libraries are **not** redistributed in this repository and must
be obtained separately:

| Library | Where to get it | Required for |
|---|---|---|
| `DTStageDriver.dll` | Z-axis motor manufacturer's SDK | Z-axis focus control |
| `LinkamIpc.dll` | Build from the Linkam IPC SDK, or use the pre-built copy in `csharp_ipc_builder/bin/x64/Release/` | Reading live temperature from Linkam NEXUS |

Place `DTStageDriver.dll` in `src/hardware/lib/` (the path is set by `Z_AXIS_DLL` in `config.py`).
The C# source used to build the IPC bridge is provided in `csharp_ipc_builder/`.

### 4. Configuration
Edit `config.py` to match your hardware setup:
- Screen coordinates for Linkam NEXUS GUI
- Serial port for Z-axis motor
- Capture region coordinates
- Model weight paths

## Quick Start

### Model-only Demo (no hardware required)
```bash
# Test YOLO detection
python demo/test_yolo_infer.py --input dataset/yolo_detect/sample/images/<image>.png

# Test U-Net++ segmentation
python demo/test_unet_infer.py --input dataset/unet_segment/sample/images/<image>.png
```

### Batch Measurement (hardware required)
`main.py` takes the inclusion database CSV produced by the global scan, and a rank range
to measure. Ranks are ordered by detection confidence (descending).

```bash
# Measure the top-ranked inclusion
python main.py --start 1 --end 1

# Measure inclusions ranked 9 to 50 from a custom database
python main.py --csv experiments/global_scan/inclusions_with_morphology.csv --start 9 --end 50

# Short form, custom output directory
python main.py -s 1 -e 10 -o ./output
```

| Argument | Short | Default | Meaning |
|---|---|---|---|
| `--csv` | `-c` | `experiments/global_scan/inclusions_with_morphology.csv` | Inclusion database (CSV) |
| `--start` | `-s` | `config.BATCH_START_RANK` | First rank to measure (1-based, inclusive) |
| `--end` | `-e` | `config.BATCH_END_RANK` | Last rank to measure (inclusive) |
| `--output` | `-o` | `config.OUTPUT_BASE` | Output directory |

A ready-to-run example of the global scan output (`inclusions_with_morphology.csv`) is
included in `experiments/global_scan/`, so the batch workflow can be launched on real data
without re-scanning a slide.

## Project Structure
```
├── src/
│   ├── hardware/              # Hardware automation control
│   ├── dl_model/              # Deep learning models (YOLO + U-Net++)
│   ├── tracking_autofocus/    # Target locking and auto-focus algorithms
│   ├── measurement/           # Homogenization temperature measurement
│   └── utils/                 # Common utilities
├── csharp_ipc_builder/        # C# project for Linkam IPC DLL
├── weights/                   # Trained model weights (YOLO26n, U-Net++/ResNet34)
├── dataset/                   # Demonstration subset of the annotated dataset
├── experiments/               # Numerical results behind the figures of the paper
├── training_output/           # Training logs, per-epoch metrics, curves and checkpoints
├── docs/                      # Documentation
├── demo/                      # Quick test scripts
├── config.py                  # Global configuration
├── main.py                    # Main program entry
├── requirements.txt
├── LICENSE
└── README.md
```

### Not included in this repository
- `docs/vendor_sdk/` — manufacturer SDK bundles; redistributed by their vendors under
  their own terms.
- The **complete** `dataset/` — only a demonstration subset is included here. The full
  annotated dataset (4,362 detection annotations and 14,057 segmentation masks) is
  archived on Zenodo.
- Two training artefacts are byte-identical duplicates of the weights shipped in
  `weights/` and are therefore not stored twice:
  `training_output/unetpp_res34/unet_train_results/best_model.pth` and
  `training_output/yolo26n/yolo_train_results/weights/best.pt`.

## Data Availability
- **Complete annotated dataset** (4,362 detection annotations and 14,057 segmentation
  masks) and the **raw experimental data** are archived on Zenodo: https://doi.org/10.5281/zenodo.23114354
- **Trained model weights** — `yolo26n_best.pt` ships with this repository under `weights/`.
  `unetpp_res34_best.pth` (99.7 MiB) exceeds the practical size of a Git object and is
  attached to the [`v1.0.0` release](https://github.com/yijiu12/FluidInclusion-AutoTh-System/releases/tag/v1.0.0);
  download it into `weights/` to run the pipeline without downloading the dataset.
- The full **training artefacts** (per-epoch metrics, curves, hyperparameters, evaluation
  reports and checkpoints) are included in `training_output/`.
- A **demonstration subset** of the dataset and all **numerical results** behind the
  figures are included in `dataset/` and `experiments/`.

## Citation
If you use this code in your research, please cite our paper:
```
Li, S., Li, W., Wang, G., Shao, R. Automated fluid inclusion homogenization temperature
measurement powered by machine learning and GUI automation. Computers & Geosciences.
to be added upon publication
```
Machine-readable citation metadata is provided in `CITATION.cff`.

## License
MIT License - see LICENSE file for details.

## Contact
- Corresponding author: Weiqiang Li (School of Earth Sciences and Engineering, Nanjing University)
- Email: liweiqiang@nju.edu.cn
- Repository: https://github.com/yijiu12/FluidInclusion-AutoTh-System
