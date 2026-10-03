# Run Tutorial

All commands are run from the repository root, because every runtime module does
`import config` from the project root.

## Environment Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Hardware Parameters
Edit `config.py` to match your setup:
- Screen coordinates for Linkam NEXUS GUI (`NEXUS_POS`, `TEMP_*_POS`, `X_INPUT_POS`, `Y_INPUT_POS`, `SCREEN_REGION`)
- Serial port for Z-axis motor (`COM_PORT`, default `3` = COM3)
- Stage step calibration (`STEP_X_UM_PER_SCREEN`, `STEP_Y_UM_PER_SCREEN`, `DIR_X`, `DIR_Y`, `UM_PER_PIXEL`)
- Heating programme (`STAGE1_RATE`, `STAGE1_TARGET`, `STAGE2_RATE`, `TARGET_TEMP_MAX`, `SAMPLE_INTERVAL`)
- Homogenization criterion (`STABILITY_WINDOW`, `RATIO_THRESHOLD`)
- Model weight paths (`YOLO_MODEL_PATH`, `UNET_MODEL_PATH`)

### 3. Verify the Hardware Chain
There is no separate hardware self-test script. Verify the chain in this order before a
full run:
1. Start Linkam NEXUS and leave the window at the calibrated position.
2. Mount a slide and run a single target measurement (see *Batch Measurement* below with
   `--start 1 --end 1`).
3. Confirm from the console output that (a) live temperature is read back, (b) the XY
   stage moves to the stored coordinate, and (c) the captured frame contains an inclusion.

## Workflows

### 1. Global Scan — build the inclusion database
Scans the slide, detects inclusions in every field of view, and writes the database.

```bash
# Hardware required
python -m src.measurement.integrated_scan_analysis
```

**Output**: per-field image captures, detections and gas-liquid statistics.

### 2. Global Scan Analysis — morphology statistics and figures
Re-derives the inclusion database and morphology plots from a scan result directory.
This step needs no hardware and runs directly on the example data shipped with the
repository.

```bash
python src/measurement/analysis_scan.py
```

**Input**: `experiments/global_scan/` (set by `SCAN_RESULT_DIR` in the script)
**Output**: `experiments/global_scan/morphology_statistics/`
- `inclusions_with_morphology.csv` — per-inclusion coordinates, size, aspect ratio, circularity
- `morphology_summary.csv` — aggregate statistics
- distribution plots

### 3. Batch Measurement
Measures homogenization temperature for a rank range of inclusions from the database.
Ranks are ordered by detection confidence, descending.

```bash
# Measure the top-ranked inclusion only
python main.py --start 1 --end 1

# Measure inclusions ranked 9 to 50
python main.py -c experiments/global_scan/inclusions_with_morphology.csv -s 9 -e 50

# Custom output directory
python main.py -s 1 -e 10 -o ./output
```

| Argument | Short | Default | Meaning |
|---|---|---|---|
| `--csv` | `-c` | `experiments/global_scan/inclusions_with_morphology.csv` | Inclusion database (CSV) |
| `--start` | `-s` | `config.BATCH_START_RANK` | First rank to measure (1-based, inclusive) |
| `--end` | `-e` | `config.BATCH_END_RANK` | Last rank to measure (inclusive) |
| `--output` | `-o` | `config.OUTPUT_BASE` | Output directory |

**Workflow per target**: relocate to the stored coordinate → 3D spatial search → mechanical
recentering → auto-focus → two-stage heating ramp → real-time target locking and gas-liquid
segmentation → stability-window homogenization detection → active cooling before the next
target.

**Output per target** (see `experiments/single_inclusion_sample/`):
- `report.json` — full measurement metadata (target, coordinates, timestamps, detected Th)
- `temperature_ratio_curve.csv` — temperature vs. gas-liquid ratio time series
- `curve_plot.png` — visualization of the curve
- `vis/`, `roi_raw/`, `full_ss/`, `full_det/` — per-frame segmentation and detection
  visualizations

## Model-only Demo (No Hardware Required)
Test the deep learning models without any hardware:

```bash
# YOLO26n fluid inclusion detection
python demo/test_yolo_infer.py --input dataset/yolo_detect/sample/images/<image>.png

# U-Net++ gas-liquid segmentation
python demo/test_unet_infer.py --input dataset/unet_segment/sample/images/<image>.png
```

Both scripts write their visualizations to `test_output/`.

## Notes
1. Always perform a test run on a known sample before a long batch.
2. Keep the NEXUS window position and screen resolution fixed — GUI coordinates are absolute.
3. Keep the Linkam stage temperature within its safe operating range (`Z_LIMIT_MIN/MAX`,
   `TARGET_TEMP_MAX` in `config.py`).
4. Clean the slide before scanning to avoid dust particles triggering false detections.
5. For higher precision, lower `STAGE2_RATE`; this lengthens each measurement but narrows
   the temperature interval between samples near homogenization.
