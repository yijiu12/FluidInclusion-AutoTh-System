# Model Weights

> `unetpp_res34_best.pth` is 99.7 MiB, which exceeds the practical size of a Git object,
> so it is **not** committed to the source tree. Download it from the
> [`v1.0.0` release](https://github.com/yijiu12/FluidInclusion-AutoTh-System/releases/tag/v1.0.0)
> and place it in this directory next to `yolo26n_best.pt`.

## YOLO26n - Fluid Inclusion Detection
- **File**: `yolo26n_best.pt`
- **Test metrics**: precision = 78.56%, recall = 72.37%, mAP@0.5 = 82.35%, mAP@0.5:0.95 = 56.34%
- **Inference speed**: 6.30 FPS on CPU (158.7 ms per 1024×1024 frame)
- **Input size**: 1024×1024
- **Classes**: 1 (fluid inclusion)

## U-Net++ (ResNet34) - Gas-Liquid Segmentation
- **File**: `unetpp_res34_best.pth`
- **Test metrics**: pixel accuracy = 93.79%, mIoU = 83.91%, mDice = 91.09%
  (per class IoU: background 93.38%, liquid 82.28%, gas 76.06%)
- **Best validation mIoU during training**: 84.05% (epoch 21)
- **Inference speed**: 7.02 FPS on CPU (142.5 ms per 256×256 crop)
- **Input size**: 256×256
- **Classes**: 3 (background, liquid, gas)

`unetpp_res34_best.pth` is stored with `torch.save` as a state dict; load it with
`UNET_ENCODER = "resnet34"` and `NUM_CLASSES = 3` (see `config.py`).

