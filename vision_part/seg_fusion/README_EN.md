# split_yolo_seg

## Overview

`split_yolo_seg` is an experimental workflow for instance segmentation in mixed scenes containing `robot`, `person`, and `car` objects.

The core issue addressed by this project is partial annotation in the custom dataset:

- only the `robot` category is annotated in the custom data;
- `person` and `car` may still appear in the same images;
- if this dataset is used as fully supervised data, unannotated `person` and `car` instances may be incorrectly treated as background, degrading both classification and segmentation quality.

To mitigate this problem, the current solution adopts a staged training and weight-fusion strategy:

1. train a target-domain model using the custom `robot` data;
2. freeze the backbone during the first stage to reduce damage to generic visual features;
3. fuse the custom-domain weights with weights trained on COCO data;
4. use a shared backbone with two heads to preserve both generic-category capability and `robot`-specific capability.

This README focuses on the training workflow, repository structure, experimental process, and intermediate export artifacts. It does not cover deployment, runtime integration, or production inference services.

## Background

In the current custom dataset, only `robot` is explicitly annotated, while `person` and `car` may still be present in the same frames.

Under a conventional supervised training setup, unannotated objects are implicitly treated as background. This introduces two major risks:

- suppression of `person` and `car` features that should otherwise be preserved from general-purpose pretraining;
- contamination of `robot` feature learning, especially near boundaries and overlapping regions.

For this reason, the project does not rely on a single-stage fine-tuning strategy. Instead, it separates target-domain adaptation from generic-category preservation.

## Method

### Stage 1: Robot-Oriented Domain Adaptation

The first stage adapts the model to the target domain using the custom `robot` dataset.

Key design choices:

- the backbone is frozen during training;
- adaptation is performed primarily in the head.

The objective is to:

- preserve as much generic pretrained representation as possible;
- improve segmentation quality for the `robot` category without rewriting the full feature extractor.

### Stage 2: Weight Fusion

After robot-oriented adaptation, the resulting weights are fused with weights trained on COCO data.

The current structural design is:

- one shared backbone;
- two separate heads;
- one head biased toward preserving generic categories such as `person` and `car`;
- one head biased toward strengthening the custom `robot` category.

The goal of this stage is to enhance `robot` performance while retaining generic-category recognition capability.

## Repository Structure

The following files and subdirectories are currently organized in this repository:

- [train.py](train.py): training entry point for the current first-stage robot adaptation workflow
- [video_prediction.py](video_prediction.py): video validation script for `.pt` checkpoints
- [onnx_inference.py](onnx_inference.py): offline ONNX inference script
- [onnx_camera_inference.py](onnx_camera_inference.py): ONNX camera inference script
- [demo_video.py](demo_video.py): video demo script
- [demo/](demo): demo-related resources
- [data_boost/](data_boost): data augmentation scripts
- [fusion_export/](fusion_export): model fusion, weight composition, and export scripts
- [ckpt/](ckpt): intermediate checkpoints and exported artifacts

More specifically:

- [data_boost/data_copypaste.py](data_boost/data_copypaste.py) is used for Copy-Paste augmentation;
- [data_boost/data_zoomout.py](data_boost/data_zoomout.py) is used for Zoom-out / small-object augmentation;
- [fusion_export/robotbase](fusion_export/robotbase) contains experiments that preserve the complete `robot` head;
- [fusion_export/fusion10](fusion_export/fusion10) contains experiments that preserve the main YOLO branch while fusing the `robot` head.

## Environment Requirements

Python 3.10 or later is recommended. Install the following core dependencies:

```bash
pip install ultralytics torch opencv-python onnx onnxruntime numpy
```

For GPU usage, install the appropriate PyTorch build matching the local CUDA version.

## Current Training Configuration

The current training entry point is [train.py](train.py), which uses the following default configuration:

- pretrained model: `yolo11s-seg.pt`
- dataset: `data_3/dataset.yaml`
- epochs: `100`
- image size: `640 x 640`
- device: `device=[1]`
- frozen layers: `freeze=10`

At the current stage, this script should be understood as the entry point for `robot`-oriented domain adaptation rather than the final end-to-end training framework.

## Available Checkpoints and Intermediate Artifacts

The [ckpt](ckpt) directory currently contains the following files:

- [robot_best.pt](ckpt/robot_best.pt)
- [person_car_best.pt](ckpt/person_car_best.pt)
- [fusion_model.onnx](ckpt/fusion_model.onnx)

These files are intermediate research artifacts and should not be treated as finalized release assets.

In particular, `fusion_model.onnx` represents an intermediate exported result from the current fusion workflow and is intended for offline validation rather than final deployment.

## Quick Start

### Training

Run the first-stage training workflow in the current directory:

```bash
python train.py
```

If training parameters need to be changed, update [train.py](train.py) directly.

### Weight Fusion and Export

Fusion and export logic is maintained under [fusion_export/](fusion_export). Available scripts depend on the specific experiment implementation, for example:

```bash
python fusion_export/robotbase/fusion_robotbase_export.py
python fusion_export/fusion10/fusion_branch10_export.py
```

Notes:

- different scripts correspond to different fusion strategies;
- exported ONNX files are intermediate research artifacts;
- output order, class definition, and mask parsing must remain consistent with downstream post-processing logic.

### Offline Validation

To validate an intermediate exported model offline, use:

```bash
python onnx_inference.py --model ckpt/fusion_model.onnx --source /path/to/input
```

To validate `.pt` checkpoints directly, use:

```bash
python video_prediction.py
```

## Recommended Experimental Workflow

The recommended workflow is as follows:

1. train a `robot`-oriented model using the custom dataset;
2. evaluate the intermediate `robot` checkpoint;
3. fuse this checkpoint with weights that preserve `person` and `car` capability;
4. export an intermediate ONNX model;
5. validate the fused result using offline inference scripts.

## Important Notes

1. The custom dataset is partially annotated, and all training results must be interpreted under this assumption.
2. Freezing the backbone in the first stage is an intentional design choice to preserve generic pretrained representations.
3. The current [train.py](train.py) script is only the first-stage training entry point and does not represent the final full training framework.
4. Weight fusion remains experimental and requires dedicated validation for category consistency, output behavior, and stability.
5. If category order is modified at any stage, class indices must be verified explicitly. Under the current assumption, the custom dataset uses `robot` as class index `2`.
6. The output order of ONNX-exported models must remain consistent with the post-processing scripts; otherwise, class or mask parsing errors may occur.
7. If legacy scripts are migrated into the current repository layout, special attention should be paid to historical absolute-path configurations. All paths in this document are relative to the current repository root.

## Scope

This README covers only the following aspects:

- training methodology;
- design background;
- repository organization;
- intermediate fusion and export workflow;
- current experimental process.

The following topics are intentionally excluded and should be maintained in a separate project or separate documentation:

- deployment workflows;
- runtime integration;
- TensorRT-specific conversion and execution pipelines;
- production inference services.
