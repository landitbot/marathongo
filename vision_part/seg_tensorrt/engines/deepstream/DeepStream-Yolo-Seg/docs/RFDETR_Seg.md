# RF-DETR-Seg usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_rfdetr_seg file](#edit-the-config_infer_primary_rfdetr_seg-file)

##

### Convert model

#### 1. Download the RF-DETR repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_rfdetr_seg.py` file from `DeepStream-Yolo-Seg/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [RF-DETR](https://github.com/roboflow/rf-detr) releases (example for RF-DETR-Seg-Preview)

```
wget https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for RF-DETR-Seg-Preview)

```
python3 export_rfdetr_seg.py -w rf-detr-seg-preview.pt --dynamic
```

**NOTE**: Minimum detection confidence threshold (example for conf-threshold = 0.25)

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
--conf-threshold 0.25
```

**NOTE**: NMS IoU threshold (example for iou-threshold = 0.45)

```
--iou-threshold 0.45
```

**NOTE**: Maximum number of output detections (example for max-detections = 300)

```
--max-detections 300
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo-Seg` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo-Seg` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 = 10.2
  ```

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg
```

##

### Edit the config_infer_primary_rfdetr_seg file

Edit the `config_infer_primary_rfdetr_seg.txt` file according to your model (example for RF-DETR-Seg-Preview)

```
[property]
...
onnx-file=rf-detr-seg-preview.onnx
...
num-detected-classes=91
...
parse-bbox-func-name=NvDsInferParseYoloSeg
...
```

**NOTE**: To output the masks, use

```
[property]
...
output-instance-mask=1
segmentation-threshold=0.5
...
```

**NOTE**: The **RF-DETR-Seg** do not resize the input with padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=0
...
```
