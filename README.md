# Safety Identifier YOLO

A YOLO-based safety detection system that identifies personal protective equipment (PPE) in images and videos.

## Features

- Detects helmets and heads images/videos
- Optional GPU acceleration for faster processing
- Support for custom YOLO models
- Automatically downloads the default model from Google Drive if missing
- Supports images, video files, and webcam input
- Saves processed results to output directory

## Setup For PREDICTION

1. **Create and activate virtual environment**
   ```bash
   create your own virtual enviorment 
   ```

2. **Install dependencies**
   ```bash
   pip install requirements.txt requirements.txt.txt
   ```

3. **Install GPU PyTorch (Optional but recommended)**
   ```bash
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Guide to use:

### Argparse Options:
```bash
# Process an image
python main.py --source path/to/image.jpg

# Process a video
python main.py --source path/to/video.mp4

# Use webcam (0 is default camera)
python main.py --source 0
```

### Advanced Options
```bash
python main.py --source image.jpg \
    --model path/to/model.pt \
    --confidence 0.5 \
    --output results/ \
    --imgsz 640
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | Required | Image/video path or webcam index (0, 1, etc.) |
| `--model` | best.pt | Path to YOLO model weights |
| `--confidence` | 0.35 | Confidence threshold for detections |
| `--iou` | 0.45 | IoU threshold for bounding boxes |
| `--imgsz` | 640 | Input image size |
| `--output` | results | Output directory for processed files |
| `--device` | auto | Device: 'cpu' or GPU index (0, 1, etc.) |

## Detection Classes

- **Helmet (Green)**: Safety helmets
- **Head (Red)**: Exposed heads

## Output
```
Output will be saved in results directior, as "filename_PPE"
```
## Model

The default model (`best.pt`) is automatically downloaded from Google Drive on first run. You can also use your own custom YOLO models by specifying the `--model` parameter.

## Requirements
Use:
```
cd "Webevis-internship/saftey_identifier_yolo"

pip install -r requirements.txt
```


## NOTE
To review video based results, check out the before and after video under this link:
```
https://drive.google.com/drive/folders/1nElnO3NpckkNUBCmyAkScuWQDxpLz6zW?usp=sharing
```
---

## SETUP FOR TRAINING
Upload the notebook in Google colab or kaggle according to your preference and simply run all the cells in order with your **"WandB"** API KEY.

---

## Stats of best.pt
```
Trained till 94 Epocs due to Early Stopping:

- mAP@50        : 0.636
- mAP@50-95     : 0.429
- Precision     : 0.611
- Recall        : 0.610

```