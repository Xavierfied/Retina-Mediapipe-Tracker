# MediaPipe + RetinaFace Detection

A unified detection system focused on MediaPipe landmark models and RetinaFace face detection.

## Features

- **Face** — face landmark detection via MediaPipe Face Landmarker
- **Hands** — hand landmark detection via MediaPipe
- **Pose** — pose landmark detection via MediaPipe
- **RetinaFace** — face detection and landmarks via RetinaFace
- Supports images, video files, and webcam input
- Results saved to `results/` as `{filename}_{detector}.{ext}`

## Setup

1. **Create and activate a virtual environment**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Face landmark detection
python main.py --source image.jpg --detector face

# RetinaFace detection
python main.py --source image.jpg --detector retina

# Hand tracking
python main.py --source video.mp4 --detector hands

# Pose estimation
python main.py --source 0 --detector pose
```

## Parameters

```
 Parameter     Default    Description 

 `--source`   `Required`    Image/video path or webcam index (e.g. `0`) 
 `--detector`  `face`     Detection mode: `face` | `hands` | `pose` | `retina` 
 `--output`   `results`   Output directory 
 `--threshold`  `0.3`      RetinaFace score threshold 
```

## Project Structure

```
main.py              # entry point — parses args and dispatches to the right runner
utils/
  args.py            # argument definitions
  face.py            # face landmark runner
  hands.py           # hand tracking runner
  pose.py            # pose estimation runner
  retina.py          # RetinaFace face detection runner
weights/
  face_landmarker.task
  hand_landmarker.task
  pose_landmarker.task
```

## Output

Results are saved in the `results/` directory as `{filename}_{detector}.{ext}`, e.g.:
- `image_face.jpg`
- `video_hands.mp4`
- `webcam_pose.mp4`
- `image_retina.jpg`

MediaPipe `.task` model files are stored in `weights/` and auto-downloaded on first use.

## NOTE
RetinaFace is only for performing processing on images and video and not live feed.


# Future Update:
In the next update the current project shall be moved to sub-branch and in the "main" branch will contain a training script for you to input your own people labled dataset and effectively use it as an attendance system and, for Pose Detection variant, it will be combined with face detection to track a person with their name until they are out of the frame :)

