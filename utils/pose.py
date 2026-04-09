import time
import urllib.request
from pathlib import Path

import cv2 as cv
import mediapipe as mp

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
MODEL_PATH = Path("weights/pose_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

BaseOptions         = mp.tasks.BaseOptions
PoseLandmarker      = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode


def _download_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[↓] Downloading pose_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    print(f"[✓] Saved to {MODEL_PATH}")


def _annotate(frame, results):
    h, w = frame.shape[:2]
    if not getattr(results, "pose_landmarks", None):
        return frame
    for pose in results.pose_landmarks:
        for lm in pose:
            cv.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
    return frame


def run(source, output_dir: Path):
    _download_model()

    is_image  = isinstance(source, Path) and source.suffix.lower() not in VIDEO_EXTS
    is_webcam = isinstance(source, int)

    if is_image:
        out_path = output_dir / f"{source.stem}_pose{source.suffix}"
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
        )
        image  = cv.imread(str(source))
        rgb    = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with PoseLandmarker.create_from_options(opts) as det:
            results = det.detect(mp_img)
        cv.imwrite(str(out_path), _annotate(image, results))

    elif not is_webcam:
        out_path = output_dir / f"{source.stem}_pose.mp4"
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
        )
        cap = cv.VideoCapture(str(source))
        fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Cannot read video: {source}")
        h, w = frame.shape[:2]
        writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        with PoseLandmarker.create_from_options(opts) as det:
            ts_ms = 0
            while ret:
                rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                writer.write(_annotate(frame, det.detect_for_video(mp_img, ts_ms)))
                ts_ms += int(1000.0 / fps)
                ret, frame = cap.read()
        writer.release()
        cap.release()

    else:
        out_path = output_dir / "webcam_pose.mp4"
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
        )
        cap    = cv.VideoCapture(source)
        fps    = cap.get(cv.CAP_PROP_FPS) or 30.0
        w      = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        start  = time.time()
        with PoseLandmarker.create_from_options(opts) as det:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts_ms  = int((time.time() - start) * 1000)
                rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame  = _annotate(frame, det.detect_for_video(mp_img, ts_ms))
                writer.write(frame)
                cv.imshow("Pose Detection — Q to quit", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        writer.release()
        cap.release()
        cv.destroyAllWindows()
