from pathlib import Path

import cv2 as cv
import gdown
from ultralytics import YOLO

from utils.args import get_args
from utils.predict import draw_bb

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
GOOGLE_DRIVE_FILE_ID = "1DOUGmdmvsV9c3cO2oqCgQ2kcw-3sFFqB"


# Model Loading
def download_model(file_id: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        return output_path
    
    print(f"[↓] Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_path), quiet=False)
    return output_path


# Drawing of bounding boxes on each frame
def process_media(model, source, output_dir, args):
    is_image = isinstance(source, Path) and source.suffix.lower() not in VIDEO_EXTS
    is_webcam = isinstance(source, int)
    
    predict = lambda frame: model.predict(
        frame, conf=args.confidence, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]

    if is_image:
        frame = cv.imread(str(source))
        frame = draw_bb(frame, predict(str(source)))
        cv.imwrite(str(output_dir / f"{source.stem}_ppe{source.suffix}"), frame)
    
    # Video processing module - no display
    elif not is_webcam:
        cap = cv.VideoCapture(str(source))
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv.VideoWriter(str(output_dir / f"{source.stem}_ppe.mp4"),
                                cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = draw_bb(frame, predict(frame))
            writer.write(frame)
        
        cap.release()
        writer.release()
    
    # Webcam processing module - with real-time display
    else:
        cap = cv.VideoCapture(source)
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv.VideoWriter(str(output_dir / "webcam_ppe.mp4"),
                                cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = draw_bb(frame, predict(frame))
            writer.write(frame)
            cv.imshow("Live PPE Detection — Press Q to quit", frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        writer.release()
        cv.destroyAllWindows()

    print("[✓] Done.")


def main():
    args = get_args()
    model_path = Path(args.model)
    
    # Auto-download default model if missing
    if not model_path.exists() and args.model == "best.pt":
        model_path = download_model(GOOGLE_DRIVE_FILE_ID, Path("weights") / "best.pt")
    
    model = YOLO(str(model_path))
    source = int(args.source) if args.source.isdigit() else Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_media(model, source, output_dir, args)


if __name__ == "__main__":
    main()