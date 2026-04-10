from pathlib import Path

import cv2 as cv
from retinaface import RetinaFace
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def _annotate(frame, resp):
    if resp is None:
        return frame

    for face_id, data in resp.items():
        x1, y1, x2, y2 = data["facial_area"]
        score = data.get("score", 0.0)
        landmarks = data.get("landmarks", {})

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(
            frame,
            f"{score:.2f}",
            (x1, max(y1 - 8, 0)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for lm in landmarks.values():
            cv.circle(frame, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)

    return frame


def run(args, source, output_dir: Path):
    threshold = getattr(args, "threshold", 0.3)

    is_image = isinstance(source, Path) and source.suffix.lower() not in VIDEO_EXTS

    if is_image:
        out_path = output_dir / f"{source.stem}_retina{source.suffix}"
        image = cv.imread(str(source))
        resp = RetinaFace.detect_faces(image, threshold=threshold)
        cv.imwrite(str(out_path), _annotate(image, resp))
        return

    # vid/webcam
    cap = cv.VideoCapture(source if isinstance(source, int) else str(source))
    is_webcam = isinstance(source, int)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Cannot read from source: {source}")

    h, w = frame.shape[:2]
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    out_path = output_dir / ("webcam_retina.mp4" if is_webcam else f"{source.stem}_retina.mp4")
    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    try:
        while ret:
            resp = RetinaFace.detect_faces(frame, threshold=threshold)
            frame = _annotate(frame, resp)
            writer.write(frame)

            if is_webcam:
                cv.imshow("RetinaFace Detection — Q to quit", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            ret, frame = cap.read()
    finally:
        writer.release()
        cap.release()
        cv.destroyAllWindows()