from unittest.mock import DEFAULT

import cv2 as cv
import numpy as np

CLASS_COLORS = {
    1: ((83,  200,   0), "helmet"),   # helmet - green
    0: ((68,   23, 255), "head"),   # head   - red
    2: ((0,   214, 255), "person"),   # person - yellow
}

DEFAULT_CLASS_COLOR = (255, 255, 255)

def get_box_stats(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf  = float(box.conf[0])
    cls   = int(box.cls[0])
    color, label = CLASS_COLORS.get(cls, ((255, 255, 255), "unknown"))
    return x1, y1, x2, y2, conf, cls, label, color


def draw_bb(frame, result):
    for box in result.boxes:
        x1, y1, x2, y2, conf, cls, label, color = get_box_stats(box)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv.putText(frame, text, (x1, y1 - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame