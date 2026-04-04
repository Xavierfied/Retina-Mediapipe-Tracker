import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="PPE DETECTION SAFTEY IDENTIFIER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
    )

    parser.add_argument(
    "--source",
    type=str,
    required=True,
    help="Type image/video path or webcam index '0'.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to YOLO weights (default:best.pt)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confidence threshold for detections (default:0.35)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU(Intersection over union) threshold for bounding box detections (default:0.45)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Img size boxes (default:640)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output folder (default: results/)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' auto | 'cpu' | '0' GPU"
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box thickness (default: 2)"
    )

    return parser.parse_args()
