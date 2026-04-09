import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="MediaPipe + RetinaFace Detection Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source image.jpg --detector face
  python main.py --source video.mp4 --detector hands
  python main.py --source 0        --detector pose
  python main.py --source image.jpg --detector retina
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image/video path or webcam index (e.g. 0).",
    )

    parser.add_argument(
        "--detector",
        type=str,
        choices=["face", "hands", "pose", "retina"],
        default="face",
        help="Detection mode: face | hands | pose | retina (default: face)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results/)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Score threshold for RetinaFace detection (default: 0.3)",
    )

    return parser.parse_args()
