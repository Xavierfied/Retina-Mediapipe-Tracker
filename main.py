from pathlib import Path

from utils.args import get_args
from utils import face, hands, pose, retina

RUNNERS = {
    "face":   lambda args, src, out: face.run(src, out),
    "hands":  lambda args, src, out: hands.run(src, out),
    "pose":   lambda args, src, out: pose.run(src, out),
    "retina": lambda args, src, out: retina.run(args, src, out),
}


def main():
    args = get_args()
    source = int(args.source) if args.source.isdigit() else Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    RUNNERS[args.detector](args, source, output_dir)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
