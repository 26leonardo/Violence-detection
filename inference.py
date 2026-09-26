"""Inference entry point: classify a single video file with a trained checkpoint.

Example usage:
    python inference.py --checkpoint checkpoints/lightweight_tsm/best.pt --video /path/to/clip.avi
"""

import argparse
import json

from inference.predict import ViolenceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run violence detection on a video file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to classify.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on the violence probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = ViolenceDetector(args.checkpoint)
    result = detector.predict(args.video, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
