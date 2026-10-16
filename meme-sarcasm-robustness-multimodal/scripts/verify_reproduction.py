"""Exit nonzero if independent training or checkpoint reload fails comparison."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.reproduction import compare_metrics, compare_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_metrics", required=True)
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--reference_predictions")
    parser.add_argument("--candidate_predictions")
    args = parser.parse_args()
    if bool(args.reference_predictions) != bool(args.candidate_predictions):
        parser.error("Both prediction paths are required for an exact reload check")
    candidate = json.loads(Path(args.candidate_metrics).read_text(encoding="utf-8"))
    passed = compare_metrics(args.reference_metrics, candidate, args.tolerance)
    if args.reference_predictions:
        passed = compare_predictions(args.reference_predictions, args.candidate_predictions) and passed
    print("REPRODUCTION OK" if passed else "REPRODUCTION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
