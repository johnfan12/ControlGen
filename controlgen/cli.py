from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlgen.generator import DatasetConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ControlGen dataset samples")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSONL samples",
    )
    args = parser.parse_args()

    config = DatasetConfig(count=args.count, seed=args.seed)
    samples = generate_dataset(config)

    if args.output is None:
        for sample in samples:
            print(json.dumps(sample.to_dict()))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict()) + "\n")


if __name__ == "__main__":
    main()
