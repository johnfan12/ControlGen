from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlgen.generator import DatasetConfig, GrammarConfig, generate_dataset
from controlgen.simulate import SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graph-based MIMO ControlGen dataset samples")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("--min-nodes", type=int, default=8, help="Minimum number of graph nodes")
    parser.add_argument("--max-nodes", type=int, default=24, help="Maximum number of graph nodes")
    parser.add_argument("--min-io", type=int, default=3, help="Minimum input/output channel count")
    parser.add_argument("--max-io", type=int, default=5, help="Maximum input/output channel count")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSONL samples",
    )
    args = parser.parse_args()

    config = DatasetConfig(
        count=args.count,
        seed=args.seed,
        sim_config=SimulationConfig(duration=args.duration, dt=args.dt),
    )
    grammar = GrammarConfig(
        graph_size_range=(args.min_nodes, args.max_nodes),
        io_channels_range=(args.min_io, args.max_io),
    )
    samples = generate_dataset(config, grammar_config=grammar)

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
