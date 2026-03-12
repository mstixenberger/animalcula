"""CLI entrypoints for the project."""

from __future__ import annotations

import argparse

from animalcula.config import Config
from animalcula.sim.world import World


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="animalcula")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a headless simulation")
    run_parser.add_argument("--config", default="config/default.yaml")
    run_parser.add_argument("--ticks", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=None)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = Config.from_yaml(args.config)
        world = World(config=config, seed=args.seed)
        world.step(args.ticks)
        print(f"tick={world.tick} seed={world.seed}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
