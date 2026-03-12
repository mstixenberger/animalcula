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
    run_parser.add_argument("--seed-demo", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = Config.from_yaml(args.config)
        world = World(config=config, seed=args.seed)
        if args.seed_demo:
            world.seed_demo_archetypes()
        world.step(args.ticks)
        stats = world.stats()
        print(
            " ".join(
                [
                    f"tick={stats.tick}",
                    f"seed={world.seed}",
                    f"population={stats.population}",
                    f"nodes={stats.node_count}",
                    f"total_energy={stats.total_energy:.3f}",
                ]
            )
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
