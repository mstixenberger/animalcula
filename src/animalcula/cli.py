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
    run_parser.add_argument("--save", default=None)
    run_parser.add_argument("--resume", default=None)
    run_parser.add_argument("--set", action="append", default=[])

    report_parser = subparsers.add_parser("report", help="Report summary stats from a checkpoint")
    report_parser.add_argument("checkpoint")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        if args.resume is not None:
            world = World.load(args.resume)
            if args.set:
                world.config = world.config.with_overrides(args.set)
        else:
            config = Config.from_yaml(args.config)
            if args.set:
                config = config.with_overrides(args.set)
            world = World(config=config, seed=args.seed)
        if args.seed_demo and args.resume is None:
            world.seed_demo_archetypes()
        world.step(args.ticks)
        if args.save is not None:
            world.save(args.save)
        stats = world.stats()
        print(_format_stats(world.seed, stats))
        return 0

    if args.command == "report":
        world = World.load(args.checkpoint)
        print(_format_stats(world.seed, world.stats()))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _format_stats(seed: int, stats: object) -> str:
    return " ".join(
        [
            f"tick={stats.tick}",
            f"seed={seed}",
            f"population={stats.population}",
            f"nodes={stats.node_count}",
            f"total_energy={stats.total_energy:.3f}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
