"""CLI entrypoints for the project."""

from __future__ import annotations

import argparse
import json

from animalcula.analysis.sweep import run_sweep
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

    events_parser = subparsers.add_parser("events", help="Print checkpoint events as JSON lines")
    events_parser.add_argument("checkpoint")

    sweep_parser = subparsers.add_parser("sweep", help="Run a sequential parameter sweep")
    sweep_parser.add_argument("--config", default="config/default.yaml")
    sweep_parser.add_argument("--sweep", required=True)
    sweep_parser.add_argument("--ticks", type=int, default=1)
    sweep_parser.add_argument("--seed", type=int, default=None)
    sweep_parser.add_argument("--seed-demo", action="store_true")
    sweep_parser.add_argument("--out", required=True)

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

    if args.command == "events":
        world = World.load(args.checkpoint)
        for event in world.events:
            print(
                json.dumps(
                    {
                        "tick": event.tick,
                        "event_type": event.event_type,
                        "creature_id": event.creature_id,
                        "parent_ids": list(event.parent_ids),
                        "energy": event.energy,
                    }
                )
            )
        return 0

    if args.command == "sweep":
        completed = run_sweep(
            config_path=args.config,
            sweep_path=args.sweep,
            ticks=args.ticks,
            seed=args.seed,
            seed_demo=args.seed_demo,
            out_path=args.out,
        )
        print(f"completed={completed} out={args.out}")
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
            f"births={stats.births}",
            f"deaths={stats.deaths}",
            f"reproductions={stats.reproductions}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
