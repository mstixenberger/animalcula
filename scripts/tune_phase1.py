"""Run a coarse multi-seed tuning batch with live progress and aggregate summaries."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from animalcula.analysis.seedbank import promote_seed_bank
from animalcula.analysis.sweep import (
    _iter_combinations,
    _load_sweep,
    _run_sweep_combination,
    aggregate_sweep_records,
)
from animalcula.config import Config
from animalcula.sim.world import World


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tune_phase1")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sweep", default="config/sweeps/phase1_default_economy.yaml")
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-top", default=None)
    parser.add_argument("--top-runs", type=int, default=3)
    parser.add_argument("--top-creatures", type=int, default=5)
    parser.add_argument("--promote-out-dir", default=None)
    parser.add_argument("--promote-rounds", type=int, default=3)
    parser.add_argument("--no-seed-demo", action="store_false", dest="seed_demo")
    parser.set_defaults(seed_demo=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_config = Config.from_yaml(args.config)
    sweep = _load_sweep(args.sweep)
    combinations = list(_iter_combinations(sweep))
    seeds = [int(value) for value in args.seeds.split(",") if value]
    records: list[dict[str, object]] = []

    run_count = len(combinations) * len(seeds)
    jobs = [
        (
            run_index,
            seed,
            overrides,
            base_config.to_dict(),
            args.ticks,
            args.seed_demo,
            args.turbo,
        )
        for run_index, (overrides, seed) in enumerate(
            [(overrides, seed) for overrides in combinations for seed in seeds],
            start=1,
        )
    ]

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _run_phase1_case,
                run_index,
                run_count,
                seed,
                overrides,
                base_config_dict,
                ticks,
                seed_demo,
                turbo,
            ): run_index
            for run_index, seed, overrides, base_config_dict, ticks, seed_demo, turbo in jobs
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)

    records.sort(key=lambda record: record["run"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(aggregate_sweep_records(records), indent=2), encoding="utf-8")

    seed_bank_path: Path | None = None
    seed_bank_manifest_path: Path | None = None
    promotion_manifest_path: str | None = None
    if args.save_top is not None or args.promote_out_dir is not None:
        seed_bank_path = Path(args.save_top) if args.save_top is not None else out_path.with_suffix(".seedbank.json")
        seed_bank_manifest_path = seed_bank_path.with_suffix(".manifest.json")
        _save_top_seed_bank(
            records=records,
            config_path=args.config,
            ticks=args.ticks,
            seed_demo=args.seed_demo,
            turbo=args.turbo,
            top_runs=max(0, args.top_runs),
            top_creatures=max(0, args.top_creatures),
            out_path=seed_bank_path,
            manifest_path=seed_bank_manifest_path,
        )
    if args.promote_out_dir is not None and seed_bank_path is not None:
        promotion_manifest = promote_seed_bank(
            config_path=args.config,
            genomes_path=str(seed_bank_path),
            ticks=args.ticks,
            seeds=seeds,
            turbo=args.turbo,
            rounds=max(0, args.promote_rounds),
            top=max(0, args.top_creatures),
            out_dir=args.promote_out_dir,
            workers=max(1, args.workers),
        )
        promotion_manifest_path = promotion_manifest["manifest_path"]

    saved_parts = [f"saved={out_path}", f"summary={summary_path}"]
    if seed_bank_path is not None:
        saved_parts.append(f"seed_bank={seed_bank_path}")
    if seed_bank_manifest_path is not None:
        saved_parts.append(f"seed_bank_manifest={seed_bank_manifest_path}")
    if promotion_manifest_path is not None:
        saved_parts.append(f"promotion={promotion_manifest_path}")
    print(" ".join(saved_parts), flush=True)
    return 0


def _run_phase1_case(
    run_index: int,
    run_count: int,
    seed: int,
    overrides: dict[str, object],
    base_config_dict: dict[str, object],
    ticks: int,
    seed_demo: bool,
    turbo: bool,
) -> dict[str, object]:
    _, record = _run_sweep_combination(
        index=run_index,
        overrides=overrides,
        base_config_dict=base_config_dict,
        ticks=ticks,
        seed=seed,
        seed_demo=seed_demo,
        turbo=turbo,
    )
    record["run"] = run_index
    record["run_count"] = run_count
    record["seed"] = seed
    return record


def _save_top_seed_bank(
    *,
    records: list[dict[str, object]],
    config_path: str,
    ticks: int,
    seed_demo: bool,
    turbo: bool,
    top_runs: int,
    top_creatures: int,
    out_path: Path,
    manifest_path: Path,
) -> None:
    ranked_records = sorted(
        records,
        key=lambda record: (
            record["ended_extinct"],
            -float(record["interestingness"]),
            -float(record["species_turnover"]),
            -float(record["observed_species_count"]),
            -float(record["population"]),
            int(record["run"]),
        ),
    )[:top_runs]
    base_config = Config.from_yaml(config_path)
    exported_payload: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []

    for rank, record in enumerate(ranked_records, start=1):
        world = _replay_phase1_world(
            config=base_config,
            overrides=dict(record["overrides"]),
            seed=int(record["seed"]),
            ticks=ticks,
            seed_demo=seed_demo,
            turbo=turbo,
        )
        top_payload = world.top_creature_payload(n=top_creatures, metric="energy")
        for creature_index, creature in enumerate(top_payload, start=1):
            creature["source_run"] = int(record["run"])
            creature["source_seed"] = int(record["seed"])
            creature["source_rank"] = rank
            creature["source_creature_rank"] = creature_index
            creature["source_overrides"] = dict(record["overrides"])
            exported_payload.append(creature)
        manifest_runs.append(
            {
                "rank": rank,
                "run": int(record["run"]),
                "seed": int(record["seed"]),
                "overrides": dict(record["overrides"]),
                "interestingness": record["interestingness"],
                "species_turnover": record["species_turnover"],
                "observed_species_count": record["observed_species_count"],
                "population": record["population"],
                "exported_creatures": len(top_payload),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(exported_payload, indent=2), encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "config": config_path,
                "ticks": ticks,
                "seed_demo": seed_demo,
                "turbo": turbo,
                "top_runs_requested": top_runs,
                "top_creatures_per_run": top_creatures,
                "selected_run_count": len(ranked_records),
                "exported_creature_count": len(exported_payload),
                "seed_bank_path": str(out_path),
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _replay_phase1_world(
    *,
    config: Config,
    overrides: dict[str, object],
    seed: int,
    ticks: int,
    seed_demo: bool,
    turbo: bool,
) -> World:
    world_config = config.with_overrides([f"{key}={value}" for key, value in overrides.items()])
    world = World(config=world_config, seed=seed, turbo=turbo)
    if seed_demo:
        world.seed_demo_archetypes()
    world.step(ticks)
    return world


if __name__ == "__main__":
    raise SystemExit(main())
