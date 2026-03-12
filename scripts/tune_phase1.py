"""Run a coarse multi-seed tuning batch with live progress and aggregate summaries."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from animalcula.analysis.sweep import (
    _iter_combinations,
    _load_sweep,
    _run_sweep_combination,
    aggregate_sweep_records,
)
from animalcula.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tune_phase1")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sweep", default="config/sweeps/phase1_default_economy.yaml")
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--out", required=True)
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
    print(f"saved={out_path} summary={summary_path}", flush=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
