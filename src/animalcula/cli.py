"""CLI entrypoints for the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from animalcula.analysis.metrics import trophic_balance_score
from animalcula.analysis.seedbank import evaluate_seed_bank, promote_seed_bank
from animalcula.analysis.sweep import run_sweep
from animalcula.config import Config
from animalcula.sim.world import World
from animalcula.viz.debug_viewer import launch_viewer


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
    run_parser.add_argument("--seed-from", default=None)
    run_parser.add_argument("--set", action="append", default=[])
    run_parser.add_argument("--log-stats", default=None)
    run_parser.add_argument("--log-stats-sqlite", default=None)
    run_parser.add_argument("--log-every", type=int, default=1)
    run_parser.add_argument("--turbo", action="store_true")

    view_parser = subparsers.add_parser(
        "view",
        help="Open the minimal local debug viewer",
        description="Open the minimal local debug viewer. Falls back to a self-contained HTML viewer when Tk is unavailable.",
    )
    view_parser.add_argument("--config", default="config/default.yaml")
    view_parser.add_argument("--seed", type=int, default=None)
    view_parser.add_argument("--seed-demo", action="store_true")
    view_parser.add_argument("--resume", default=None)
    view_parser.add_argument("--seed-from", default=None)
    view_parser.add_argument("--set", action="append", default=[])
    view_parser.add_argument("--turbo", action="store_true")
    view_parser.add_argument("--warmup-ticks", type=int, default=120)
    view_parser.add_argument("--steps-per-frame", type=int, default=4)
    view_parser.add_argument("--frame-delay-ms", type=int, default=33)
    view_parser.add_argument("--canvas-width", type=int, default=900)
    view_parser.add_argument("--canvas-height", type=int, default=900)
    view_parser.add_argument("--viewer-backend", choices=("auto", "tk", "html"), default="auto")
    view_parser.add_argument("--html-out", default=None)
    view_parser.add_argument("--max-frames", type=int, default=600)

    report_parser = subparsers.add_parser("report", help="Report summary stats from a checkpoint")
    report_parser.add_argument("checkpoint")

    events_parser = subparsers.add_parser("events", help="Print checkpoint events as JSON lines")
    events_parser.add_argument("checkpoint")

    phylogeny_parser = subparsers.add_parser("phylogeny", help="Export checkpoint phylogeny")
    phylogeny_parser.add_argument("checkpoint")
    phylogeny_parser.add_argument("--format", choices=("json", "newick"), default="json")

    species_parser = subparsers.add_parser("species", help="Print checkpoint species snapshots as JSON lines")
    species_parser.add_argument("checkpoint")

    phenotypes_parser = subparsers.add_parser("phenotypes", help="Print checkpoint phenotype snapshots as JSON lines")
    phenotypes_parser.add_argument("checkpoint")

    phenotype_vectors_parser = subparsers.add_parser(
        "phenotype-vectors",
        help="Print checkpoint phenotype vectors as JSON lines",
    )
    phenotype_vectors_parser.add_argument("checkpoint")

    extract_parser = subparsers.add_parser("extract-genomes", help="Export top creatures from a checkpoint")
    extract_parser.add_argument("checkpoint")
    extract_parser.add_argument("--top", type=int, default=10)
    extract_parser.add_argument("--out", required=True)

    evaluate_parser = subparsers.add_parser("evaluate-genomes", help="Evaluate exported genomes across fresh runs")
    evaluate_parser.add_argument("genomes")
    evaluate_parser.add_argument("--config", default="config/default.yaml")
    evaluate_parser.add_argument("--ticks", type=int, default=100)
    evaluate_parser.add_argument("--seeds", default="41,42,43")
    evaluate_parser.add_argument("--workers", type=int, default=1)
    evaluate_parser.add_argument("--turbo", action="store_true")
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--save-top", default=None)
    evaluate_parser.add_argument("--top", type=int, default=5)

    promote_parser = subparsers.add_parser("promote-genomes", help="Run multi-round seed-bank promotion")
    promote_parser.add_argument("genomes")
    promote_parser.add_argument("--config", default="config/default.yaml")
    promote_parser.add_argument("--ticks", type=int, default=100)
    promote_parser.add_argument("--seeds", default="41,42,43")
    promote_parser.add_argument("--workers", type=int, default=1)
    promote_parser.add_argument("--turbo", action="store_true")
    promote_parser.add_argument("--rounds", type=int, default=3)
    promote_parser.add_argument("--top", type=int, default=5)
    promote_parser.add_argument("--out-dir", required=True)

    sweep_parser = subparsers.add_parser("sweep", help="Run a parameter sweep")
    sweep_parser.add_argument("--config", default="config/default.yaml")
    sweep_parser.add_argument("--sweep", required=True)
    sweep_parser.add_argument("--ticks", type=int, default=1)
    sweep_parser.add_argument("--seed", type=int, default=None)
    sweep_parser.add_argument("--seed-demo", action="store_true")
    sweep_parser.add_argument("--out", required=True)
    sweep_parser.add_argument("--turbo", action="store_true")
    sweep_parser.add_argument("--workers", type=int, default=1)

    nursery_parser = subparsers.add_parser("nursery", help="Run a seeded nursery simulation")
    nursery_parser.add_argument("--config", default="config/nursery.yaml")
    nursery_parser.add_argument("--ticks", type=int, default=100)
    nursery_parser.add_argument("--seed", type=int, default=None)
    nursery_parser.add_argument("--top", type=int, default=5)
    nursery_parser.add_argument("--save-top", default=None)
    nursery_parser.add_argument("--out", required=True)
    nursery_parser.add_argument("--turbo", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        world = _load_or_create_world(args)
        if args.log_stats is not None or args.log_stats_sqlite is not None:
            _run_with_stats_log(
                world=world,
                ticks=args.ticks,
                log_path=args.log_stats,
                sqlite_path=args.log_stats_sqlite,
                log_every=args.log_every,
            )
        else:
            world.step(args.ticks)
        if args.save is not None:
            world.save(args.save)
        stats = world.stats()
        print(_format_stats(world.seed, stats))
        return 0

    if args.command == "view":
        world = _load_or_create_world(args)
        if args.resume is None and args.warmup_ticks > 0:
            _warmup_world_with_progress(world, args.warmup_ticks)
        html_viewer = launch_viewer(
            world,
            steps_per_frame=max(1, args.steps_per_frame),
            frame_delay_ms=max(1, args.frame_delay_ms),
            canvas_width=max(200, args.canvas_width),
            canvas_height=max(200, args.canvas_height),
            backend=args.viewer_backend,
            html_out_path=args.html_out,
            max_frames=max(1, args.max_frames),
        )
        if html_viewer is not None:
            print(f"saved_html_viewer={html_viewer}")
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
                        "genome_hash": event.genome_hash,
                        "color_rgb": list(event.color_rgb),
                    }
                )
            )
        return 0

    if args.command == "phylogeny":
        world = World.load(args.checkpoint)
        if args.format == "newick":
            print(world.phylogeny_newick())
        else:
            print(json.dumps(world.get_phylogeny()))
        return 0

    if args.command == "species":
        world = World.load(args.checkpoint)
        for snapshot in world.species_snapshots():
            print(json.dumps(snapshot))
        return 0

    if args.command == "phenotypes":
        world = World.load(args.checkpoint)
        for snapshot in world.phenotype_snapshots():
            print(json.dumps(snapshot))
        return 0

    if args.command == "phenotype-vectors":
        world = World.load(args.checkpoint)
        for vector in world.phenotype_vectors():
            print(json.dumps(vector))
        return 0

    if args.command == "extract-genomes":
        world = World.load(args.checkpoint)
        world.export_top_creatures(path=args.out, n=args.top, metric="energy")
        print(f"saved={args.out} top={args.top}")
        return 0

    if args.command == "evaluate-genomes":
        report = evaluate_seed_bank(
            config_path=args.config,
            genomes_path=args.genomes,
            ticks=args.ticks,
            seeds=[int(value) for value in args.seeds.split(",") if value],
            turbo=args.turbo,
            workers=args.workers,
            out_path=args.out,
            save_top_path=args.save_top,
            top=args.top,
        )
        promoted = f" promoted={args.save_top}" if args.save_top is not None else ""
        print(f"saved={args.out} evaluated={report['candidate_count']}{promoted}")
        return 0

    if args.command == "promote-genomes":
        manifest = promote_seed_bank(
            config_path=args.config,
            genomes_path=args.genomes,
            ticks=args.ticks,
            seeds=[int(value) for value in args.seeds.split(",") if value],
            turbo=args.turbo,
            rounds=args.rounds,
            top=args.top,
            out_dir=args.out_dir,
            workers=args.workers,
        )
        print(
            f"saved={manifest['manifest_path']} rounds={manifest['rounds_completed']} final={manifest['final_genomes_path']}"
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
            turbo=args.turbo,
            workers=args.workers,
        )
        print(f"completed={completed} out={args.out}")
        return 0

    if args.command == "nursery":
        config = Config.from_yaml(args.config)
        world = World(config=config, seed=args.seed, turbo=args.turbo)
        world.seed_demo_archetypes()
        world.step(args.ticks)
        world.save(args.out)
        if args.save_top is not None:
            world.export_top_creatures(path=args.save_top, n=args.top, metric="energy")
        top_creatures = world.get_top_creatures(n=args.top, metric="energy")
        top_summary = ",".join(f"{creature.id}:{creature.energy:.3f}" for creature in top_creatures)
        print(f"saved={args.out} top_creatures={top_summary}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _warmup_world_with_progress(world: World, ticks: int) -> None:
    if ticks <= 0:
        return
    if not sys.stderr.isatty():
        world.step(ticks)
        return
    chunk = max(1, min(20, ticks // 24 or 1))
    completed = 0
    while completed < ticks:
        step_ticks = min(chunk, ticks - completed)
        world.step(step_ticks)
        completed += step_ticks
        _print_progress_bar(label="warming viewer", completed=completed, total=ticks)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _print_progress_bar(*, label: str, completed: int, total: int, width: int = 28) -> None:
    safe_total = max(total, 1)
    ratio = max(0.0, min(1.0, completed / safe_total))
    filled = min(width, int(round(width * ratio)))
    bar = ("#" * filled) + ("-" * (width - filled))
    sys.stderr.write(f"\r{label} [{bar}] {completed}/{total}")
    sys.stderr.flush()


def _format_stats(seed: int, stats: object) -> str:
    return " ".join(
        [
            f"tick={stats.tick}",
            f"seed={seed}",
            f"drag_multiplier={stats.drag_multiplier:.2f}",
            f"nutrient_strength_multiplier={stats.nutrient_source_strength_multiplier:.2f}",
            f"light_intensity={stats.light_intensity:.2f}",
            f"light_direction_degrees={stats.light_direction_degrees:.1f}",
            f"population={stats.population}",
            f"peak_population={stats.peak_population}",
            f"population_variance={stats.population_variance:.3f}",
            f"population_capacity_fraction={stats.population_capacity_fraction:.3f}",
            f"peak_population_capacity_fraction={stats.peak_population_capacity_fraction:.3f}",
            f"crowding_multiplier={stats.crowding_multiplier:.3f}",
            f"peak_crowding_multiplier={stats.peak_crowding_multiplier:.3f}",
            f"nodes={stats.node_count}",
            f"total_energy={stats.total_energy:.3f}",
            f"mean_creature_energy={stats.mean_creature_energy:.3f}",
            f"max_creature_energy={stats.max_creature_energy:.3f}",
            f"nutrient_total={stats.nutrient_total:.3f}",
            f"detritus_total={stats.detritus_total:.3f}",
            f"chemical_a_total={stats.chemical_a_total:.3f}",
            f"chemical_b_total={stats.chemical_b_total:.3f}",
            f"births={stats.births}",
            f"deaths={stats.deaths}",
            f"reproductions={stats.reproductions}",
            f"speciations={stats.speciation_events}",
            f"species_extinctions={stats.species_extinctions}",
            f"species_turnover={stats.species_turnover}",
            f"predation_kills={stats.predation_kills}",
            f"environment_perturbations={stats.environment_perturbations}",
            f"species={stats.species_count}",
            f"observed_species={stats.observed_species_count}",
            f"peak_species={stats.peak_species_count}",
            f"peak_species_fraction={stats.peak_species_fraction:.3f}",
            f"lineages={stats.lineage_count}",
            f"runaway_dominance={'true' if stats.runaway_dominance_detected else 'false'}",
            f"diversity={stats.diversity_index:.3f}",
            f"complexity={stats.mean_nodes_per_creature:.2f}",
            f"mean_edges_per_creature={stats.mean_edges_per_creature:.2f}",
            f"mean_motor_edges_per_creature={stats.mean_motor_edges_per_creature:.2f}",
            f"mean_segment_length_per_creature={stats.mean_segment_length_per_creature:.2f}",
            f"mean_mouths_per_creature={stats.mean_mouths_per_creature:.2f}",
            f"mean_grippers_per_creature={stats.mean_grippers_per_creature:.2f}",
            f"mean_sensors_per_creature={stats.mean_sensors_per_creature:.2f}",
            f"mean_photoreceptors_per_creature={stats.mean_photoreceptors_per_creature:.2f}",
            f"mean_speed_recent={stats.mean_speed_recent:.3f}",
            f"mean_age_ticks={stats.mean_age_ticks:.2f}",
            f"max_age_ticks={stats.max_age_ticks}",
            f"active_grip_latches={stats.active_grip_latch_count}",
            f"peak_grip_latches={stats.peak_grip_latch_count}",
            f"mean_gripper_contact_signal={stats.mean_gripper_contact_signal:.3f}",
            f"mean_gripper_active_signal={stats.mean_gripper_active_signal:.3f}",
            f"longest_species_lifespan={stats.longest_species_lifespan}",
            f"mean_extinct_species_lifespan={stats.mean_extinct_species_lifespan:.2f}",
            f"autotrophs={stats.autotroph_count}",
            f"herbivores={stats.herbivore_count}",
            f"predators={stats.predator_count}",
            f"trophic_balance={trophic_balance_score(stats.autotroph_count, stats.herbivore_count, stats.predator_count):.3f}",
        ]
    )


def _stats_record(stats: object) -> dict[str, object]:
    return {
        "tick": stats.tick,
        "population": stats.population,
        "peak_population": stats.peak_population,
        "population_variance": stats.population_variance,
        "population_capacity_fraction": stats.population_capacity_fraction,
        "peak_population_capacity_fraction": stats.peak_population_capacity_fraction,
        "crowding_multiplier": stats.crowding_multiplier,
        "peak_crowding_multiplier": stats.peak_crowding_multiplier,
        "drag_multiplier": stats.drag_multiplier,
        "nutrient_source_strength_multiplier": stats.nutrient_source_strength_multiplier,
        "light_intensity": stats.light_intensity,
        "light_direction_degrees": stats.light_direction_degrees,
        "nodes": stats.node_count,
        "total_energy": stats.total_energy,
        "mean_creature_energy": stats.mean_creature_energy,
        "max_creature_energy": stats.max_creature_energy,
        "nutrient_total": stats.nutrient_total,
        "detritus_total": stats.detritus_total,
        "chemical_a_total": stats.chemical_a_total,
        "chemical_b_total": stats.chemical_b_total,
        "births": stats.births,
        "deaths": stats.deaths,
        "reproductions": stats.reproductions,
        "speciation_events": stats.speciation_events,
        "species_extinctions": stats.species_extinctions,
        "species_turnover": stats.species_turnover,
        "predation_kills": stats.predation_kills,
        "environment_perturbations": stats.environment_perturbations,
        "lineage_count": stats.lineage_count,
        "species_count": stats.species_count,
        "observed_species_count": stats.observed_species_count,
        "peak_species_count": stats.peak_species_count,
        "peak_species_fraction": stats.peak_species_fraction,
        "runaway_dominance_detected": stats.runaway_dominance_detected,
        "diversity_index": stats.diversity_index,
        "mean_nodes_per_creature": stats.mean_nodes_per_creature,
        "mean_edges_per_creature": stats.mean_edges_per_creature,
        "mean_motor_edges_per_creature": stats.mean_motor_edges_per_creature,
        "mean_segment_length_per_creature": stats.mean_segment_length_per_creature,
        "mean_mouths_per_creature": stats.mean_mouths_per_creature,
        "mean_grippers_per_creature": stats.mean_grippers_per_creature,
        "mean_sensors_per_creature": stats.mean_sensors_per_creature,
        "mean_photoreceptors_per_creature": stats.mean_photoreceptors_per_creature,
        "mean_speed_recent": stats.mean_speed_recent,
        "mean_age_ticks": stats.mean_age_ticks,
        "max_age_ticks": stats.max_age_ticks,
        "active_grip_latch_count": stats.active_grip_latch_count,
        "peak_grip_latch_count": stats.peak_grip_latch_count,
        "mean_gripper_contact_signal": stats.mean_gripper_contact_signal,
        "mean_gripper_active_signal": stats.mean_gripper_active_signal,
        "longest_species_lifespan": stats.longest_species_lifespan,
        "mean_extinct_species_lifespan": stats.mean_extinct_species_lifespan,
        "autotroph_count": stats.autotroph_count,
        "herbivore_count": stats.herbivore_count,
        "predator_count": stats.predator_count,
        "trophic_balance_score": trophic_balance_score(
            stats.autotroph_count,
            stats.herbivore_count,
            stats.predator_count,
        ),
    }


def _run_with_stats_log(
    world: World,
    ticks: int,
    log_path: str | None,
    sqlite_path: str | None,
    log_every: int,
) -> None:
    records: list[str] = []
    connection: sqlite3.Connection | None = None
    logged_event_count = 0
    if log_path is not None:
        output = Path(log_path)
        output.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path is not None:
        sqlite_output = Path(sqlite_path)
        sqlite_output.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(sqlite_output)
        _initialize_stats_sqlite(connection, world=world)
        logged_event_count = _sync_events_sqlite(connection, world.events, start_index=0)

    for tick_index in range(1, ticks + 1):
        world.step(1)
        if connection is not None:
            logged_event_count = _sync_events_sqlite(connection, world.events, start_index=logged_event_count)
        if tick_index % log_every == 0 or tick_index == ticks:
            record = _stats_record(world.stats())
            if log_path is not None:
                records.append(json.dumps(record))
            if connection is not None:
                _append_stats_sqlite(connection, record)

    if log_path is not None:
        output.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    if connection is not None:
        connection.close()


def _initialize_stats_sqlite(connection: sqlite3.Connection, world: World) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            seed INTEGER NOT NULL,
            turbo INTEGER NOT NULL,
            config_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stats_log (
            tick INTEGER PRIMARY KEY,
            population INTEGER NOT NULL,
            peak_population INTEGER NOT NULL,
            population_variance REAL NOT NULL,
            population_capacity_fraction REAL NOT NULL,
            peak_population_capacity_fraction REAL NOT NULL,
            crowding_multiplier REAL NOT NULL,
            peak_crowding_multiplier REAL NOT NULL,
            drag_multiplier REAL NOT NULL,
            nutrient_source_strength_multiplier REAL NOT NULL,
            light_intensity REAL NOT NULL,
            light_direction_degrees REAL NOT NULL,
            nodes INTEGER NOT NULL,
            total_energy REAL NOT NULL,
            mean_creature_energy REAL NOT NULL,
            max_creature_energy REAL NOT NULL,
            nutrient_total REAL NOT NULL,
            detritus_total REAL NOT NULL,
            chemical_a_total REAL NOT NULL,
            chemical_b_total REAL NOT NULL,
            births INTEGER NOT NULL,
            deaths INTEGER NOT NULL,
            reproductions INTEGER NOT NULL,
            speciation_events INTEGER NOT NULL,
            species_extinctions INTEGER NOT NULL,
            species_turnover INTEGER NOT NULL,
            predation_kills INTEGER NOT NULL,
            environment_perturbations INTEGER NOT NULL,
            lineage_count INTEGER NOT NULL,
            species_count INTEGER NOT NULL,
            observed_species_count INTEGER NOT NULL,
            peak_species_count INTEGER NOT NULL,
            peak_species_fraction REAL NOT NULL,
            runaway_dominance_detected INTEGER NOT NULL,
            diversity_index REAL NOT NULL,
            mean_nodes_per_creature REAL NOT NULL,
            mean_edges_per_creature REAL NOT NULL,
            mean_motor_edges_per_creature REAL NOT NULL,
            mean_segment_length_per_creature REAL NOT NULL,
            mean_mouths_per_creature REAL NOT NULL,
            mean_grippers_per_creature REAL NOT NULL,
            mean_sensors_per_creature REAL NOT NULL,
            mean_photoreceptors_per_creature REAL NOT NULL,
            mean_speed_recent REAL NOT NULL,
            mean_age_ticks REAL NOT NULL,
            max_age_ticks INTEGER NOT NULL,
            active_grip_latch_count INTEGER NOT NULL,
            peak_grip_latch_count INTEGER NOT NULL,
            mean_gripper_contact_signal REAL NOT NULL,
            mean_gripper_active_signal REAL NOT NULL,
            longest_species_lifespan INTEGER NOT NULL,
            mean_extinct_species_lifespan REAL NOT NULL,
            autotroph_count INTEGER NOT NULL,
            herbivore_count INTEGER NOT NULL,
            predator_count INTEGER NOT NULL,
            trophic_balance_score REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events_log (
            event_index INTEGER PRIMARY KEY,
            tick INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            creature_id INTEGER NOT NULL,
            parent_ids_json TEXT NOT NULL,
            energy REAL NOT NULL,
            genome_hash TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT OR REPLACE INTO run_metadata (id, seed, turbo, config_json)
        VALUES (1, ?, ?, ?)
        """,
        (
            world.seed,
            1 if world.turbo else 0,
            json.dumps(world.config.to_dict(), sort_keys=True),
        ),
    )
    connection.commit()


def _append_stats_sqlite(connection: sqlite3.Connection, record: dict[str, object]) -> None:
    columns = list(record.keys())
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT OR REPLACE INTO stats_log ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(record[column] for column in columns),
    )
    connection.commit()


def _sync_events_sqlite(connection: sqlite3.Connection, events: list[object], start_index: int) -> int:
    for event_index, event in enumerate(events[start_index:], start=start_index):
        connection.execute(
            """
            INSERT OR REPLACE INTO events_log (
                event_index,
                tick,
                event_type,
                creature_id,
                parent_ids_json,
                energy,
                genome_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_index,
                event.tick,
                event.event_type,
                event.creature_id,
                json.dumps(list(event.parent_ids)),
                event.energy,
                event.genome_hash,
            ),
        )
    connection.commit()
    return len(events)

def _load_or_create_world(args: argparse.Namespace) -> World:
    if args.resume is not None:
        world = World.load(args.resume)
        world.turbo = args.turbo
        if args.set:
            world.config = world.config.with_overrides(args.set)
    else:
        config = Config.from_yaml(args.config)
        if args.set:
            config = config.with_overrides(args.set)
        world = World(config=config, seed=args.seed, turbo=args.turbo)
    if args.seed_from is not None and args.resume is None:
        world.seed_from_exported_genomes(args.seed_from)
    if args.seed_demo and args.resume is None:
        world.seed_demo_archetypes()
    return world


if __name__ == "__main__":
    raise SystemExit(main())
