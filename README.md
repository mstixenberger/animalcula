# Animalcula

Animalcula is a 2D artificial life simulator centered on evolved, physics-based microscopic creatures. The current project focus is to build a headless, testable simulation core first, then layer on visualization, analytics, and later a Rust acceleration path.

## Status

The repository now has an executable headless prototype. The main design source is `ANIMALCULA_SPEC.md`, and the current implementation includes a `uv`-managed Python package, a headless CLI, YAML config loading, dedicated default/nursery/turbo profiles, phase-ordered world stepping, overdamped spring-mass physics, soft node repulsion, gripper latch springs with release thresholds, short-range grip capture against nearby prey nodes, grid-backed nutrient/light/chemical/detritus fields, nutrient diffusion and decay, deterministic nutrient-source shifting on a timestep cycle, deterministic long-cycle nutrient-epoch reseeding with source-strength multipliers, dominance-triggered nutrient perturbations when monoculture lock-in is detected, deterministic medium-cycle light seasons with rotating direction and dim/bright intensity swings, deterministic long-cycle drag-regime shifts, chemical diffusion and decay, finite nutrient consumption, detritus recycling, direct detritus scavenging by mouths, deterministic nutrient sources, deterministic starter archetypes, a direct genome encoding for morphology and CTRNN parameters, inherited genome color genes for visual lineage tracking, genome-driven reproduction, nonzero structural/node-role/motor-topology exploration in the shipped headless profiles, structural mutation that can add a new node and edge, node-type mutation across body/mouth/gripper/sensor/photoreceptor roles, motor-topology mutation that can turn passive edges into motorized joints, bounded hidden-neuron evolution in CTRNN brains, automatic CTRNN output resizing to keep control channels aligned with morphology, genome export/import for headless seeding, exported-seed-bank evaluation and survivor-based multi-round promotion, a CTRNN brain runtime with energy, age, field-gradient sensing, sensor-node chemical sensing, and gripper contact/active signals, motorized edges with brain-driven actuation, brain-driven chemical emission, explicit motor and grip upkeep energy costs, a gripper-specialized bite predation path that can consume actively gripped prey, alarm-chemical deposition on attack, a seeded predator archetype with a real bite-capable output layout and a grazer-basin starting position for trophic bootstrapping, explicit predation kill logging/counting, an explicit reproduce-output control channel when a brain exposes one, stable creature IDs, creature age tracking, deterministic genome hashes plus lineage colors in lifecycle events, phenotype snapshots, checkpoint phylogeny export in JSON and Newick formats, JSONL periodic stats logging, SQLite periodic stats logging with embedded run metadata and lifecycle-event capture, a Tk local viewer plus automatic self-contained HTML fallback on machines without `_tkinter`, viewer-visible lineage colors in both debug backends, low-resolution nutrient and light field overlays in both debug backends, and lineage-aware stats/reporting with lineage count, genome-distance species clustering, clustered species snapshots, explicit speciation events, species-extinction events, environment-perturbation events, observed/peak species counts, species-turnover summaries, peak-species-share and runaway-dominance tracking, peak-population and population-variance tracking, carrying-capacity fraction tracking, direct crowding-multiplier tracking, nutrient and detritus pool visibility in headless reports, total chemical A/B field visibility in headless reports, population-level recent-speed visibility in headless reports, mean and max creature-age visibility in headless reports, mean and max creature-energy visibility in headless reports, mean edge, motor-edge, and segment-length visibility in headless reports, mean mouth and gripper counts per creature in headless reports, mean sensor and photoreceptor counts per creature in headless reports, active and peak grip-latch visibility in headless reports, current seasonal light intensity and direction visibility in headless reports, population-level gripper contact and active-signal visibility in headless reports, longest-species-lifespan tracking, extinct-species lifespan averages, active drag-regime and nutrient-regime reporting, Shannon diversity, trophic structure counts, mean node-count complexity, and trophic-balance scoring throughout CLI, sweep, and seed-bank analysis output, with shared interestingness ranking now penalizing monoculture lock-in and rewarding both the spec’s midrange carrying-capacity band and balanced food webs.

## Development Priorities

1. Test-driven simulation scaffold
2. Headless execution and deterministic runs
3. Checkpoints, logging, and parameter sweeps
4. Minimal debug visualization
5. Rust port for hot paths
6. Browser frontend and advanced analytics

## Working Principles

- Continuous test-driven development
- Documentation updated alongside code
- Reproducible runs through seeded randomness
- `uv` only for Python environments, dependency management, and command execution
- Small, frequent, descriptive commits
- Keep a Changelog from the beginning

## Planned Initial Architecture

- Python package for orchestration and initial simulation
- NumPy/Numba for early performance
- Later Rust core with PyO3 bindings
- Headless CLI as the primary workflow
- Minimal local viewer for debugging

## Near-Term Milestone

Milestone 1 is to prove that hand-seeded creatures can survive, feed, and reproduce in a stable headless simulation with:

- overdamped spring-mass physics
- nutrient and light fields
- CTRNN brains
- energy accounting
- asexual reproduction
- checkpoints and basic metrics

## Repository Conventions

- See `AGENTS.md` for living project context and architectural rules.
- See `CONTRIBUTING.md` for engineering workflow and commit expectations.
- See `CHANGELOG.md` for notable changes.
- See `docs/tuning/phase1.md` for current tuning workflow and findings.

## Quickstart

```bash
uv sync --group dev
uv run pytest
uv run animalcula run --config config/default.yaml --ticks 10 --seed 42
uv run animalcula run --config config/default.yaml --ticks 10 --seed 42 --seed-demo
uv run animalcula run --config config/default.yaml --ticks 10 --seed 42 --seed-demo --save checkpoints/demo.json
uv run animalcula run --resume checkpoints/demo.json --ticks 100
uv run animalcula run --config config/default.yaml --ticks 1 --seed 42 --seed-demo --set energy.reproduction_threshold=0.1
uv run animalcula report checkpoints/demo.json
uv run animalcula events checkpoints/demo.json
uv run animalcula phylogeny checkpoints/demo.json
uv run animalcula phylogeny checkpoints/demo.json --format newick
uv run animalcula species checkpoints/demo.json
uv run animalcula phenotypes checkpoints/demo.json
uv run animalcula extract-genomes checkpoints/demo.json --top 10 --out checkpoints/top_creatures.json
uv run animalcula evaluate-genomes checkpoints/top_creatures.json --config config/default.yaml --ticks 300 --seeds 41,42,43 --workers 4 --out checkpoints/top_creatures.report.json --save-top checkpoints/promoted_top_creatures.json --top 3
uv run animalcula promote-genomes checkpoints/top_creatures.json --config config/default.yaml --ticks 300 --seeds 41,42,43 --workers 4 --rounds 3 --top 3 --out-dir checkpoints/promotion_rounds
uv run animalcula run --config config/default.yaml --ticks 100 --seed 42 --seed-demo --log-stats logs/demo.jsonl --log-every 10
uv run animalcula run --config config/default.yaml --ticks 100 --seed 42 --seed-demo --log-stats-sqlite logs/demo.sqlite --log-every 10
uv run animalcula sweep --config config/default.yaml --sweep sweep.yaml --ticks 100 --seed 42 --seed-demo --workers 4 --out results.jsonl
uv run animalcula nursery --ticks 100 --seed 42 --out checkpoints/nursery.json
uv run animalcula nursery --ticks 100 --seed 42 --top 5 --save-top checkpoints/top_creatures.json --out checkpoints/nursery.json
uv run animalcula run --config config/default.yaml --ticks 0 --seed 42 --seed-from checkpoints/top_creatures.json
uv run animalcula run --config config/turbo.yaml --ticks 300 --seed 42 --seed-demo --turbo
uv run animalcula view --config config/default.yaml --seed 42 --seed-demo
uv run animalcula view --config config/default.yaml --seed 42 --seed-demo --viewer-backend html --html-out /tmp/animalcula_view.html --max-frames 600
uv run python scripts/tune_phase1.py --ticks 1000 --seeds 41,42,43 --workers 4 --turbo --out /tmp/animalcula_phase1.jsonl
uv run python scripts/tune_phase1.py --ticks 1000 --seeds 41,42,43 --workers 4 --turbo --out /tmp/animalcula_phase1.jsonl --save-top /tmp/animalcula_phase1.seedbank.json --top-runs 3 --top-creatures 5 --promote-out-dir /tmp/animalcula_phase1_promotion --promote-rounds 3
```

## Next Build Step

Build upward from the headless nursery/recycling/turbo/genome loop into fuller evolution and analytics: validate the new nutrient-epoch/light/drag variation plus dominance-triggered perturbations across longer multi-seed runs, use the new peak-share and runaway-dominance metrics to separate healthy turnover from monoculture lock-in, study promotion manifests for genome-hash carryover and diversity drift across rounds, and keep tuning predator consistency now that capture-assisted predation can bootstrap and brain capacity can widen.
