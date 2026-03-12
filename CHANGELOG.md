# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project will adopt semantic versioning once versioned releases begin.

## [Unreleased]

### Added

- Initial project operating documents: `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, and `CHANGELOG.md`
- A documented engineering policy for continuous test-driven development
- A documented local git workflow with frequent, descriptive commits
- A documented Milestone 1 architecture and delivery direction
- Initial `uv`-managed Python package scaffold with a headless CLI entrypoint
- Default YAML configuration and minimal public API surface
- First passing test suite covering config loading, world stepping, deterministic seeding, and CLI execution
- Node-level runtime types and overdamped physics helpers
- Phase-ordered world stepping with basic node integration under test
- Edge runtime types and spring accumulation for connected body structure
- Grid-backed environment fields with a static directional light gradient
- Creature energy helpers and a world energy phase with basal cost and photosynthesis
- Mouth-based nutrient feeding and lifecycle cleanup for depleted creatures
- Deterministic nutrient source placement in the default world
- Deterministic starter archetype seeding for demo populations
- World stats reporting and richer CLI run summaries, including demo seeding
- JSON checkpoint save/load for world-state roundtrips
- A first asexual reproduction path for energy-rich creatures
- Deterministic offspring mutation on reproduction
- CLI save/resume support for checkpoint-driven runs
- CLI checkpoint reporting for quick saved-state inspection
- Nested config overrides for experiment-driven CLI runs
- CTRNN brain runtime with sensor inputs and simple motor-force integration
- Brain-equipped deterministic starter archetypes in the demo population
- Motorized edges driven by brain outputs for simple joint actuation
- Offspring brain weights, biases, and time constants mutate during reproduction
- Sequential YAML-driven parameter sweep command with JSONL result export
- Initial interestingness score included in sweep results
- Offspring motor strengths mutate during reproduction
- Stable creature IDs and append-only birth/death/reproduction event logging
- Event counts exposed through stats, CLI reporting, and sweep results
- Interestingness scoring now rewards lifecycle activity as well as population/energy
- Checkpoint-backed CLI event export for direct run-history inspection
- Periodic JSONL stats logging during headless runs
- Creature-level population safeguards with crowding pressure and optional immigration floor
- A dedicated `config/nursery.yaml` profile and `animalcula nursery` workflow for generous bootstrap runs
- A dedicated `config/turbo.yaml` profile and CLI `--turbo` mode for faster headless exploration
- Detritus field storage with recycling back into the nutrient grid
- Grid diffusion/decay helpers for evolving environment fields
- Soft node repulsion for first-pass contact handling in the physics loop
- Field-gradient sensing added to the brain input pipeline alongside light, nutrients, and energy
- Creature age tracking and normalized age sensing in the runtime and checkpoint format
- Nursery runs can now export top survivors as standalone JSON seed artifacts
- Direct genome encoding for morphology and CTRNN parameters
- Genome-driven reproduction and genome persistence in creature/checkpoint payloads
- `animalcula run --seed-from ...` to bootstrap a world from exported genome artifacts
- Deterministic genome hashes in lifecycle events and CLI event export
- Lineage count and Shannon diversity now surface in stats, CLI output, stats logs, and sweep results
- Brain outputs can now explicitly permit or suppress reproduction when a creature has a reproduce control channel
- Motor actuation now contributes to per-tick energy cost
- Mouth nodes can now scavenge energy directly from detritus patches
- A first predation path allows bite-output mouths to drain overlapping creatures and gain transferred energy
- Predation kills now surface in lifecycle events, stats, and headless reporting
- The seeded predator archetype now has a bite-capable brain layout
- A first structural mutation path can add a new body node and edge during genome mutation
- Coarse species counts now surface in stats, CLI output, and sweep results alongside exact lineage counts
- Trophic-role counts and mean node-count complexity now surface in stats, CLI output, and sweep results
- `animalcula species <checkpoint>` now prints coarse per-species population snapshots from saved worlds
- `animalcula phenotypes <checkpoint>` now prints per-creature phenotype snapshots from saved worlds
- Chemical A/B environment grids with diffusion and decay
- Sensor-node chemical sensing, including local chemical gradients in brain inputs
- Brain-driven chemical emission and alarm-chemical deposition during predation
- A first gripper mechanics slice with latch creation, release, checkpoint persistence, and grip-spring forces
- Gripper contact and active-state signals in the brain input pipeline
- Node-type genome mutation across body, mouth, gripper, sensor, and photoreceptor roles
- Automatic CTRNN output resizing during genome mutation so control-channel counts stay valid as morphology changes
- Motor-topology genome mutation that can convert passive edges into motorized joints
- Genome-distance species clustering used for species counts and species snapshot grouping
- Explicit `speciation` events in the lifecycle log, plus speciation counts in headless stats and sweep output
- Species-extinction events and longest-species-lifespan tracking in the headless species ledger
- `animalcula extract-genomes <checkpoint> --top N --out ...` for generic checkpoint-to-seed export
- Sweep output now includes explicit run-health flags such as extinction/speciation/predation presence
- A workerized `scripts/tune_phase1.py` batch runner for multi-seed tuning with raw and aggregate outputs
- Aggregate sweep summaries now include diversity and trophic-balance averages
- Focused phase-1 sweep configs for economy and trophic-balance tuning
- A living tuning note in `docs/tuning/phase1.md` with current viability findings and validation commands
- A spec-aligned seeded triangle predator archetype with a mouth, gripper, and three motorized edges

### Changed

- Locked Python environment and dependency management to `uv` only
- Feeding now consumes nutrients from the field instead of sampling an effectively infinite food source
- Brain configs now default to the expanded 16-channel sensing vector used by the current world runtime
- Species and phenotype snapshot exports now expose richer morphology and motion summaries
- Active grip latches now contribute to the per-tick energy budget
- The default starter trio now matches the spec more closely: passive alga, grazer, and triangular predator instead of a mislabeled light-feeding generalist
