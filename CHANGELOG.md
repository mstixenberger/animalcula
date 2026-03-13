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
- A minimal local `animalcula view` command backed by a Tk debug viewer and richer render snapshots
- Species-turnover reporting in headless stats, CLI output, stats logs, sweep records, and phase-1 tuning summaries
- `animalcula evaluate-genomes` for ranking exported seed banks across fresh multi-seed runs and exporting promoted seed artifacts
- `animalcula promote-genomes` for chaining multi-round seed-bank evaluation and survivor promotion with a round-by-round manifest
- Promotion manifests now include genome-hash carryover, top-rank continuity, and diversity-drift analytics across rounds
- `scripts/tune_phase1.py` can now export survivor seed banks from top-ranked runs and optionally launch multi-round promotion directly from the tuning batch
- Default, turbo, and nursery profiles now enable low nonzero structural, node-role, and motor-topology mutation rates for real exploratory evolution
- Predation and trophic-role classification now require gripper-bearing morphology, reducing accidental predator labeling from mouth-only descendants
- Sweep and seed-bank analysis now expose a trophic-balance score for ranking ecologies by role mix, not only size/activity
- Grippers can now capture nearby prey nodes and bite damage can apply to actively gripped victims, restoring a mechanical path for specialized predators to secure kills
- CTRNN genomes can now mutate a bounded hidden-neuron prefix independently of control-output count, so controller capacity can grow or shrink across generations without breaking output wiring
- Environment configs can now shift nutrient-source locations on a deterministic timestep cycle, adding the first spec-facing slice of ongoing environmental variation
- Environment configs can now also rotate and dim/brighten the light gradient on a deterministic seasonal cycle
- Environment configs can now also cycle long-horizon drag regimes across configured multipliers
- Headless runs now detect runaway species dominance from sustained >80% monoculture streaks and preserve that state across checkpoints
- Shared interestingness scoring now penalizes peak-share and runaway-dominance signals so automated ranking deprioritizes monoculture lock-in
- Headless runs now track peak population and population variance across the full run, preserving the population series summary across checkpoints

### Changed

- Locked Python environment and dependency management to `uv` only
- Feeding now consumes nutrients from the field instead of sampling an effectively infinite food source
- Brain configs now default to the expanded 16-channel sensing vector used by the current world runtime
- Species and phenotype snapshot exports now expose richer morphology and motion summaries
- Active grip latches now contribute to the per-tick energy budget
- The default starter trio now matches the spec more closely: passive alga, grazer, and triangular predator instead of a mislabeled light-feeding generalist
- `World.snapshot()` now carries read-only render payloads so local visualization can stay decoupled from mutable sim internals
- Sweep and tuning aggregates now summarize observed/peak species counts and extinct-species lifespan averages alongside concurrent species counts
- Shared interestingness scoring now also admits turnover-aware ranking inputs for automated seed-bank evaluation
- Seed-bank promotion now exports survivor artifacts from evaluation runs instead of merely re-emitting the original input candidates
- The shipped headless profiles no longer keep structural exploration effectively disabled by default
- Predator-heavy long-run summaries are now less inflated by control-layout artifacts because mouth-only biters fall back to herbivore reporting and behavior
- Headless analysis output can now distinguish balanced role mixes from one-sided herbivore or predator booms
- The seeded triangle predator now starts on the grazer nutrient basin so default demo worlds bootstrap real trophic contact instead of isolating the predator on a separate nutrient source
- Old config/checkpoint payloads now backfill sane defaults for the new hidden-neuron evolution settings instead of failing to load
- Old config/checkpoint payloads now also backfill sane defaults for nutrient-source shifting settings instead of failing to load
- Old config/checkpoint payloads now backfill sane defaults for seasonal light-cycle settings instead of failing to load
- Phase-1 tuning can now hand off directly into seed-bank promotion without manual checkpoint/export steps between the batch run and promotion loop
- Headless stats, logs, sweeps, and seed-bank analysis now expose the active drag multiplier so long-cycle regime shifts remain observable during tuning
- Headless stats, logs, sweeps, tuning summaries, and seed-bank reports now also expose peak species share plus runaway-dominance flags for long-run ecology triage
- Sweep and seed-bank ranking now use the new dominance signals instead of merely logging them
- Headless stats, logs, sweeps, tuning summaries, and seed-bank reports now also expose peak population and population variance for boom-bust triage
