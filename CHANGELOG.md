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
- Detritus field storage with recycling back into the nutrient grid
- Grid diffusion/decay helpers for evolving environment fields
- Soft node repulsion for first-pass contact handling in the physics loop

### Changed

- Locked Python environment and dependency management to `uv` only
- Feeding now consumes nutrients from the field instead of sampling an effectively infinite food source
