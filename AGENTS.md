# Animalcula Agent Context

This file is a living document. It exists to give future coding agents and human contributors durable project context, working rules, and architectural intent. Update it whenever the implementation, workflow, testing discipline, or project priorities materially change.

## Project Status

- Repository state: executable bootstrap scaffold
- Current source of truth: `ANIMALCULA_SPEC.md`
- Current priority: expand the node-level physics slice into creature- and field-level simulation
- Primary product mode: headless simulation and parameter tuning
- Initial implementation stack: Python + NumPy/Numba
- Deferred stack: Rust core via PyO3, browser frontend

## Architectural Direction

Build the project in layers:

1. Headless simulation core
2. Logging, checkpoints, and parameter sweeps
3. Minimal debug visualization
4. Rust acceleration for hot paths
5. Rich browser visualization and analytics

The project must maintain a stable seam between orchestration and simulation so the Python core can later be replaced by a Rust implementation without breaking higher-level tooling.

## Non-Negotiables

- Practice continuous test-driven development from the start.
- Add or update tests in the same change as production code.
- Prefer small, composable modules over large feature files.
- Keep the simulation headless by default.
- All randomness must be reproducible through explicit seeds.
- Documentation is part of the deliverable, not an afterthought.
- Use `uv` exclusively for Python environment and dependency management.
- Maintain local git history from the beginning with frequent, descriptive commits.
- Maintain `CHANGELOG.md` using Keep a Changelog conventions.

## TDD Workflow

Use this loop continuously:

1. Write or extend a failing test.
2. Implement the smallest change that makes it pass.
3. Refactor while keeping tests green.
4. Update documentation if behavior, interfaces, or decisions changed.
5. Commit the change with a verbose message explaining intent and scope.

### Testing Expectations

- Every new module should arrive with at least one direct test.
- Every bug fix should start with a regression test when practical.
- Add smoke tests for end-to-end simulation runs early.
- Protect invariants aggressively:
  - no NaN or inf in runtime state
  - deterministic results for fixed seeds
  - checkpoint save/load roundtrips
  - bounded energy accounting
  - mutation and decoding invariants

### Minimum Test Categories

- Unit tests for physics, brain, energy, lifecycle, config
- Property-style invariants where useful
- Smoke tests for short seeded runs
- CLI tests for critical commands once the CLI exists

## Documentation Rules

Documentation must be developed in parallel with the codebase.

### Required docs to keep current

- `README.md`
  - project purpose
  - current status
  - quickstart
  - development workflow
- `AGENTS.md`
  - current project context and operational rules
- `CHANGELOG.md`
  - notable changes only
  - grouped under `Added`, `Changed`, `Fixed`, `Removed`
- `CONTRIBUTING.md`
  - engineering workflow, tests, commit conventions

### Documentation triggers

Update docs when any of the following changes:

- repo layout
- public API
- config shape
- execution workflow
- test strategy
- milestone goals
- git workflow or release approach

## Git Workflow

This project starts with local-only tracking. Remote collaboration can be layered on later without changing the basic flow.

### Branching

- Default branch: `main`
- Work on short-lived topic branches once active development begins
- In the earliest bootstrap phase, working directly on `main` is acceptable if commits remain small and well-described

### Commit discipline

- Commit often
- Keep each commit narrow in purpose
- Use verbose commit messages that explain both what changed and why
- Avoid batching unrelated edits together

Recommended commit structure:

```text
Short imperative summary

- what changed
- why it changed
- what tests were added or updated
- any follow-up intentionally deferred
```

### Local tracking rules

- Make a commit at every stable checkpoint
- Do not leave large uncommitted feature piles
- If documentation changes materially, commit them with the implementation they describe
- If a feature is half-built and unstable, either finish a minimal slice or park it behind a safe boundary before committing

## Changelog Rules

Follow Keep a Changelog format.

- Add entries under `Unreleased`
- Record only user-visible or contributor-relevant changes
- Do not dump raw commit messages into the changelog
- Roll `Unreleased` into a versioned section when releases begin

## Definition of Done

A task is not done until:

- code exists
- tests exist and pass for the changed behavior
- relevant docs are updated
- changelog entry is added when appropriate
- the change is committed or explicitly ready to commit

## Initial Build Plan

### Milestone 1

Prove that seeded creatures can survive, feed, and reproduce in a headless simulation.

Scope:

- Python package scaffold
- YAML config
- overdamped 2D spring-mass physics
- nutrient, light, and detritus fields
- CTRNN brains
- mouth feeding and photosynthesis
- asexual reproduction
- death and detritus recycling
- checkpoints and basic stats
- simple local debug viewer

Explicitly deferred:

- Rust core
- browser frontend
- grippers
- predation damage
- chemicals
- sexual reproduction
- species clustering
- advanced analytics UI

Current implementation baseline:

- `uv`-managed Python package scaffold
- YAML config loading
- headless CLI stub
- deterministic world seed handling
- phase-ordered world stepping
- node-level overdamped dynamics helpers
- spring-connected edge accumulation for body structure
- grid-backed nutrient and light field infrastructure
- simple creature energy accounting with basal cost and photosynthesis
- mouth-based nutrient feeding and dead-creature lifecycle cleanup
- deterministic nutrient source placement in the default world
- deterministic starter archetype seeding for demo worlds
- world stats reporting and richer CLI run summaries
- JSON checkpoint save/load for headless workflow continuity
- first asexual reproduction path for energy-rich creatures
- deterministic offspring mutation on reproduction
- CLI save/resume support for checkpoint-driven runs
- CLI checkpoint reporting for quick headless inspection
- nested config overrides for experiment-driven CLI runs
- CTRNN brain runtime with sensor inputs and simple motor-force output
- demo archetypes now include deterministic starter brains where appropriate
- motorized edges driven by brain outputs for simple joint actuation
- offspring brain weights, biases, and taus mutate during reproduction
- sequential YAML-driven parameter sweep command for headless exploration
- first interestingness score for ranking sweep results
- offspring motor strengths mutate during reproduction
- stable creature IDs and an append-only event log for birth, death, and reproduction
- initial passing pytest suite

## Update Protocol

When making a meaningful architectural or workflow decision:

1. Update this file with the new rule or context.
2. Update the relevant human-facing doc if needed.
3. Add a changelog note if the change affects contributors or users.

If this file drifts from reality, fix the file rather than working around it.
