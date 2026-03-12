# Animalcula

Animalcula is a 2D artificial life simulator centered on evolved, physics-based microscopic creatures. The current project focus is to build a headless, testable simulation core first, then layer on visualization, analytics, and later a Rust acceleration path.

## Status

The repository now has an executable headless prototype. The main design source is `ANIMALCULA_SPEC.md`, and the current implementation includes a `uv`-managed Python package, a headless CLI, YAML config loading, dedicated default/nursery/turbo profiles, phase-ordered world stepping, overdamped spring-mass physics, soft node repulsion, grid-backed nutrient/light/detritus fields, nutrient diffusion and decay, finite nutrient consumption, detritus recycling, direct detritus scavenging by mouths, deterministic nutrient sources, deterministic starter archetypes, a direct genome encoding for morphology and CTRNN parameters, genome-driven reproduction, a first structural mutation path that can add a body node and edge, genome export/import for headless seeding, a CTRNN brain runtime with energy, age, and field-gradient sensing, motorized edges with brain-driven actuation, explicit motor actuation energy costs, a first bite-output predation path for overlapping mouth contact, a seeded predator archetype with a real bite-capable output layout, explicit predation kill logging/counting, an explicit reproduce-output control channel when a brain exposes one, stable creature IDs, creature age tracking, deterministic genome hashes in lifecycle events, and lineage-aware stats/reporting with lineage count, a coarse species count, Shannon diversity, trophic structure counts, and mean node-count complexity.

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
uv run animalcula run --config config/default.yaml --ticks 100 --seed 42 --seed-demo --log-stats logs/demo.jsonl --log-every 10
uv run animalcula sweep --config config/default.yaml --sweep sweep.yaml --ticks 100 --seed 42 --seed-demo --out results.jsonl
uv run animalcula nursery --ticks 100 --seed 42 --out checkpoints/nursery.json
uv run animalcula nursery --ticks 100 --seed 42 --top 5 --save-top checkpoints/top_creatures.json --out checkpoints/nursery.json
uv run animalcula run --config config/default.yaml --ticks 0 --seed 42 --seed-from checkpoints/top_creatures.json
uv run animalcula run --config config/turbo.yaml --ticks 300 --seed 42 --seed-demo --turbo
```

## Next Build Step

Build upward from the headless nursery/recycling/turbo/genome loop into fuller evolution and analytics: tighter energy tuning, richer motor control, stronger lineage/speciation metrics, and a less ad hoc seed library format.
