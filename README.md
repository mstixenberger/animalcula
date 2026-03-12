# Animalcula

Animalcula is a 2D artificial life simulator centered on evolved, physics-based microscopic creatures. The current project focus is to build a headless, testable simulation core first, then layer on visualization, analytics, and later a Rust acceleration path.

## Status

The repository now has an executable bootstrap scaffold. The main design source is `ANIMALCULA_SPEC.md`, and the current implementation includes a `uv`-managed Python package, a minimal headless CLI, YAML config loading, phase-ordered world stepping, node-level overdamped physics helpers, spring-connected node integration, and a passing test suite.

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
```

## Next Build Step

Build upward from spring-connected body structure into field-aware world state: environment grids, sampling, and the first energy loop.
