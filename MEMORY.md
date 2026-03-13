# Animalcula Working Memory

This file is a rolling handoff note for active implementation progress. Keep it concise, current, and operational. Durable project rules belong in `AGENTS.md`; this file is for recent progress, current focus, and likely next slices.

## Current State

- Branch state: working directly on `main` with frequent stable checkpoints
- Product focus: spec-aligned headless simulation, logging, and tuning loop
- Test status at last update: `uv run pytest` passing after `baaf044`
- Recent completed slices:
  - direct trophic-balance weighting in shared interestingness scoring
  - deterministic nutrient epoch reseeding with source-strength multipliers
  - dominance-triggered nutrient perturbations with explicit environment events
- SQLite periodic stats logging
- SQLite run metadata table with seed/turbo/effective config payload
- direct crowding-stress visibility in CLI/logs/sweeps/seed-bank summaries
- SQLite event-table capture for lifecycle/environment events during logged runs
- nutrient and detritus pool visibility in headless stats/logs/sweeps/seed-bank summaries
- total chemical A/B field visibility in headless stats/logs/sweeps/seed-bank summaries

## Current Priority

- Keep tightening the headless ecosystem loop before adding broader UI work
- Prefer spec-aligned environmental variation, observability, and tuning improvements
- Maintain clean, narrow commits with matching tests/docs

## Active Constraints

- Do not stop after each slice; continue into the next sensible spec-aligned task
- Update tests in the same change as code
- Update `README.md`, `CHANGELOG.md`, and `AGENTS.md` when behavior/workflow materially changes
- Use `uv` for Python execution and `apply_patch` for file edits

## Likely Next Slices

- Expand long-run observability where runtime mechanics already exist but are not yet surfaced
- Continue closing small headless-spec gaps before attempting larger UI or Rust work
