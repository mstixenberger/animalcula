# Animalcula Working Memory

This file is a rolling handoff note for active implementation progress. Keep it concise, current, and operational. Durable project rules belong in `AGENTS.md`; this file is for recent progress, current focus, and likely next slices.

## Current State

- Branch state: working directly on `main` with frequent stable checkpoints
- Product focus: spec-aligned headless simulation, logging, and tuning loop
- Test status at last update: focused phylogeny/export tests passing after wiring lineage-colored event persistence
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
- population-level recent-speed visibility in headless stats/logs/sweeps/seed-bank summaries
- active and peak grip-latch visibility in headless stats/logs/sweeps/seed-bank summaries
- current seasonal light intensity and direction visibility in headless stats/logs/sweeps/seed-bank summaries
- mean gripper contact and active-signal visibility in headless stats/logs/sweeps/seed-bank summaries
- mean and max creature-age visibility in headless stats/logs/sweeps/seed-bank summaries
- mean and max creature-energy visibility in headless stats/logs/sweeps/seed-bank summaries
- mean mouth and gripper counts in headless stats/logs/sweeps/seed-bank summaries
- mean sensor and photoreceptor counts in headless stats/logs/sweeps/seed-bank summaries
- mean edge count and segment length in headless stats/logs/sweeps/seed-bank summaries
- mean motor-edge count in headless stats/logs/sweeps/seed-bank summaries
- `animalcula view` now has an HTML fallback path for machines without Tkinter
- the HTML fallback viewer now has playback-speed control plus per-frame trophic summary cards
- genomes now carry inherited RGB lineage colors that persist through checkpoints and exported genome payloads
- both debug viewer backends now render lineage colors instead of relying only on trophic-role outlines
- saved lifecycle events now preserve lineage colors, enabling checkpoint phylogeny reconstruction for dead as well as living branches
- `animalcula phylogeny` now emits checkpoint lineage graphs as JSON or Newick from recorded parent pointers
- both debug viewer backends now render coarse nutrient and light field overlays instead of showing creatures against a contextless background
- both debug viewer backends now support selectable nutrient/light/chemical/detritus overlays rather than a single baked field blend
- render snapshots now carry species ID, genome hash, parent ID, and birth/age metadata for living creatures
- both debug viewer backends now support click-to-inspect creature metadata, with an HTML inspector card stack and Tk overlay selection readout
- `animalcula view` now defaults to a clearer nutrient-first field mode and higher steps-per-frame, and demo archetypes start with slightly higher energy for a better first watch
- both debug viewer backends now support following the selected creature and zooming in, which should materially improve the “blob” first impression problem
- both debug viewer backends now support ambient auto-cycling camera mode and stronger fast-forward controls, which better matches the long-run display plus tuning target
- the shipped demo archetypes are now more legible and visibly specialized: a four-node worm grazer replaces the old two-node line, seeded bodies are slightly larger, and the viewers render dedicated glyphs for mouths, grippers, sensors, and photoreceptors

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

- Close the next spec-facing runtime or viewer gap with clear user-visible payoff
- Prefer slices that make evolving lineages/ecologies easier to inspect without committing to the full browser frontend yet
- Good next candidates: lightweight field visualization in the debug viewer, richer checkpoint/species lineage summaries, or first phenotype-space export
- Good next candidates: richer checkpoint/species lineage summaries, first phenotype-space export, or stronger species/lineage analytics beyond single-creature inspection
