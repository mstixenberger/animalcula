# Phase 1 Tuning

Phase 1 tuning is focused on one question first: can the seeded world stay alive long enough to reproduce without immediate extinction or runaway collapse?

## Tools

Use the workerized batch runner:

```bash
uv run python scripts/tune_phase1.py \
  --config config/default.yaml \
  --sweep config/sweeps/phase1_balance.yaml \
  --ticks 1000 \
  --seeds 41,42,43 \
  --workers 4 \
  --turbo \
  --out /tmp/animalcula_phase1_balance.jsonl
```

Useful sweep configs:

- `config/sweeps/phase1_default_economy.yaml`
- `config/sweeps/phase1_balance.yaml`

## Current Findings

As of March 12, 2026, after correcting the starter predator archetype to match the spec more closely:

- A viable conservative basin exists around:
  - `energy.reproduction_threshold=30.0`
  - `environment.nutrient_source_strength=4.0`
  - `energy.feed_rate=0.02`
  - `energy.photosynthesis_rate=0.008`
- Across 1,000 ticks and seeds `41,42,43`, that basin stayed alive in all runs and typically ended near:
  - population `7`
  - reproductions `4`
  - deaths `0`
  - species count `3`
  - autotrophs `1`
  - herbivores `1`
  - predators `5`
- A 5,000-tick stress check with the same basin also stayed alive across seeds `41,42,43`:
  - population `10-13`
  - reproductions `21-25`
  - deaths `14-15`
  - predation kills `14-15`
  - autotrophs `2`
  - herbivores `1`
  - predators `7-10`

## Interpretation

This is good enough to call the simulation viable in a headless prototype sense.

It is not yet good enough to call the ecology balanced:

- the world remains predator-heavy
- species count stays flat at `3`
- speciation events remain `0`
- the herbivore lineage persists but does not diversify

That means the next tuning/design work should target trophic balance and diversification, not just raw survival:

- tighten predator specialization and payoff
- improve grazer competitiveness
- increase the chance of divergent morphology/brain evolution
- keep validating with multi-seed runs at `1000+` ticks
