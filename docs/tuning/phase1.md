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

The tuning batch can now hand off directly into the seed-bank workflow by replaying the highest-ranked runs, exporting their top survivors into one combined seed artifact, and optionally launching multi-round promotion:

```bash
uv run python scripts/tune_phase1.py \
  --config config/default.yaml \
  --sweep config/sweeps/phase1_balance.yaml \
  --ticks 1000 \
  --seeds 41,42,43 \
  --workers 4 \
  --turbo \
  --out /tmp/animalcula_phase1.jsonl \
  --save-top /tmp/animalcula_phase1.seedbank.json \
  --top-runs 3 \
  --top-creatures 5 \
  --promote-out-dir /tmp/animalcula_phase1_promotion \
  --promote-rounds 3
```

This emits:

- the usual raw JSONL tuning results
- a `.summary.json` aggregate summary
- a combined seed-bank JSON file built from the strongest replayed runs
- a seed-bank manifest describing which runs contributed survivors
- optional promotion round reports plus `promotion.json`

The raw JSONL and `.summary.json` outputs now include turnover-oriented ecology fields in addition to end-state counts:

- `species_turnover`
- `observed_species_count`
- `peak_species_count`
- `peak_species_fraction`
- `mean_extinct_species_lifespan`
- `drag_multiplier`
- `runaway_dominance_detected`
- `trophic_balance_score`

Once a run produces a checkpoint with promising survivors, use the headless seed-bank loop to promote candidates without manual inspection:

```bash
uv run animalcula extract-genomes checkpoints/demo.json --top 10 --out /tmp/animalcula_seed_bank.json
uv run animalcula evaluate-genomes /tmp/animalcula_seed_bank.json \
  --config config/default.yaml \
  --ticks 300 \
  --seeds 41,42,43 \
  --workers 4 \
  --turbo \
  --out /tmp/animalcula_seed_bank.report.json \
  --save-top /tmp/animalcula_seed_bank.promoted.json \
  --top 3
```

For longer bootstrap loops, chain survivor promotion across multiple rounds and inspect the emitted manifest plus per-round reports:

```bash
uv run animalcula promote-genomes /tmp/animalcula_seed_bank.json \
  --config config/default.yaml \
  --ticks 300 \
  --seeds 41,42,43 \
  --workers 4 \
  --turbo \
  --rounds 3 \
  --top 3 \
  --out-dir /tmp/animalcula_promotion_rounds
```

The resulting `promotion.json` manifest now highlights:

- genome-hash carryover from one promoted round to the next
- whether the top-ranked genome stayed stable across rounds
- diversity drift between each round's input and promoted seed banks

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
- As of March 13, 2026, the shipped default/turbo/nursery profiles now use low nonzero structural, node-role, and motor-topology mutation rates.
  On the same conservative basin, a `2,500`-tick turbo run at seed `41` reached:
  - speciation events `1`
  - observed species `4`
  - peak species `4`
- As of March 13, 2026, predation is now gripper-specialized instead of being passively available to every mouth-only descendant with a bite channel.
  The first specialization pass made predator counts realistic, but also exposed that true predators were rarely converting encounters into kills.
- As of March 13, 2026, grippers can now capture nearby prey nodes and bite damage can be applied to actively gripped victims, while the seeded predator now starts on the grazer nutrient basin to bootstrap real contact.
  On the same conservative basin:
  - a `1,000`-tick turbo run at seed `41` ended with population `5`, deaths `1`, and `predation_kills=1`
  - `5,000`-tick turbo runs at seeds `41` and `42` ended with `predation_kills=10` and `0` respectively
  - seed `41` also reached deaths `11`, species turnover `3`, observed species `5`, and predators `1` / herbivores `13` / autotrophs `2`
  - seed `42` remained viable but softer, ending near predators `1`, herbivores `11`, autotrophs `2`, and observed species `4`
- As of March 13, 2026, CTRNN genomes can also mutate a bounded hidden-neuron prefix independently of morphology-required outputs.
  This does not yet solve predator inconsistency by itself, but it removes a clear architecture ceiling: reproduction can now widen or narrow controller capacity over generations without breaking the existing motor/grip/bite wiring.
- As of March 13, 2026, the shipped headless profiles also enable deterministic nutrient-source shifting every `1000` ticks.
  This is the first environmental-variation slice from the spec. In the current conservative basin check, a `1,000`-tick turbo run at seed `41` still held the same baseline (`population=5`, `deaths=1`, `predation_kills=1`), so the feature is in without destabilizing the existing short-horizon ecology.
- As of March 13, 2026, the shipped headless profiles also enable deterministic medium-cycle light seasons: direction rotates and intensity swings between configured minimum and maximum values every `10,000` ticks across a four-step cycle.
  This completes the next spec-facing environmental variation slice without changing the short-horizon baseline, but it raises the priority on longer `10,000+` multi-seed checks because phototroph and photoreceptor lineages will now see real seasonal pressure.
- As of March 13, 2026, the shipped headless profiles also enable deterministic long-cycle drag-regime shifts every `100,000` ticks, with the active multiplier now exposed in headless stats/logs/sweeps.
  This lands the first spec-facing major environmental shift without adding obstacle mechanics yet. The immediate follow-up is empirical, not structural: long multi-seed runs now need to verify whether regime changes trigger useful turnover instead of simple collapse.
- As of March 13, 2026, headless reporting also tracks peak species share and flags runaway dominance once a species holds more than `80%` of the population for over `5000` ticks.
  That gives tuning runs a first direct monoculture alarm instead of inferring collapse or stagnation only from end-state species counts.

## Interpretation

This is good enough to call the simulation viable in a headless prototype sense, but predator consistency is still the main biological tuning target.

It is not yet good enough to call the ecology balanced:

- the world remains predator-heavy
- species count stays flat at `3`
- speciation events remain `0`
- the herbivore lineage persists but does not diversify

That means the next tuning/design work should target predator payoff and broader trophic balance, not just raw survival:

- raise predator payoff carefully now that specialization is enforced
- improve grazer competitiveness
- validate the new exploratory mutation defaults across more long-horizon multi-seed runs
- keep validating with multi-seed runs at `1000+` ticks
