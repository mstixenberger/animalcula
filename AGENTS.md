Do not stop after every task, resume with the next task what makes the most sense regarding to spec - only stop when the full implemention is complete

# Animalcula Agent Context

This file is a living document. It exists to give future coding agents and human contributors durable project context, working rules, and architectural intent. Update it whenever the implementation, workflow, testing discipline, or project priorities materially change.

## Project Status

- Repository state: executable headless prototype
- Current source of truth: `ANIMALCULA_SPEC.md`
- Current priority: build the permanent browser frontend path in parallel with headless ecosystem tuning
- Primary product mode: headless simulation plus live browser inspection
- Initial implementation stack: Python + NumPy/Numba
- Deferred stack: Rust core via PyO3, browser frontend

## Architectural Direction

Build the project in layers:

1. Headless simulation core
2. Logging, checkpoints, and parameter sweeps
3. Minimal debug visualization
4. Rich browser visualization and analytics
5. Rust acceleration for hot paths

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
- `MEMORY.md`
  - recent progress checkpoints
  - current implementation focus
  - likely next slices for continuity
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
- finite nutrient consumption from the field during feeding
- deterministic nutrient source placement in the default world
- nutrient diffusion/decay and detritus recycling in the environment update
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
- event counts exposed through stats, CLI reporting, and sweep results
- interestingness scoring now rewards lifecycle activity, not just static population/energy
- checkpoint-backed CLI event export for direct observation of run history
- periodic JSONL stats logging during headless runs
- periodic SQLite stats logging during headless runs
- SQLite stats logs now also store run metadata for seed/turbo/effective-config recovery
- SQLite stats logs now also capture the lifecycle/environment event stream alongside periodic stats rows
- creature-level population safeguards: crowding pressure and optional immigration floor
- dedicated nursery config profile and nursery CLI workflow for generous bootstrapping runs
- soft node repulsion for first-pass contact avoidance in dense physics states
- light and nutrient field gradients now feed the sensing phase alongside average intensity and energy
- `turbo` world/CLI mode plus `config/turbo.yaml` for faster headless parameter exploration
- creature age is now tracked, serialized, and exposed as a normalized brain input
- nursery runs can export top survivors as standalone JSON seed artifacts
- direct genome encoding now exists for morphology and CTRNN parameters
- reproduction now mutates genomes and decodes offspring from them instead of mutating raw runtime state
- fresh worlds can seed themselves from exported genome artifacts via the headless workflow
- lifecycle events now carry genome hashes to support later lineage/speciation analysis
- world stats, CLI output, and sweep results now expose lineage count and Shannon diversity
- brains can now explicitly gate reproduction if they expose an extra reproduce output beyond motor channels
- the energy budget now charges for motor actuation as well as basal upkeep
- mouths can now scavenge energy directly from detritus patches, not just wait for recycling
- a first predation slice exists: bite outputs on mouth nodes can drain overlapping victims and transfer energy
- predation kills now surface as a distinct event type and stats count for headless analysis
- the seeded amoeba-lite archetype now exposes motor plus bite outputs, so the default seed set matches the current predation wiring better
- genomes now have a first structural mutation path that can grow one extra body node and edge
- coarse species counting is now derived from genome structure and exposed in headless stats/reporting
- headless stats/reporting now include trophic-role counts and mean node-count complexity
- checkpoint inspection now includes per-species snapshot summaries via the CLI
- checkpoint inspection now includes per-creature phenotype snapshot summaries via the CLI
- chemical A/B fields now diffuse and decay alongside nutrient/light/detritus fields
- sensor nodes now contribute chemical intensity and gradient signals to the brain input pipeline
- brains can now emit chemical A/B signals into the environment
- predation now emits an alarm-style chemical trace from attacked creatures
- gripper nodes can now latch across creatures, apply latch-spring forces, and release when grip outputs drop or yield is exceeded
- gripper contact/active state is now part of the sensed control state and grip upkeep contributes to the energy budget
- genome mutation can now change node roles across body, mouth, gripper, sensor, and photoreceptor types
- genome mutation now resizes CTRNN output capacity to stay aligned with morphology control requirements while preserving surplus control channels
- genome mutation can now toggle passive edges into motorized joints, broadening the control topology that evolution can explore
- species counting and species snapshots now use genome-distance clustering instead of a pure coarse-signature placeholder
- the lifecycle/event stream now records explicit `speciation` events for newly observed clustered species
- species turnover tracking now records `species_extinction` events and keeps first-seen/last-seen timing for lifespan metrics
- the headless CLI now supports generic checkpoint-to-seed extraction via `extract-genomes`, not just nursery-specific export
- headless stats, CLI output, stats logs, sweep records, and tuning summaries now expose observed/peak species counts, aggregate species turnover, and extinct-species lifespan averages
- exported genome seed banks can now be evaluated across fresh multi-seed runs and promoted back into smaller ranked seed artifacts through the headless CLI
- multi-round `promote-genomes` workflows now chain survivor-derived promoted artifacts across rounds and emit per-round reports plus a promotion manifest
- promotion manifests now summarize genome-hash carryover, top-rank continuity, and diversity drift across rounds for headless lineage analysis
- `scripts/tune_phase1.py` can now export combined seed banks from the highest-ranked tuning runs and optionally chain them straight into multi-round promotion manifests
- the shipped default, turbo, and nursery profiles now use low nonzero structural/node-role/motor-topology mutation rates so long-run headless runs can actually explore beyond the starter morphology basin
- predation is now treated as a gripper-specialized niche in the simulation and trophic-role reporting, so mouth-only descendants default back to herbivore/scavenger behavior instead of inflating predator counts
- grippers can now capture nearby prey nodes instead of requiring exact overlap, and bite damage can be applied to actively gripped victims to make specialized predators mechanically viable again
- the seeded triangle predator now starts on the grazer nutrient basin so headless demo worlds bootstrap real predator-prey contact instead of isolating the predator on a separate food source
- CTRNN genomes can now grow or shrink a bounded hidden-neuron prefix independently of required control outputs, so reproduction can increase controller capacity without breaking morphology-aligned output wiring
- environment configs can now shift nutrient-source locations on a deterministic timestep cycle, covering the first spec-facing slice of environmental variation without breaking reproducibility
- light gradients can now rotate and dim/brighten on a deterministic medium-horizon seasonal cycle, so environmental variation now covers both nutrient and light regimes
- long-horizon drag regimes can now cycle deterministically across configured multipliers, and the active drag multiplier is exposed through headless stats/logs/sweeps for tuning visibility
- long-horizon nutrient epochs can now fully reseed nutrient-source layouts while cycling source-strength multipliers, and the active nutrient multiplier is exposed through headless stats/logs/sweeps/seed-bank reports for tuning visibility
- runaway-dominance detection can now trigger deterministic nutrient-source perturbations, recorded as explicit `environment_perturbation` events and exposed through headless stats/logs/sweeps/seed-bank reports
- `animalcula view` now auto-falls back to a self-contained HTML viewer when Tkinter is unavailable, so local visual inspection still works on non-Tk Python builds
- the HTML fallback viewer now includes playback-speed control plus per-frame trophic summary cards, so the Tk-free path is useful for actual inspection instead of only crash-avoidance
- headless stats/logs/sweeps/seed-bank reports now also expose peak species share and runaway-dominance detection so long runs can distinguish diversification from monoculture lock-in
- shared interestingness scoring now penalizes monoculture-heavy runs using peak species share and explicit runaway-dominance detection, so automated ranking favors turnover over lock-in
- headless stats/logs/sweeps/seed-bank reports now also expose peak population and population variance so tuning can distinguish flatlined runs from boom-bust ecology
- headless stats/logs/sweeps/seed-bank reports now also expose current and peak carrying-capacity fractions, and shared interestingness scoring now rewards the spec’s midrange population band instead of blindly favoring cap-saturation
- headless stats/logs/sweeps/seed-bank reports now also expose current and peak crowding multipliers, so the existing over-cap metabolic stress is directly inspectable during tuning
- headless stats/logs/sweeps/seed-bank reports now also expose total chemical A/B field mass, so signaling activity is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose total nutrient and detritus pools, so stored environmental energy is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose population-level recent speed, so locomotion is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose active and peak grip-latch counts, so gripper-heavy predator mechanics are visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose current seasonal light intensity and direction, so light-regime shifts are visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean gripper contact and active signals, so predator engagement quality is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean and max creature ages, so lifecycle durability is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean and max creature energy, so individual survivor quality is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean mouth and gripper counts, so morphology mix is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean sensor and photoreceptor counts, so sensing morphology is visible in headless analysis
- headless stats/logs/sweeps/seed-bank reports now also expose mean motor-edge count, so control topology is visible in headless analysis
- direct CLI run/report output and periodic stats logs now also expose trophic-balance scoring, so single-run inspection uses the same ecology metric vocabulary as sweeps and seed-bank analysis
- shared interestingness scoring now also rewards trophic balance directly, so automated ranking favors actual mixed food webs rather than treating role balance as a passive side metric
- sweep and seed-bank analysis now expose a trophic-balance score so headless selection loops can reward viable mixed ecologies, not just high population
- initial passing pytest suite
- genomes now carry inherited RGB color genes for cosmetic lineage tracking without changing species clustering behavior
- checkpoint payloads, phenotype snapshots, and both debug viewer backends now preserve and render lineage colors so evolving families are visually trackable
- saved lifecycle events now preserve lineage colors too, and the CLI can export checkpoint phylogeny graphs as JSON or Newick from the recorded parent pointers
- both debug viewer backends now also render coarse nutrient and light field overlays so the current environment is visible during local inspection
- both debug viewer backends now support cycling/selecting nutrient, light, chemical A/B, and detritus overlays for fuller spec-aligned field inspection
- render snapshots now carry species/lineage metadata for living creatures, and both debug viewer backends now support basic click-to-inspect creature metadata
- the shipped demo/viewer experience is now tuned for clearer first impressions: nutrient-first field rendering, faster visible motion, and slightly more durable seeded archetypes
- both debug viewer backends now support follow-and-zoom around the selected creature, making the local viewer materially more useful for real movement inspection
- both debug viewer backends now also support ambient auto-cycling camera behavior and stronger fast-forward controls, which matters for long-running display use as well as tuning
- the shipped demo archetypes are now materially more legible for viewing: a four-node worm grazer replaces the old two-node grazer, seeded bodies use larger tripod/triangle silhouettes, and both debug viewer backends draw specialized mouth/gripper/sensor/photoreceptor glyphs
- `animalcula view` now pre-warms fresh worlds by default before opening, and the repo ships `config/display.yaml` for a more active display/inspection profile without ad hoc CLI overrides
- ambient viewer selection is now activity-aware instead of purely ID-ordered, so the display camera tends to stay on critters that are moving or otherwise visually alive
- both debug viewer backends now draw translucent creature silhouettes beneath the node graph, so body plans read more like organisms than disconnected springs
- both debug viewer backends now draw a compact on-canvas identity label for the selected creature, which keeps species/role context visible during follow mode
- the deterministic `--seed-demo` founding population now starts at nine creatures instead of three, keeping the same three archetype species but giving the default ecology/viewer paths more actual contact opportunities
- genomes now carry bounded visual-only traits for silhouette scale, appendage glyph scale, and body banding; they are inherited, mutated, surfaced in phenotype data, and rendered in both viewers without changing physical interaction radii
- `animalcula view` now shows a TTY progress bar during warmup, so the pre-view inspection path communicates progress instead of looking stuck
- HTML viewer launches now auto-open the generated file in the browser by default; keep `--no-open-browser` for scriptable/test workflows
- `animalcula view` now also shows a second `recording html viewer` TTY progress phase when the HTML fallback is baking frames, so the terminal no longer appears stalled after warmup
- both debug viewer backends now surface a compact ecology HUD/history layer with species/diversity, recent lifecycle-event deltas, short population/species/predator traces, and selected-creature speed/energy-trend readouts
- the permanent frontend path now starts in earnest with `animalcula web`: a live FastAPI/WebSocket browser app shell; Tk remains a fallback/debug viewer and should not absorb product-facing UI work that belongs in the browser

## Update Protocol

When making a meaningful architectural or workflow decision:

1. Update this file with the new rule or context.
2. Update the relevant human-facing doc if needed.
3. Add a changelog note if the change affects contributors or users.

If this file drifts from reality, fix the file rather than working around it.
