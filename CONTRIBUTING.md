# Contributing

## Development Model

This project uses test-driven development from the start. Changes should be built in small vertical slices that include tests, code, and documentation together.

## Required Workflow

1. Start from a failing test or an explicit test gap.
2. Implement the smallest useful change.
3. Run the relevant tests locally.
4. Update documentation affected by the change.
5. Add a changelog note when the change is notable.
6. Commit with a descriptive, multi-line message.

## Commit Guidance

- Commit often.
- Keep commits focused on one concern.
- Prefer verbose commit bodies over terse messages.
- Mention the tests added or updated in the commit body.

Example:

```text
Add deterministic CTRNN update skeleton

- introduce the first brain module and unit tests
- validate tau handling and bounded outputs
- document the module's role in the simulation stack
- defer sensor packing until the world model exists
```

## Testing Expectations

At minimum, contributors should add or maintain:

- unit tests for the changed logic
- regression tests for bug fixes
- smoke tests for simulation flow when relevant
- deterministic seed coverage for stochastic behavior

Protect these invariants early:

- no NaN or inf in simulation state
- stable checkpoint roundtrips
- deterministic fixed-seed behavior
- valid creature/genome decode results

## Documentation Expectations

Keep these files current:

- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- module or API documentation when introduced

## Changelog

This repository follows Keep a Changelog. Add notable entries under `Unreleased` in `CHANGELOG.md`.
