# Working Memory

Long-term project context, key decisions, naming conventions, and lessons learned.

## Evolvable Articulated Morphologies (2026-03-18)

### Problem

Articulated appendages (chains of nodes + motorized edges) never emerged because:
1. Structural mutation only adds 1 node per reproduction attached to a random parent — produces bushy, not linear, topologies
2. No selective pressure for peripheral functional nodes
3. Brain can't easily bootstrap coordinated wave motion across a new chain

### Solution

Four-part fix without new physics:

1. **Chain-extension mutation**: Finds terminal (degree-1) nodes and extends them outward with a motorized edge, inheriting the terminal's node type. Single mutation step produces linear appendages.

2. **Brain warm-start**: When chain extension fires, sets a recurrent coupling weight from parent motor neuron to new motor neuron (0.2–0.6) and offsets the time constant (1.1–1.5x parent tau). Bootstraps phase-shifted oscillation.

3. **Reach bonus**: `mouth_reach_bonus` and `gripper_reach_bonus` multiply feeding/capture effectiveness by distance from center-of-mass. Creates selective pressure for mouths/grippers on stalks.

4. **Observability**: `mean_chain_length` stat (BFS diameter) tracks whether linear morphologies are actually emerging.

### Key Design Decisions

- Chain extension inherits terminal node type (not random) — ensures mouths/grippers stay at tips
- Motor strength on new chain edges is drawn from `gauss(1.5, 0.3)` (biased strong) to ensure appendages move
- `max_nodes_per_creature` (default 16) gates both structural and chain-extension mutations
- All new config fields have backward-compatible defaults of 0.0 so old checkpoints load unchanged
- Reach bonus uses `1.0 + bonus * min(1.0, dist/max_extent)` — linear ramp capped at 1.0

### What Was Deferred

- Explicit `motor_phase` gene (try brain warm-start first)
- Angular joint constraints (floppy chains are biologically plausible)
- Per-node mass variation
- Specialized FLAGELLUM node type (BODY + motors already function as flagella)
