# Animalcula — An Artificial Life Simulator with Evolved Physics-Based Creatures

## Project Vision

A 2D artificial life simulation where soft-body articulated creatures inhabit a viscous
microscopic world. Creatures evolve morphology, locomotion, feeding strategy, and combat
through neuroevolution — small recurrent neural networks whose weights and body plans are
subject to mutation and selection. The simulation produces emergent food webs, predator-prey
arms races, speciation events, and phylogenetic trees — all driven by physically-grounded
mechanics, not hand-coded behaviors.

**Aesthetic target:** footage of tardigrades, rotifers, and paramecia under a dark-field
microscope — translucent bodies, visible internal structure, caustic light, organic motion.

---

## 1. The World

### 1.1 Domain

- **2D continuous space**, toroidal (wraps around) or bounded with soft walls.
- Recommended world size: 1000×1000 units (tunable).
- Spatial grid overlay for field quantities (nutrients, light, chemicals) at ~5 unit
  resolution → 200×200 grid cells.

### 1.2 Physics Regime: Life at Low Reynolds Number

All motion is **overdamped** — there is no inertia. This is physically accurate for
microscopic life in viscous fluid (Purcell's "scallop theorem" regime). The governing
equation for each node is simply:

```
F_net = γ · v
```

Where `γ` is the drag coefficient and `v` is velocity. No `F = ma` integration needed —
velocity is directly proportional to force at every instant. This is:

- **Computationally cheap** (no momentum state, no instability from large accelerations)
- **Biologically authentic** (real microorganisms experience this)
- **Evolutionarily interesting** (reciprocal motion produces zero displacement — creatures
  must evolve non-reciprocal gaits to move, exactly like real microswimmers)

### 1.3 Environment Fields (on spatial grid)

| Field         | Dynamics                          | Role                            |
|---------------|-----------------------------------|---------------------------------|
| **Nutrients** | Diffusion + localized sources + depletion by feeding | Primary energy source for autotrophs and detritivores |
| **Light**     | Static or slowly shifting gradient from a "sun" | Energy source for photosynthesizers; sensory input |
| **Chemical A**| Emitted by organisms, diffuses and decays | Pheromone / alarm signal — creatures can evolve to sense it |
| **Chemical B**| Same mechanics, different channel  | Allows multiple signaling strategies |
| **Detritus**  | Deposited on death, slowly decays into nutrients | Recycling loop; enables scavenger niche |

Nutrient sources should be **spatially heterogeneous and time-varying**: pulsing vents,
moving patches, seasonal cycles. This prevents monoculture equilibria.

### 1.4 Environmental Variation (Seasons / Epochs)

To drive ongoing adaptation and prevent evolutionary stagnation:

- **Short cycle (~1000 timesteps):** nutrient source locations shift slowly.
- **Medium cycle (~10,000 timesteps):** light direction/intensity changes ("seasons").
- **Long cycle (~100,000 timesteps):** major environmental shifts — a nutrient source dies,
  a new obstacle appears, drag coefficient changes. These are extinction-level events that
  prune the tree of life and open niches.

---

## 2. Creature Architecture

### 2.1 Body: Spring-Mass Segment Graph

Each creature is a **planar graph** of **nodes** connected by **edges (segments)**.

```
Node:
  - position: (x, y)       — world coordinates
  - velocity: (x, y)       — derived from forces (overdamped)
  - radius: float           — collision radius, visual size
  - type: enum {body, mouth, gripper, sensor, photoreceptor}
  - drag_coeff: float       — can vary per node (e.g., flagella have lower drag)

Edge (Segment):
  - connects: (node_a, node_b)
  - rest_length: float
  - stiffness: float        — spring constant
  - damping: float          — dashpot constant
  - has_motor: bool         — can the neural net actuate this joint?
  - motor_torque_max: float — peak torque
  - current_angle: float    — measured
  - visual_width: float     — for rendering
```

**Constraints:**
- Minimum 2 nodes, maximum 12 nodes (tunable, but >12 gets expensive).
- Graph must be connected.
- Typical creature: 3–8 nodes.

### 2.2 Body Node Types and Their Physics

**Body nodes:** Standard structural nodes. Provide shape and mass.

**Mouth nodes:** Can absorb nutrients from the grid or from adjacent dead matter. Can also
"bite" — if a mouth node overlaps another creature's body node, it deals damage proportional
to the closing force of the jaw (motor torque on adjacent edge). Damage transfers energy
from victim to attacker.

**Gripper nodes:** When two gripper nodes from different creatures come within contact
distance, the neural net can choose to "latch" — a temporary spring is created between them.
Breaking free requires exceeding the latch spring's yield force. This enables:
- Predators grabbing prey
- Prey evolving slippery surfaces (high yield force to latch onto)
- Cooperative transport (if two creatures both benefit from gripping)

**Sensor nodes:** Detect local field values (nutrients, chemicals, light) in their vicinity.
Feed into neural net input layer.

**Photoreceptor nodes:** Detect light intensity and direction (2D). Creatures with multiple
photoreceptors can triangulate light sources — phototaxis evolves naturally.

### 2.3 Force Computation (per timestep)

For each creature, compute forces on all nodes:

1. **Spring forces:** Hooke's law along each edge. `F = -k(|d| - L_rest) * d_hat`
2. **Damping forces:** Velocity-dependent. `F = -c * v_relative_along_edge`
3. **Motor torques:** Neural net outputs desired torque on each motor joint. Converted to
   tangential forces on connected nodes.
4. **Drag forces:** `F_drag = -γ * v_node` (this is the dominant term)
5. **Contact forces:** Short-range repulsion between nodes of different creatures (and same
   creature for self-avoidance). Soft-sphere: `F = k_contact * max(0, r_sum - |d|)^1.5`
6. **Grip forces:** Latch springs between gripping creatures.

Then: `v_node = F_net / γ` (overdamped regime — velocity is instantaneous).

Position update: `x_new = x_old + v * dt`

### 2.4 Energy Budget

Each creature carries an **energy** scalar (float, starts at birth energy).

| Activity             | Energy cost per timestep               |
|----------------------|----------------------------------------|
| Existing (metabolism)| `E_basal * num_nodes` (bigger = more expensive) |
| Motor actuation      | `E_motor * |torque_applied|` per motor |
| Gripping (holding)   | `E_grip * num_active_latches`          |
| Chemical emission    | `E_chem * emission_rate`               |

| Activity             | Energy gain                            |
|----------------------|----------------------------------------|
| Nutrient absorption  | `E_feed * local_nutrient_conc` (via mouth nodes) |
| Photosynthesis       | `E_photo * local_light * num_photoreceptors` |
| Predation            | `victim.energy * transfer_efficiency` on kill |
| Scavenging           | Absorb from detritus grid cells        |

**Death** occurs when energy ≤ 0. The creature's remaining structural energy becomes a
detritus patch on the grid.

**Critical design parameter:** The energy economics **must** make pure photosynthesis barely
viable and predation risky but rewarding. Suggested starting ratio: a predator needs to
consume ~4 autotrophs worth of energy to survive. Tune this — it's the most important
parameter for producing interesting food webs.

---

## 3. The Brain: CTRNN (Continuous-Time Recurrent Neural Network)

### 3.1 Why CTRNN

Standard feedforward nets can't produce rhythmic locomotion without explicit oscillator
circuits. CTRNNs have **intrinsic dynamics** — they can oscillate, have memory, exhibit
transient responses — all from the same architecture. They're the standard choice in
evolutionary robotics for good reason.

### 3.2 CTRNN Equations

Each neuron `i` has a state `s_i` that evolves as:

```
τ_i * ds_i/dt = -s_i + Σ_j (w_ij * σ(s_j + θ_j)) + I_i
```

Where:
- `τ_i` = time constant of neuron i (evolvable, range [0.1, 10.0])
- `w_ij` = connection weight from j to i (evolvable)
- `θ_j` = bias of neuron j (evolvable)
- `σ` = sigmoid activation, `σ(x) = 1 / (1 + exp(-x))`
- `I_i` = external input (sensor values)

Output of neuron i: `o_i = σ(s_i + θ_i)`

Discretized with Euler: `s_i(t+dt) = s_i(t) + dt/τ_i * [-s_i(t) + Σ_j(...) + I_i]`

### 3.3 Network Topology

```
INPUTS (variable per morphology):
  - Per sensor node: local nutrient gradient (dx, dy), chemical A, chemical B
  - Per photoreceptor: light intensity, light direction (dx, dy)
  - Per joint: current angle, angular velocity
  - Per gripper: contact (0/1), grip active (0/1)
  - Global: own energy level (normalized), own age (normalized)

HIDDEN LAYER:
  - N_hidden neurons (evolvable, range [4, 24])
  - Fully connected recurrent (all-to-all within hidden + self-connections)

OUTPUTS (variable per morphology):
  - Per motor joint: desired torque [-1, 1] scaled by motor_torque_max
  - Per gripper: grip/release (thresholded at 0.5)
  - Per mouth: bite force [0, 1]
  - Chemical emission rate [0, 1]
  - Reproduce signal (thresholded, only acts if energy > reproduction_threshold)
```

### 3.4 Genome Encoding

The genome is a flat float vector encoding, in order:

```
[morphology_genes | network_topology_genes | weight_matrix | biases | time_constants | meta_genes]
```

**Morphology genes:**
- Number of nodes (integer, encoded as float, rounded)
- Per node: relative position (polar coords from parent node), radius, type (discretized)
- Per edge: rest length, stiffness, damping, has_motor, motor_max_torque
- Connectivity pattern (which nodes connect — encoded as adjacency probabilities)

**Network genes:**
- Number of hidden neurons
- All w_ij weights
- All θ_j biases  
- All τ_i time constants

**Meta genes:**
- Mutation rate (self-adaptive)
- Mutation step size
- Preferred reproduction mode (asexual bias vs. sexual bias)
- Color (RGB — purely cosmetic but allows visual lineage tracking)

---

## 4. Evolution

### 4.1 No Explicit Fitness Function

There is **no fitness function**. Selection is entirely natural:

- Creatures that acquire enough energy reproduce.
- Creatures that run out of energy die.
- That's it.

This is crucial — an explicit fitness function biases evolution toward the designer's
preconceptions. Natural selection from energy dynamics produces surprising strategies.

### 4.2 Reproduction

**Asexual (fission):**
- Creature signals reproduce when energy > threshold.
- Energy splits ~50/50 between parent and offspring.
- Offspring genome = parent genome + mutations.
- Offspring spawns adjacent to parent.

**Sexual (conjugation):**
- Two creatures with energy > threshold come into physical contact.
- Both signal reproduce simultaneously.
- Offspring genome = crossover of both parents + mutations.
- Energy cost split among both parents.
- Requires **evolved coordination** — this is hard, so it should be rewarding (crossover
  explores the fitness landscape faster).

### 4.3 Mutation Operators

| Operator              | Rate (self-adaptive) | Effect                         |
|-----------------------|---------------------|--------------------------------|
| Weight perturbation   | ~0.05 per weight    | Gaussian noise, σ from meta-gene |
| Bias perturbation     | ~0.05 per bias      | Same                           |
| Time constant perturb | ~0.02 per τ         | Log-normal noise               |
| Add node              | ~0.005 per birth    | Insert node, random connections |
| Remove node           | ~0.003 per birth    | Remove least-connected node    |
| Add edge              | ~0.01 per birth     | Connect two unconnected nodes  |
| Remove edge           | ~0.005 per birth    | Remove random non-bridge edge  |
| Node type change      | ~0.002 per birth    | Mutate node type               |
| Hidden neuron add     | ~0.005 per birth    | Add neuron to hidden layer     |
| Hidden neuron remove  | ~0.003 per birth    | Remove neuron                  |
| Duplication           | ~0.001 per birth    | Duplicate a subgraph           |

### 4.4 Speciation Tracking

Define genomic distance between two creatures:

```
d(A, B) = α * morphology_distance + β * weight_distance + γ * topology_distance
```

Cluster the population periodically using this metric (DBSCAN or similar). Assign species
IDs. Track species populations over time. A new species is "born" when a cluster splits.

---

## 5. Architecture and Performance

### 5.1 Why Not Pure Python

Back-of-envelope calculation for the hot loop:

- 300 creatures × 6 nodes avg = 1800 nodes
- Force computation per node: ~20 FLOPs (springs, drag, contacts)
- Contact detection: O(N²) naive or O(N log N) with spatial hash → ~1800² = 3.24M checks
- CTRNN update per creature: ~N_hidden² ≈ 150 multiplies
- Per timestep total: ~5M FLOPs
- Target: 1000 timesteps/sec for fast-forward → 5 GFLOPS

Pure Python with loops: ~10M FLOPs/sec → **500× too slow**.
NumPy vectorized: ~500M FLOPs/sec → **10× too slow** for the contact detection.
**Rust core + Python orchestration: easily 5+ GFLOPS → comfortable margin.**

### 5.2 Recommended Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND                          │
│        Browser (HTML5 Canvas / WebGL / PixiJS)       │
│  - Renders creature bodies, fields, effects          │
│  - Controls: pause, speed, follow creature, inspect  │
│  - Plots: population, phylogeny, energy stats        │
│  - Communicates via WebSocket                        │
└───────────────┬─────────────────────────────────────┘
                │ WebSocket (JSON or MessagePack frames)
┌───────────────▼─────────────────────────────────────┐
│                 PYTHON LAYER                         │
│  - FastAPI or similar WebSocket server               │
│  - Orchestration: init, start/stop, parameter tuning │
│  - Analytics: phylogeny tree, species clustering,    │
│    population time series, UMAP embeddings           │
│  - Data logging (SQLite or Parquet files)            │
│  - Calls into Rust core via PyO3 bindings            │
└───────────────┬─────────────────────────────────────┘
                │ PyO3 FFI (zero-copy where possible)
┌───────────────▼─────────────────────────────────────┐
│                  RUST CORE                           │
│  - All per-timestep physics (forces, integration)    │
│  - Spatial hash grid for contact detection           │
│  - CTRNN forward pass (batch over all creatures)     │
│  - Energy accounting                                 │
│  - Reproduction / death lifecycle                    │
│  - Genome mutation operators                         │
│  - Grid field diffusion (nutrients, chemicals)       │
│  - Serialization of world state snapshots            │
└─────────────────────────────────────────────────────┘
```

### 5.3 Rust Core Design

```rust
// Core data structures (sketch)

struct Node {
    pos: Vec2,
    vel: Vec2,        // computed each step, not integrated
    radius: f32,
    node_type: NodeType,
    drag_coeff: f32,
}

struct Edge {
    a: usize,         // index into creature's node array
    b: usize,
    rest_length: f32,
    stiffness: f32,
    damping: f32,
    has_motor: bool,
    motor_max_torque: f32,
}

struct CTRNN {
    states: Vec<f32>,
    weights: Vec<Vec<f32>>,  // or flat array for cache efficiency
    biases: Vec<f32>,
    time_constants: Vec<f32>,
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
}

struct Creature {
    id: u64,
    parent_ids: (u64, Option<u64>),  // supports asexual and sexual
    born_at: u64,                     // timestep
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    brain: CTRNN,
    genome: Genome,
    energy: f32,
    species_id: u32,
    color: (u8, u8, u8),
    alive: bool,
}

struct World {
    creatures: Vec<Creature>,
    nutrient_grid: Grid2D<f32>,
    light_grid: Grid2D<f32>,
    chem_a_grid: Grid2D<f32>,
    chem_b_grid: Grid2D<f32>,
    detritus_grid: Grid2D<f32>,
    spatial_hash: SpatialHash,
    tick: u64,
    rng: SmallRng,
    config: SimConfig,
}
```

### 5.4 Spatial Hash for Collision Detection

Divide world into cells of size `2 * max_creature_radius`. Each node registers in its cell.
Contact checks only occur between nodes in the same or adjacent cells. This brings contact
detection from O(N²) to approximately O(N) for uniformly distributed creatures.

### 5.5 PyO3 Bindings

Expose to Python:

```python
# Python-side API (via PyO3)
import animalcula

world = animalcula.World(config)       # create world from config dict
world.seed_creatures(n=100)            # seed initial random creatures

# Run N steps, return snapshot
snapshot = world.step(n=100)
# snapshot contains: all creature positions/shapes, grid field values,
#                    birth/death events, species counts

# Query
tree = world.get_phylogeny()           # full lineage graph
stats = world.get_statistics()         # population, energy, diversity metrics
creature = world.get_creature(id=42)   # inspect specific creature

# Control
world.set_config({"nutrient_rate": 0.5})  # live parameter tuning
world.save("checkpoint.bin")
world.load("checkpoint.bin")
```

### 5.6 Alternative: Numba/NumPy First, Rust Later

If you want to prototype faster before committing to Rust:

1. **Phase 1:** Pure Python + NumPy for structure. Use **Numba @njit** for the force
   computation inner loop and CTRNN update. This gets you ~50-100× over pure Python.
   Good enough for ~100 creatures during development.

2. **Phase 2:** Once the design stabilizes, port the hot path to Rust via PyO3. Keep
   Python for orchestration and analytics.

This avoids premature optimization while still being playable during development.

---

## 6. Visualization and Display

### 6.1 Option A: Browser Frontend (Recommended)

**Stack:** Python backend (FastAPI + uvicorn) → WebSocket → Browser (PixiJS or raw WebGL)

**Implementation policy:** this is the product-facing frontend path and should begin as soon as
the simulation can emit stable live snapshots. Local Tk/Pygame-style viewers remain useful
debug fallbacks, but they are not the long-term UI target and should not receive major product
features that would later be reimplemented in the browser.

**Why browser:**
- PixiJS makes 2D rendering with thousands of sprites trivial and GPU-accelerated.
- WebGL shaders enable the microscope aesthetic (glow, blur, caustics) cheaply.
- UI controls (sliders, plots) are easy with standard HTML/JS.
- Runs on any machine, shareable via URL.
- Recharts or D3 for live analytics plots.

**Data flow (per frame at ~30fps):**

```
Backend                                    Frontend
   │                                          │
   │  ◄── request_frame ──────────────────    │
   │                                          │
   │  ── frame_data (MessagePack) ────────►   │
   │     {                                    │
   │       creatures: [{                      │
   │         id, nodes: [{x,y,r,type}],       │
   │         edges: [{a,b,width}],            │
   │         color, energy, species_id        │
   │       }, ...],                           │
   │       fields: {                          │
   │         nutrients: [grid as flat f32],    │
   │         light: [grid as flat f32]        │
   │       },                                 │
   │       stats: {tick, pop, births, deaths} │
   │     }                                    │
   │                                          │
   │  ◄── user_command ──────────────────     │
   │      {action: "set_speed", value: 10}    │
```

### 6.2 Option B: Pygame / Arcade (Simpler, Local Only)

Good for rapid prototyping. Pygame can handle ~2000 draw calls at 60fps which is enough
for 300 creatures with 6 segments each. No network overhead. But no fancy shaders
and harder to add UI controls.

### 6.3 Visual Design: The Microscope Aesthetic

**Creature rendering:**
- Each segment (edge) drawn as a tapered rounded rectangle, slightly translucent.
- Nodes drawn as circles — mouth nodes red-tinted, grippers have small "claw" marks,
  photoreceptors have a bright dot.
- Creature body has a subtle internal glow (lighter center, darker edges).
- Color inherited from lineage (genome color gene) — allows visual species tracking.
- When gripping: draw a thin line between the latched grippers, pulsing.

**Environment rendering:**
- Nutrient field as a soft green haze (low-resolution texture, bilinear interpolated).
- Light field as a warm yellow gradient.
- Chemical fields as faint colored clouds (blue, purple).
- Detritus as small brownish specks.
- Background: dark navy/black with subtle noise texture (dark-field microscopy look).
- Optional: animated caustic light pattern overlay (shader effect).

**Camera:**
- Default: overview of entire world.
- Click creature: camera follows it, zooms in slightly.
- Scroll: zoom in/out (continuous).
- When zoomed in close, individual segment springs and motor activity become visible.

### 6.4 UI Layout: The Control Room

The browser interface is divided into a main viewport and collapsible side panels. The
design principle: **the world view always dominates** — panels slide in/out and never
cover more than 40% of screen width. Dark theme throughout (matches microscope aesthetic).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ▶ ⏸ ⏩ x1 x10 x100 x1000  │ Tick: 142,857  Pop: 287  Species: 6  │ ⚙ 🔬 📊 │
├──────────────────────────────────────────────────┬──────────────────────────┤
│                                                  │ ┌──────────────────────┐ │
│                                                  │ │   ACTIVE PANEL       │ │
│                                                  │ │   (one of):          │ │
│                                                  │ │                      │ │
│              WORLD VIEWPORT                      │ │   🔬 Inspector       │ │
│              (WebGL canvas)                      │ │   ⚙  Tuning         │ │
│                                                  │ │   📊 Analytics       │ │
│              click creature = select             │ │   🧬 Genome         │ │
│              scroll = zoom                       │ │   🧠 Brain          │ │
│              drag = pan                          │ │                      │ │
│              double-click = follow               │ │                      │ │
│                                                  │ │                      │ │
│                                                  │ └──────────────────────┘ │
├──────────────────────────────────────────────────┴──────────────────────────┤
│  TIMELINE BAR (mini population sparkline + event markers, scrubable)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Top Bar: Transport Controls and Status

```
[▶ Play] [⏸ Pause] [⏩ Speed: x1 | x10 | x100 | x1000 | MAX]
[⏮ Step Back] [⏭ Step Forward]   ← when paused, single-step through ticks

Tick: 142,857 | Pop: 287 | Births: 12/s | Deaths: 9/s | Species: 6
FPS: 60 | Sim rate: 4,200 ticks/s | [⚙] [🔬] [📊]
```

- **Speed slider** is continuous, not just preset steps. Drag from x1 to xMAX.
  At high speed, rendering drops to every Nth frame (adaptive frame skip).
- **Step forward/back** when paused: advance exactly 1 physics tick. Essential for
  debugging creature behavior. "Step back" requires checkpoint buffering (keep last
  ~100 ticks in memory as ring buffer).
- The three icons toggle the right-side panels.

### 6.6 🔬 Creature Inspector Panel

Activated by clicking any creature in the viewport. The selected creature gets a
highlight ring. The panel shows everything about it:

```
┌─ 🔬 CREATURE INSPECTOR ──────────────────────┐
│                                                │
│  ┌────────┐   #4201 "Blue Undulator"           │
│  │ mini   │   Species: Clade-17 (Blue)         │
│  │ body   │   Generation: 847                  │
│  │ diagram│   Parent: #3902                    │
│  └────────┘   Born: tick 139,436               │
│               Age: 3,421 ticks                 │
│               Children: 12                     │
│                                                │
│  ── VITALS ────────────────────────────────    │
│  Energy:    ████████░░ 67%  (31.2 / 50.0)     │
│  Health:    ██████████ 100%                    │
│  Speed:     2.3 units/tick                     │
│  Heading:   → 47°                              │
│                                                │
│  ── BODY ──────────────────────────────────    │
│  Nodes: 5  (2 body, 1 mouth, 1 gripper,       │
│             1 photoreceptor)                   │
│  Edges: 6  (4 motorized)                      │
│  Body mass: 3.2  │  Drag: 5.0                 │
│                                                │
│  ── ENERGY BUDGET (last 100 ticks) ────────   │
│  Income:  ████████  feeding: 0.08/tick         │
│  Expense: █████     basal: 0.005/tick          │
│           ███       motors: 0.003/tick         │
│  Net:     +0.072/tick                          │
│                                                │
│  ── BEHAVIOR (last 500 ticks) ─────────────   │
│  Distance traveled: 847 units                  │
│  Kills: 2  │  Times attacked: 5               │
│  Feeding time: 62%  │  Moving: 38%            │
│  Reproduction attempts: 3 (2 successful)       │
│                                                │
│  ── SENSORS (live) ────────────────────────   │
│  Nutrient gradient: → (0.3, -0.1)             │
│  Light: 0.72  direction: ↗                    │
│  Chem A: 0.04  │  Chem B: 0.00               │
│  Contact: gripping creature #4318              │
│                                                │
│  [🧠 View Brain]  [🧬 View Genome]            │
│  [👁 Follow]  [📌 Pin to compare]             │
│  [💀 Kill]  [⚡ Boost energy]  [🔄 Clone]     │
└────────────────────────────────────────────────┘
```

**Key interactions:**
- **Follow** — camera locks onto this creature, tracks it as it moves.
- **Pin** — pin up to 3 creatures for side-by-side comparison.
- **Kill / Boost / Clone** — god-mode interventions for testing. Kill removes it,
  Boost sets energy to max, Clone spawns an identical copy nearby.
- The **mini body diagram** is a live-rendered schematic of the creature's node graph,
  color-coded by node type, with motor activity shown as pulsing edges.

### 6.7 🧠 Brain Viewer Panel

Opened from the Inspector's "View Brain" button. Shows the CTRNN in real-time:

```
┌─ 🧠 BRAIN: Creature #4201 ───────────────────┐
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │         NETWORK GRAPH (live)              │  │
│  │                                           │  │
│  │    [S1]─┐    ┌─[H1]──[H2]─┐    ┌─[M1]   │  │
│  │    [S2]─┼────┤  ↕  ↗  ↕   ├────┤ [M2]   │  │
│  │    [S3]─┤    └─[H3]──[H4]─┘    ├─[M3]   │  │
│  │    [S4]─┘       ↕    ↕         └─[M4]   │  │
│  │              [H5]──[H6]           [G1]   │  │
│  │                                           │  │
│  │  ● node size = activation (σ(s+θ))        │  │
│  │  ● node color = blue(low) → red(high)     │  │
│  │  ● edge width = |weight|                  │  │
│  │  ● edge color = green(excitatory)          │  │
│  │                  red(inhibitory)           │  │
│  │  ● edge pulse = current signal flow       │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ── NEURON DETAIL (hover any node) ────────   │
│  Neuron H3:                                    │
│  State s = -1.42  │  Output σ = 0.19          │
│  Bias θ = 0.83    │  Time const τ = 2.40      │
│  Inputs: S2(w=1.3), H1(w=-0.7), H4(w=0.9)   │
│                                                │
│  ── INPUT/OUTPUT TRACES ───────────────────   │
│  [scrolling time-series chart, last 200 ticks] │
│                                                │
│  Inputs:  S1 ~~~∿∿∿~~~  S2 ──∿──∿──          │
│  Hidden:  H1 ∿∿∿∿∿∿∿∿  H3 ──────∿∿          │
│  Outputs: M1 ∿∿∿∿∿∿∿∿  M2 ∿∿∿∿∿∿∿∿          │
│           G1 ▁▁▁▁█▁▁▁▁  (grip fired!)        │
│                                                │
│  ── OSCILLATION ANALYSIS ──────────────────   │
│  Dominant frequency: 0.23 Hz (locomotion gait) │
│  Motor phase diagram: [polar plot]             │
│                                                │
│  [📊 Weight Matrix Heatmap]                    │
│  [📈 Full Trace Export (CSV)]                  │
└────────────────────────────────────────────────┘
```

**What makes this useful, not just pretty:**
- The **scrolling traces** let you see rhythmic motor patterns (locomotion gaits show
  as clean oscillations, feeding shows as bursts).
- The **oscillation analysis** auto-detects the dominant frequency — if a creature has
  evolved rhythmic swimming, you'll see a clear peak.
- The **weight matrix heatmap** shows the full connectivity at a glance — dense
  clusters suggest functional modules.
- The live animation of **signal flow** (edges pulse when a signal propagates) makes
  it intuitive to see how sensory input drives motor output.

### 6.8 🧬 Genome Viewer Panel

```
┌─ 🧬 GENOME: Creature #4201 ──────────────────┐
│                                                │
│  Genome length: 247 floats                     │
│  Generation: 847  │  Mutations from parent: 14 │
│                                                │
│  ── MORPHOLOGY GENES ──────────────────────   │
│  Nodes: 5  │  Edges: 6                        │
│  [interactive body plan editor view]           │
│  Highlight mutations from parent in ORANGE     │
│                                                │
│  ── BRAIN TOPOLOGY GENES ──────────────────   │
│  Hidden neurons: 6                             │
│  Total connections: 42                         │
│  Total parameters: 96                          │
│                                                │
│  ── META GENES ────────────────────────────   │
│  Mutation rate: 0.047 (self-adaptive)          │
│  Mutation σ: 0.28                              │
│  Reproduction mode: 0.8 (mostly asexual)       │
│  Color: RGB(45, 120, 200)                      │
│                                                │
│  ── LINEAGE ───────────────────────────────   │
│  ← Parent #3902 (gen 846)                      │
│  ← Grandparent #3511 (gen 845)                 │
│  ← ... [expand full ancestry]                  │
│                                                │
│  ── GENOME DIFF vs PARENT ─────────────────   │
│  [color-coded bar: unchanged=gray,             │
│   mutated=orange, structural=red]              │
│  14 point mutations, 0 structural mutations    │
│                                                │
│  [📋 Copy Genome] [💾 Save to file]            │
│  [🔬 Find relatives] [📊 Compare with...]     │
└────────────────────────────────────────────────┘
```

### 6.9 ⚙ Live Tuning Panel

This is the parameter control room. **Changes apply immediately** to the running
simulation — no restart needed. This is how you interactively find interesting regimes.

```
┌─ ⚙ LIVE TUNING ──────────────────────────────┐
│                                                │
│  ── ENERGY ECONOMICS ──────────────────────   │
│  Basal cost/node:     ●───────○──── 0.001     │
│  Motor cost/torque:   ●────○─────── 0.0005    │
│  Feed rate:           ●─────────○── 0.01      │
│  Photosynthesis rate: ●──────○───── 0.005     │
│  Predation transfer:  ●────────○─── 0.6       │
│  Repro threshold:     ●───────○──── 100       │
│  Birth energy:        ●─────○────── 50        │
│                                                │
│  ── ENVIRONMENT ───────────────────────────   │
│  Nutrient source strength: ●──────○── 2.0     │
│  Nutrient diffusion:       ●───○───── 0.1     │
│  Light intensity:          ●────────○ 1.0     │
│  Light direction:          [compass dial]      │
│  Chemical decay:           ●──○────── 0.05    │
│                                                │
│  ── PHYSICS ───────────────────────────────   │
│  Global drag:          ●──────○────── 1.0     │
│  Contact stiffness:    ●─────○─────── 500     │
│  Grip yield force:     ●───────○───── 50      │
│                                                │
│  ── EVOLUTION ─────────────────────────────   │
│  Base mutation rate:   ●─────○─────── 0.05    │
│  Structural mut rate:  ●──○────────── 0.005   │
│                                                │
│  ── POPULATION ────────────────────────────   │
│  Min population:       ●─○──────────── 20     │
│  Max population:       ●─────────○──── 500    │
│  [  ] Runaway protection (auto-perturb)        │
│                                                │
│  ── GOD MODE ──────────────────────────────   │
│  [☄ Meteor event] — kill 50% random            │
│  [🌊 Flood nutrients] — 10x nutrients 1000t    │
│  [🌑 Eclipse] — light → 0 for 5000 ticks      │
│  [🧬 Inject species] — load genome from file   │
│  [🏗 Add obstacle] — click to place wall       │
│  [🦠 Plague] — random energy drain event       │
│                                                │
│  ── PRESETS ───────────────────────────────   │
│  [Load preset ▼] [Save current as preset]      │
│  • Default balanced                            │
│  • Predator paradise (high transfer, low cost) │
│  • Harsh world (low nutrients, high cost)      │
│  • Cambrian explosion (high mutation, many     │
│    nutrients)                                  │
│                                                │
│  ── PARAMETER HISTORY ─────────────────────   │
│  [scrollable log of all changes with tick#]    │
│  tick 140,200: feed_rate 0.01 → 0.015          │
│  tick 138,500: light_intensity 1.0 → 0.3       │
│  [↩ Undo last] [📋 Export change log]          │
└────────────────────────────────────────────────┘
```

**Design principles for live tuning:**
- Every slider has a **sensible range** with soft limits (drag beyond for extreme
  values) and a **reset-to-default** button (double-click the label).
- Changes are **logged with timestamps** so you can correlate parameter tweaks with
  population events later in analysis.
- **Presets** let you instantly jump between known-interesting parameter regimes.
- **God mode events** let you stress-test the ecosystem: does it recover from a
  mass extinction? What happens if you flood nutrients? These are the most fun
  interactive moments and drive real insight into ecosystem resilience.

### 6.10 📊 Analytics Panel

Live-updating charts. These run alongside the simulation and are the main way to tell
whether something interesting is happening without staring at the viewport.

```
┌─ 📊 ANALYTICS ────────────────────────────────┐
│                                                │
│  [Tab: Population | Energy | Phylogeny |       │
│        Phenotype | Trophic | Events]           │
│                                                │
│  ── POPULATION TAB ────────────────────────   │
│  [stacked area chart: species populations      │
│   over time, colored by species, x=tick]       │
│                                                │
│  Total: 287 │ Autotroph: 152 │ Herbivore: 98  │
│  Predator: 37 │ Shannon H': 1.42              │
│                                                │
│  ── ENERGY TAB ────────────────────────────   │
│  [line chart: mean energy by species]          │
│  [line chart: total energy in system           │
│   (creatures + fields + detritus)]             │
│  Energy conservation check: ✓ (±0.01%)         │
│                                                │
│  ── PHYLOGENY TAB ─────────────────────────   │
│  [horizontal dendrogram: time on x-axis        │
│   branches = lineages, color = species          │
│   width = population, extinct = faded]          │
│  Click any branch → select creature in world   │
│  Zoom: mouse wheel │ Pan: drag                 │
│                                                │
│  ── PHENOTYPE TAB ─────────────────────────   │
│  [2D UMAP scatter: each dot = creature          │
│   color = species, size = energy                │
│   updates every 5000 ticks]                     │
│  Clusters = species; splits = speciation        │
│                                                │
│  ── TROPHIC TAB ───────────────────────────   │
│  [Sankey diagram: energy flow                   │
│   light/nutrients → autotrophs → herbivores     │
│   → predators → detritus → nutrients]           │
│  [pie chart: biomass by trophic level]          │
│                                                │
│  ── EVENTS TAB ────────────────────────────   │
│  [scrolling event log with filters]             │
│  tick 142,801: SPECIATION — Clade-17 split      │
│  tick 142,793: KILL — #4201 ate #4318           │
│  tick 142,650: EXTINCTION — Clade-12 died out   │
│  tick 142,200: PARAM — feed_rate → 0.015        │
│  [filter: births|deaths|kills|speciation|       │
│           extinction|params]                    │
│                                                │
│  [📈 Export all data (Parquet)]                 │
│  [📸 Screenshot with overlay]                   │
└────────────────────────────────────────────────┘
```

### 6.11 Timeline Bar (Bottom)

A thin horizontal bar spanning the full window width at the bottom:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ◄ ░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ► │
│   0                    ▲ speciation     ▲ param change    142,857   │
│                        ▼ extinction     ★ mass extinction           │
└─────────────────────────────────────────────────────────────────────┘
```

- **Mini sparkline** of population over all time.
- **Event markers** (triangles, stars) for speciation, extinction, parameter changes,
  god-mode events.
- **Scrubable** — drag the playhead to jump to any checkpoint. Combined with periodic
  checkpoint saves, this is the "time machine."
- Click a marker → jump to that tick and open the event in the Events log.

### 6.12 Keyboard Shortcuts

| Key       | Action                                    |
|-----------|-------------------------------------------|
| `Space`   | Play / Pause toggle                       |
| `+` / `-` | Speed up / slow down (×2 per press)      |
| `.`       | Single step forward (when paused)         |
| `,`       | Single step backward (when paused)        |
| `F`       | Follow selected creature                  |
| `Escape`  | Deselect / unfollow                       |
| `Tab`     | Cycle through panel tabs                  |
| `1-6`     | Quick switch to panel tab 1-6             |
| `G`       | Toggle grid overlay                       |
| `N`       | Toggle nutrient field visibility          |
| `L`       | Toggle light field visibility             |
| `C`       | Toggle chemical field visibility          |
| `Home`    | Reset camera to world overview            |
| `S`       | Screenshot (saves PNG + state)            |
| `Ctrl+S`  | Save checkpoint                           |
| `Ctrl+Z`  | Undo last parameter change                |

### 6.13 WebSocket Command Protocol

All UI interactions send JSON commands to the Python backend over WebSocket:

```typescript
// Frontend → Backend commands
{cmd: "set_speed", value: 100}                    // ticks per frame
{cmd: "pause"}
{cmd: "resume"}
{cmd: "step", n: 1}                               // single step
{cmd: "step_back", n: 1}                           // requires ring buffer
{cmd: "set_param", path: "energy.feed_rate", value: 0.015}
{cmd: "select_creature", id: 4201}
{cmd: "god_event", type: "meteor", params: {kill_fraction: 0.5}}
{cmd: "save_checkpoint", filename: "interesting_moment.bin"}
{cmd: "load_checkpoint", filename: "interesting_moment.bin"}
{cmd: "inject_genome", genome_file: "seeds.bin"}
{cmd: "get_creature_detail", id: 4201}            // full brain + genome
{cmd: "get_brain_traces", id: 4201, ticks: 200}   // neuron time series
{cmd: "export_data", format: "parquet"}

// Backend → Frontend frames (per render tick)
{
  type: "frame",
  tick: 142857,
  creatures: [{id, nodes, edges, color, energy, species_id, selected}],
  fields: {nutrients: [...], light: [...]},
  stats: {pop, births_sec, deaths_sec, species_count, sim_rate},
  events: [{tick, type, detail}]  // events since last frame
}

// Backend → Frontend (on creature select)
{
  type: "creature_detail",
  id: 4201,
  body: {nodes: [...], edges: [...]},
  brain: {weights: [...], biases: [...], taus: [...], states: [...],
          inputs: [...], outputs: [...]},
  genome: {raw: [...], mutations_from_parent: [...]},
  lineage: {parent_id, grandparent_id, generation, born_at},
  stats: {age, energy, speed, kills, children, ...}
}
```

---

## 7. Observation and Analytics

### 7.1 Data Logging

Every N timesteps (configurable, default 100), log to SQLite or Parquet:

**Events table:**
```
tick | event_type | creature_id | parent_id(s) | species_id | energy | genome_hash
```
Event types: `birth`, `death`, `speciation`, `predation_kill`, `reproduction`

**Population snapshots table:**
```
tick | species_id | count | mean_energy | mean_size | mean_speed | genome_centroid
```

**Full genome archive** (less frequent, every 1000 ticks):
```
tick | creature_id | genome_blob (compressed)
```

### 7.2 Phylogenetic Tree

Constructed from the events table. Each creature has parent pointer(s). The tree is a
directed acyclic graph (DAG for sexual reproduction, tree for asexual).

**Visualization:** horizontal dendrogram. X-axis = time. Each branch = a lineage.
Color = species. Branch width = population count. Extinct branches fade to gray.
This should update live during simulation.

Library options: D3.js (browser), or export to Newick format for external phylogenetics
tools (FigTree, iTOL).

### 7.3 Phenotype Space (UMAP)

Periodically (every 5000 ticks), take the full population, extract phenotype vectors:

```
phenotype = [num_nodes, num_edges, mean_segment_length, num_mouths, num_grippers,
             num_photoreceptors, body_aspect_ratio, mean_speed_last_100_ticks,
             energy_intake_rate, reproduction_rate, ...]
```

Run UMAP (or t-SNE) on these vectors. Plot as a 2D scatter, colored by species.
Over time, you see clusters form, merge, split — this is speciation visualized in
phenotype space.

### 7.4 Key Metrics Dashboard

Live-updating plots:

1. **Population over time** (stacked area chart by species)
2. **Mean energy by species** (line chart)
3. **Diversity index** (Shannon entropy of species distribution)
4. **Trophic structure** (pie chart: autotrophs / herbivores / predators)
5. **Morphological complexity** (mean node count over time)
6. **Predation rate** (kills per 1000 ticks)
7. **Innovation events** (new species per epoch, structural mutations that spread)

### 7.5 "Time Machine" Replay

Since the Rust core can serialize world state, save periodic checkpoints. Allow the user
to scrub backwards in time and replay from any checkpoint. Combined with the phylogenetic
tree, this lets you answer: "when did predators first evolve? Let me go back and watch."

---

## 8. Seeding and Bootstrapping

### 8.1 The Cold Start Problem

Random genomes produce creatures that can't move, can't feed, and die immediately.
Evolution needs a viable founding population. Solutions:

**Option A: Pre-evolved seeds.**
Run a separate "nursery" simulation with simplified physics (just locomotion + feeding,
no competition) and evolve a few viable body plans. Use these as the founding population.

**Option B: Generous initial conditions.**
Start with extremely high nutrient density and low metabolic cost. Gradually increase
difficulty over the first 10,000 ticks ("primordial soup" phase). This gives random
creatures time to stumble onto viable strategies before selection pressure ramps up.

**Option C: Hand-designed starter archetypes (recommended for first run).**
Create 3-4 minimal viable creatures by hand:
- **"Alga"**: 2 nodes, 1 photoreceptor, no motor. Just sits and photosynthesizes.
- **"Worm"**: 4 nodes in a line, 3 motors, undulatory motion, 1 mouth. Basic grazer.
- **"Amoeba"**: 3 nodes in triangle, 3 motors, 1 mouth, 1 gripper. Basic predator.

Seed ~50 of each with slight random variation. Let evolution take over from there.

### 8.2 Stability Safeguards

- **Minimum population:** If total population drops below 20, spawn random viable creatures
  (immigration from "off-screen").
- **Maximum population:** If >500 creatures, increase metabolic cost globally (environmental
  carrying capacity).
- **Runaway detection:** If any single species exceeds 80% of population for >5000 ticks,
  introduce an environmental perturbation (nutrient shift, obstacle change).

---

## 9. Interesting Emergent Phenomena to Watch For

Based on similar artificial life systems (Tierra, Avida, Framsticks, NEAT creatures),
here's what **should** emerge if the parameters are right:

1. **Locomotion gaits:** Undulation, paddling, spinning, "walking" on walls.
   Different morphologies will converge on different solutions.

2. **Predator-prey arms race:** Speed, armor (high stiffness), evasion maneuvers,
   ambush strategies (sit still, wait, grab).

3. **Phototropism:** Creatures evolving to move toward light, and shade-adapted
   species evolving to avoid competition by seeking darker regions.

4. **Chemical signaling:** Alarm pheromones (emit chemical when attacked → nearby
   conspecifics flee), trail pheromones (mark food sources).

5. **Parasitism:** Small creatures that grip onto large ones and siphon energy
   without killing them.

6. **Symbiosis:** Two species that co-locate because one produces chemical the other
   needs (possible if chemical fields serve as nutrients for some morphologies).

7. **Boom-bust cycles:** Classic predator-prey oscillations (Lotka-Volterra dynamics).

8. **Mass extinctions and adaptive radiations:** Environmental shifts wipe out
   dominant species, opening niches for rapid diversification of survivors.

9. **Evolutionary stasis punctuated by bursts:** Long periods of stability broken by
   sudden innovation (new body plan, new strategy).

10. **Red Queen dynamics:** Continuous co-evolutionary escalation between predators
    and prey, with no stable equilibrium.

---

## 10. Configuration and Tuning

### 10.1 Master Configuration File (YAML)

```yaml
world:
  width: 1000.0
  height: 1000.0
  grid_resolution: 5.0          # field grid cell size
  boundary: "toroidal"          # or "walled"

physics:
  dt: 0.01
  default_drag: 1.0
  contact_stiffness: 500.0
  contact_damping: 10.0
  grip_spring_stiffness: 200.0
  grip_yield_force: 50.0        # force to break a grip

environment:
  nutrient_diffusion_rate: 0.1
  nutrient_source_count: 5
  nutrient_source_strength: 2.0
  nutrient_decay_rate: 0.001
  light_intensity_max: 1.0
  light_direction: [1.0, 0.0]
  chemical_diffusion_rate: 0.2
  chemical_decay_rate: 0.05
  season_cycle_ticks: 10000
  epoch_cycle_ticks: 100000

energy:
  basal_cost_per_node: 0.001
  motor_cost_per_torque: 0.0005
  grip_cost: 0.002
  feed_rate: 0.01
  photosynthesis_rate: 0.005
  predation_transfer_efficiency: 0.6
  reproduction_threshold: 100.0
  birth_energy: 50.0

creatures:
  max_nodes: 12
  max_population: 500
  min_population: 20

brain:
  hidden_neurons_range: [4, 24]
  weight_range: [-5.0, 5.0]
  tau_range: [0.1, 10.0]

evolution:
  weight_mutation_rate: 0.05
  weight_mutation_sigma: 0.3
  structural_mutation_rate: 0.005
  meta_mutation_rate: 0.01

visualization:
  target_fps: 30
  websocket_port: 8765
  default_speed: 1            # ticks per frame
  max_speed: 1000
```

### 10.2 Tuning Strategy

The most critical parameters to get right, in order:

1. **Energy economics** (section 2.4 tables) — determines whether any trophic level
   beyond autotrophs is viable.
2. **Nutrient source strength vs. basal cost** — determines carrying capacity.
3. **Reproduction threshold vs. birth energy** — determines generation time.
4. **Mutation rates** — too high = no stable species; too low = no adaptation.
5. **Grip yield force** — determines whether predation is viable.

**Approach:** Start with autotrophs only, tune until stable population ~200. Then
introduce mobile herbivores, tune until they coexist with autotrophs. Then enable
predation, tune until a 3-level food web stabilizes.

---

## 11. Headless Mode and Parameter Search

This is arguably the most important operational mode. You will spend far more time
running headless parameter sweeps than watching the pretty visualizer. The architecture
must treat headless as the **primary** mode, not an afterthought.

### 11.1 Headless Execution

The simulation core (Rust or Numba) has zero rendering dependencies. The Python
orchestration layer runs it like this:

```python
import animalcula

config = animalcula.Config.from_yaml("config.yaml")
world = animalcula.World(config)
world.seed_creatures(n=100)

# Run 1 million ticks, logging every 1000
for epoch in range(1000):
    world.step(n=1000)
    stats = world.get_statistics()
    log_to_parquet(stats)
    
    if world.population_count == 0:
        print(f"Extinction at tick {world.tick}")
        break

world.save("checkpoint.bin")
```

Without rendering overhead, the target is **10,000+ ticks/sec** on the Rust backend
(~10× faster than the visualized mode). On the Numba prototype, expect ~500-1000
ticks/sec headless — still very usable for parameter exploration.

### 11.2 Speed Tricks for Headless Mode

| Technique                        | Speedup    | Trade-off                        |
|----------------------------------|------------|----------------------------------|
| Skip field diffusion grid update | ~1.5×      | Nutrients less smooth, fine for tuning |
| Larger dt (0.02 → 0.05)         | ~2.5×      | Slightly less stable springs     |
| Reduce contact check frequency  | ~1.3×      | Occasional overlap glitches      |
| Disable chemical fields entirely | ~1.2×      | No pheromone dynamics            |
| Reduce CTRNN hidden neurons cap | ~1.5×      | Simpler brains, faster to eval   |
| **Combined "turbo" mode**        | **~5-8×**  | Good enough for parameter search |

Implement as a single flag: `world = animalcula.World(config, turbo=True)`

### 11.3 Parameter Sweep Framework

The critical use case: systematically explore parameter space to find regimes that
produce interesting dynamics (stable multi-trophic food webs, arms races, speciation).

```python
import itertools
from multiprocessing import Pool

# Define parameter grid
sweep = {
    "energy.basal_cost_per_node": [0.0005, 0.001, 0.002, 0.004],
    "energy.feed_rate":           [0.005, 0.01, 0.02],
    "energy.photosynthesis_rate": [0.002, 0.005, 0.01],
    "energy.reproduction_threshold": [50, 100, 200],
}

def run_trial(params, trial_id):
    config = animalcula.Config.from_yaml("base_config.yaml")
    config.update(params)
    world = animalcula.World(config, turbo=True)
    world.seed_creatures(n=100)
    
    world.step(n=500_000)  # run 500k ticks
    
    return {
        "trial_id": trial_id,
        "params": params,
        "final_population": world.population_count,
        "species_count": world.species_count,
        "max_trophic_level": world.max_trophic_level,
        "diversity_index": world.shannon_diversity,
        "extinction_events": world.extinction_count,
        "had_predation": world.predation_kill_count > 0,
    }

# Run all combinations in parallel
combos = [dict(zip(sweep.keys(), v)) 
          for v in itertools.product(*sweep.values())]
with Pool(processes=8) as pool:
    results = pool.starmap(run_trial, 
                           [(c, i) for i, c in enumerate(combos)])
```

This gives you a table of ~108 parameter combinations × outcome metrics. Plot heatmaps
to find the "interesting" region, then narrow down.

### 11.4 What "Interesting" Means (Health Metrics)

A simulation run is scored by these automated health metrics:

| Metric                    | Boring              | Interesting              |
|---------------------------|---------------------|--------------------------|
| Final population          | 0 (extinction) or max cap | 30-80% of carrying capacity |
| Species count at end      | 1 (monoculture)     | 3+ distinct species      |
| Population variance       | 0 (flatline)        | High (boom-bust cycles)  |
| Trophic levels            | 1 (only autotrophs) | 2-3 (food web)           |
| Predation events          | 0                   | Sustained but not total  |
| Speciation events         | 0                   | Multiple over run        |
| Longest species lifespan  | = total runtime     | < runtime (turnover)     |

Compute a composite "interestingness score" from these and use it to rank parameter
combinations. The best candidates then get longer runs with full logging.

### 11.5 Automated Nursery (Pre-Evolution of Viable Seeds)

Before running the full ecosystem, use headless mode to evolve viable starter creatures:

```python
# Nursery: evolve a worm that can actually swim and feed
config = animalcula.Config.from_yaml("nursery_config.yaml")  
# nursery_config: very high nutrients, no predation, no competition
# just: can you move? can you eat?

world = animalcula.World(config, turbo=True)
world.seed_random_creatures(n=500, archetype="worm")

for gen in range(10_000):
    world.step(n=100)
    
# Extract the fittest survivors
survivors = world.get_top_creatures(n=20, metric="energy")
animalcula.save_genomes(survivors, "viable_worms.bin")
```

Then use these pre-evolved genomes as seeds for the full simulation. This solves the
cold-start problem much faster than hoping random genomes stumble into viability.

### 11.6 CLI Interface

For server/cluster use:

```bash
# Single run
animalcula run --config config.yaml --ticks 1000000 --turbo --out results/

# Parameter sweep
animalcula sweep --config base.yaml --sweep sweep.yaml --parallel 8 --out sweep_results/

# Resume from checkpoint  
animalcula run --resume checkpoint.bin --ticks 500000

# Quick stats from a completed run
animalcula report results/

# Export best genomes for seeding new runs
animalcula extract-genomes results/ --top 20 --out seeds.bin
```

---

## 12. Project Roadmap

### Phase 1: Physics Sandbox (Week 1-2)
- [ ] Implement spring-mass creature in Python (NumPy + Numba).
- [ ] Overdamped integration, basic forces.
- [ ] Hand-coded 3-4 morphologies.
- [ ] Simple pygame visualization.
- [ ] Verify creatures can swim when motors oscillate.

### Phase 2: Brains (Week 3)
- [ ] CTRNN implementation.
- [ ] Wire sensors → brain → motors.
- [ ] Verify: random brain occasionally produces locomotion.
- [ ] Implement energy model, feeding from nutrient grid.

### Phase 3: Evolution + Headless Tuning (Week 4-5)
- [ ] Genome encoding, mutation operators.
- [ ] Asexual reproduction.
- [ ] Death from energy depletion.
- [ ] **Headless mode with turbo flag (no rendering deps in core loop).**
- [ ] **Parameter sweep script (multiprocessing over config grid).**
- [ ] **Automated health metrics (interestingness scoring).**
- [ ] **Nursery mode: pre-evolve viable seed genomes.**
- [ ] Run first evolutionary experiment (~10k generations).
- [ ] Basic population tracking and logging (Parquet/SQLite).

### Phase 4: Predation and Complexity (Week 6-7)
- [ ] Gripper mechanics, latch springs.
- [ ] Mouth damage model, energy transfer.
- [ ] Chemical emission and sensing.
- [ ] Sexual reproduction.
- [ ] Environmental variation (seasons, epochs).

### Phase 5: Rust Port (Week 8-10)
- [ ] Port force computation, CTRNN, spatial hash to Rust.
- [ ] PyO3 bindings.
- [ ] Benchmark: target 1000 steps/sec with 300 creatures (10k+ headless).
- [ ] Checkpoint save/load.
- [ ] CLI (`animalcula run`, `animalcula sweep`, `animalcula report`).

### Phase 6: Visualization (Week 10-12)
- [ ] Browser frontend with PixiJS/WebGL.
- [ ] WebSocket data pipeline.
- [ ] Creature rendering with microscope aesthetic.
- [ ] Field visualization (nutrients, light, chemicals).
- [ ] UI controls (speed, pause, inspect).

Implementation note:
Begin the browser/frontend path as soon as live snapshot transport exists; do not wait for the
entire Rust port or all analytics work to finish before starting the permanent UI.

### Phase 7: Analytics (Week 12-14)
- [ ] Phylogenetic tree construction and visualization.
- [ ] Species clustering (DBSCAN on genomes).
- [ ] UMAP phenotype space.
- [ ] Live dashboard with population plots.
- [ ] Time machine replay from checkpoints.

### Phase 8: Polish and Experiments (Ongoing)
- [ ] Parameter sweeps for interesting regimes.
- [ ] Shader effects (caustics, depth of field, glow).
- [ ] Sound design? (Map population dynamics to ambient audio.)
- [ ] Record and share particularly interesting evolutionary runs.

---

## 13. Technology Stack Summary

| Component             | Technology                   | Rationale                        |
|-----------------------|------------------------------|----------------------------------|
| Physics engine        | Rust (+ rayon for threading) | Performance critical inner loop  |
| Neural nets (CTRNN)   | Rust                         | Called millions of times/sec     |
| Spatial indexing       | Rust (custom spatial hash)   | Performance critical             |
| Python bindings        | PyO3 / maturin              | Seamless Python ↔ Rust           |
| Orchestration          | Python (FastAPI)             | Flexibility, rapid iteration     |
| Analytics              | Python (NumPy, SciPy, UMAP) | Ecosystem strength               |
| Data storage           | SQLite + Parquet             | Events + time series             |
| Frontend rendering     | PixiJS (WebGL)               | GPU-accelerated 2D, beautiful    |
| Frontend UI            | HTML/JS (vanilla or Svelte)  | Lightweight, fast                |
| Plots in browser       | D3.js or Recharts            | Flexible, interactive            |
| Communication          | WebSocket (MessagePack)      | Low latency, binary efficiency   |
| Config                 | YAML                         | Human-readable                   |
| Build                  | maturin (Rust→Python wheel)  | Standard PyO3 build tool         |
| Prototyping fallback   | Python + Numba               | Before Rust port is ready        |

---

## 14. Open Design Questions

These are decisions to make during implementation, informed by experimentation:

1. **Toroidal vs. walled world?** Toroidal avoids edge effects but removes wall-hugging
   as an ecological niche. Walls add complexity but more interesting spatial dynamics.

2. **Continuous vs. discrete node types?** Current spec has discrete types (mouth, gripper,
   sensor). Alternative: each node has continuous "capability weights" for each function,
   all evolvable. More flexible but harder to visualize.

3. **Genome encoding: direct vs. developmental?** Current spec uses direct encoding
   (genome → body plan directly). Alternative: L-system or developmental encoding where
   the genome encodes growth rules. Much more powerful for producing complex symmetric
   body plans, but significantly more complex to implement.

4. **2D vs. 2.5D?** Could add a depth layer (creatures can be "above" or "below" each
   other) without full 3D physics. Adds predator evasion dimension. Worth exploring in
   Phase 8.

5. **Sound?** Mapping ecological dynamics to ambient sound (population = drone volume,
   predation events = clicks, reproduction = chimes) could make long observation sessions
   more engaging. Low priority but high coolness factor.

---

## Appendix A: References and Inspiration

- **Purcell, E.M.** "Life at Low Reynolds Number" (1977) — physics foundation
- **Beer, R.D.** CTRNN-based evolved agents — brain architecture
- **Sims, K.** "Evolving Virtual Creatures" (1994) — morphology + brain co-evolution
- **Stanley, K.O.** NEAT / HyperNEAT — topology-evolving neural nets
- **Ray, T.** Tierra — digital evolution without explicit fitness
- **Lipson, H. & Pollack, J.** Evolved physical robots — embodied neuroevolution
- **Lenia** (Chan, B.W.-C.) — continuous cellular automata, gorgeous visuals
- **Framsticks** — articulated evolved creatures with physics
- **Avida** — digital evolution with complex ecologies

---

*Named after Antonie van Leeuwenhoek's "animalcula" — the tiny creatures he first observed swimming in pond water in 1676, the moment humanity discovered microscopic life.*
