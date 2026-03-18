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

- **2D continuous space**, bounded with **soft repulsive walls** (1/r² force near edges).
  Toroidal wrapping was considered but rejected — bounded worlds avoid minimum-image
  physics complications and add wall-hugging as an ecological niche.
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

**Validation requirement:** A 2-node, 1-motor creature with identical drag coefficients
must produce zero net displacement under symmetric oscillation. This is a correctness test
for the overdamped regime — if it drifts, the physics has a symmetry-breaking bug.

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

**Nutrient throughput is finite.** Sources emit nutrients at a configured rate per tick
(not a hard set), and each grid cell has a maximum nutrient density cap
(`nutrient_max_density`). This ensures that:
- Food scarcity is genuine when population is high (creatures consume faster than sources emit)
- Resource competition drives niche differentiation rather than an artificial crowding multiplier
- Predation becomes a real alternative strategy when the nutrient floor drops

### 1.3.1 Obstacles

The world can contain **static obstacles** — circles or rectangles placed at world
initialization and during long-cycle epoch events. Obstacles use the same soft repulsion
as node-node contact. They create sheltered zones, chokepoints, and spatial heterogeneity
that drives niche differentiation. Obstacle placement is configurable and can change at
epoch boundaries.

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
  - drag_coeff: float       — evolvable per node (e.g., flagella have lower drag)
  - heading_node: bool      — node index 0 is the "head" by genome convention

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

**Heading convention:** Node 0 in the genome is the "head." The creature's facing direction
is defined as the vector from center-of-mass to the head node. Directional sensors use this
vector. Rendering draws head-first.

**Per-node drag is evolvable.** `drag_coeff` is part of the genome and subject to mutation.
This is a key locomotion mechanism: a node with low drag moves more for the same force,
breaking reciprocal symmetry (see scallop theorem in §1.2). Flagella-like structures
emerge naturally when terminal nodes evolve lower drag than body nodes.

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

Each creature carries two state scalars:
- **energy** (float, starts at birth energy) — fuel for metabolism, movement, reproduction
- **health** (float, starts at max health) — structural integrity, reduced by damage

| Activity             | Energy cost per timestep               |
|----------------------|----------------------------------------|
| Existing (metabolism)| `E_basal * num_nodes` (bigger = more expensive) |
| Motor actuation      | `E_motor * |torque_applied|` per motor |
| Gripping (holding)   | `E_grip * num_active_latches`          |
| Chemical emission    | `E_chem * emission_rate`               |
| Health regeneration  | `E_regen` per tick while health < max  |

| Activity             | Energy gain                            |
|----------------------|----------------------------------------|
| Nutrient absorption  | `E_feed * local_nutrient_conc` (via mouth nodes) |
| Photosynthesis       | `E_photo * local_light * num_photoreceptors` |
| Predation            | `victim.energy * transfer_efficiency` on kill |
| Scavenging           | Absorb from detritus grid cells        |

### 2.4.1 Health

Health is a separate axis from energy. Energy is fuel; health is structural integrity.

**Health is reduced by:**
- Bite damage from predators (primary source)
- Environmental stress (extreme drag shifts, prolonged starvation)

**Health is restored by:**
- Passive regeneration at a constant energy cost (`E_regen` per tick while health < max)

**Effect of low health:**
- Motor output is scaled by `0.5 + 0.5 * (health / max_health)` — injured creatures are
  sluggish but not helpless. A creature at 50% health has 75% motor effectiveness.
- This creates a "wounded gazelle" dynamic: injured prey are easier to catch, rewarding
  predators that wound-and-pursue rather than needing sustained contact for a kill.

**Death** occurs when health ≤ 0 OR energy ≤ 0. The creature's remaining structural energy
becomes a detritus patch on the grid.

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
INPUTS (dynamic, scales with morphology):
  - Per sensor node: local nutrient gradient (dx, dy), chemical A, chemical B
  - Per photoreceptor: light intensity, light direction (dx, dy)
  - Per motor joint: current angle, angular velocity
  - Per gripper: contact (0/1), grip active (0/1)
  - Global: own energy level (normalized), own age (normalized),
            own health (normalized)

HIDDEN LAYER:
  - N_hidden neurons (evolvable, range [4, 24])
  - Fully connected recurrent (all-to-all within hidden + self-connections)

OUTPUTS (dynamic, scales with morphology):
  - Per motor joint: desired torque [-1, 1] scaled by motor_torque_max
  - Per gripper: grip/release (thresholded at 0.5)
  - Per mouth: bite force [0, 1]
  - Chemical emission rate [0, 1]
  - Reproduce signal (thresholded, only acts if energy > reproduction_threshold)
```

**Critical implementation note:** Both input and output vector sizes are determined by the
creature's morphology (number of sensors, photoreceptors, motors, grippers, mouths). The
CTRNN input weight matrix must resize when structural mutations add or remove sensing/motor
nodes. This mirrors the existing output-resize logic. A creature with 3 sensor nodes gets
3× the sensor input channels of a creature with 1 sensor node, enabling spatial resolution
from distributed sensing and multi-photoreceptor triangulation.

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
- Mutation rate (self-adaptive, clamped to [0.001, 0.2]) — replaces global mutation rate
  for weight/bias/tau perturbations. Mutates itself each reproduction. Structural mutation
  rates (add/remove node/edge) remain global config parameters.
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
- Two creatures **of the same species** (low genome distance) with energy > threshold come
  into physical contact.
- Both signal reproduce simultaneously.
- Offspring genome = crossover of both parents + mutations.
- Energy cost split among both parents.
- Requires **evolved coordination** — this is hard, so it should be rewarding (crossover
  explores the fitness landscape faster).
- **Same-species restriction** avoids the structural alignment problem: crossover between
  creatures with different node/edge/neuron counts produces broken offspring. By restricting
  to same-species (similar topology), gene alignment is straightforward. NEAT-style
  historical innovation markings are deferred unless this proves too limiting.

### 4.3 Mutation Operators

| Operator              | Rate (self-adaptive) | Effect                         |
|-----------------------|---------------------|--------------------------------|
| Weight perturbation   | from meta-gene      | Gaussian noise, σ from meta-gene |
| Bias perturbation     | from meta-gene      | Same                           |
| Time constant perturb | from meta-gene      | Log-normal noise               |
| Add node              | ~0.005 per birth    | Insert node, random connections |
| Remove node           | ~0.003 per birth    | Remove least-connected non-bridge node |
| Add edge              | ~0.01 per birth     | Connect two unconnected nodes  |
| Remove edge           | ~0.005 per birth    | Remove random non-bridge edge  |
| Chain extension       | ~0.05 per birth     | Extend terminal node linearly with motor + inherited type |
| Node type change      | ~0.002 per birth    | Mutate node type               |
| Motor toggle          | ~0.05 per birth     | Flip motorized ↔ passive on random edge |
| Hidden neuron add     | ~0.005 per birth    | Add neuron to hidden layer     |
| Hidden neuron remove  | ~0.003 per birth    | Remove neuron                  |
| Drag coefficient      | from meta-gene      | Per-node drag perturbation     |
| Duplication           | ~0.001 per birth    | Duplicate a subgraph (deferred) |

**Bidirectional morphology is essential.** Remove-node and remove-edge operators prevent
a one-way complexity ratchet where creatures can only grow. Remove operators must preserve
graph connectivity — never remove bridge edges or bridge nodes. Without removal, creatures
accumulate useless nodes that cost basal energy but can never be shed.

Weight, bias, time constant, and drag perturbation rates are governed by the per-genome
self-adaptive mutation rate meta-gene. Structural mutation rates (add/remove node/edge,
chain extension, motor toggle, node type, hidden neuron) remain global config parameters.

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

**Implementation note:** Spatial hashing is deferred to the Rust port. The naive O(N²)
approach is acceptable in the Python prototype at current population scales (~300 creatures,
~1500 nodes). The Rust port is where O(N²) becomes a real bottleneck (2400 nodes ×
10,000 ticks/sec = 28.8 billion checks/sec).

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

### 5.6 Performance Strategy: Numba Now, Rust When Stable

The design is still actively evolving (dynamic brain inputs, health system, remove
mutations, nutrient caps, per-node drag, obstacles, etc.). Porting a moving target to Rust
wastes effort. The current strategy:

1. **Now:** Pure Python + **Numba @njit** for the three hot inner loops: node repulsion
   (`apply_node_repulsion`), edge spring forces (`apply_edge_springs`), and CTRNN step
   (`step_brain`). This gets ~50-100× over pure Python, enough for ~200 creatures.

2. **When the design stabilizes:** Port the hot path to Rust via PyO3. Keep Python for
   orchestration and analytics. "Stable" means: the core data structures (creature state,
   genome, brain I/O) stop changing between features.

3. **Stabilize the WebSocket snapshot format now.** This is the contract between the
   Python/Rust backend and the PixiJS frontend. Design it once so the frontend doesn't
   need to change when the backend language does.

---

## 6. Visualization and Display

### 6.1 Browser Frontend (Committed Path)

**Stack:** Python backend (FastAPI + uvicorn) → WebSocket → Browser (**PixiJS** / WebGL)

**This is the only product-facing frontend.** Local Tk/HTML viewers are debug fallbacks
only — no new features go there. The current inline-HTML canvas approach in `animalcula web`
should be replaced with a proper PixiJS/WebGL renderer to enable the microscope aesthetic
(shaders, glow, translucency) and handle thousands of sprites at 60fps.

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
│ ▶ ⏸ ⏩ x1 x10 x100 x1000  │ Tick: 142,857  Pop: 287  Species: 6  │ 🔬 📊 │
├──────────────────────────────────────────────────┬──────────────────────────┤
│                                                  │ ┌──────────────────────┐ │
│                                                  │ │   ACTIVE PANEL       │ │
│                                                  │ │   (one of):          │ │
│                                                  │ │                      │ │
│              WORLD VIEWPORT                      │ │   🔬 Inspector       │ │
│              (WebGL canvas)                      │ │   📊 Analytics       │ │
│                                                  │ │   🧬 Genome         │ │
│              click creature = select             │ │   🧠 Brain          │ │
│              scroll = zoom                       │ │                      │ │
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
FPS: 60 | Sim rate: 4,200 ticks/s | [🔬] [📊]
```

- **Speed slider** is continuous, not just preset steps. Drag from x1 to xMAX.
  At high speed, rendering drops to every Nth frame (adaptive frame skip).
- **Step forward/back** when paused: advance exactly 1 physics tick. Essential for
  debugging creature behavior. "Step back" requires checkpoint buffering (keep last
  ~100 ticks in memory as ring buffer).
- The two icons toggle the right-side panels (Inspector, Analytics).

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

Opened from the Inspector's "View Brain" button. Shows a **static inspection** of the
selected creature's CTRNN structure and weights (not live streaming traces).

```
┌─ 🧠 BRAIN: Creature #4201 ───────────────────┐
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │         NETWORK GRAPH (static)            │  │
│  │                                           │  │
│  │    [S1]─┐    ┌─[H1]──[H2]─┐    ┌─[M1]   │  │
│  │    [S2]─┼────┤  ↕  ↗  ↕   ├────┤ [M2]   │  │
│  │    [S3]─┤    └─[H3]──[H4]─┘    ├─[M3]   │  │
│  │    [S4]─┘       ↕    ↕         └─[M4]   │  │
│  │              [H5]──[H6]           [G1]   │  │
│  │                                           │  │
│  │  ● node size = current activation         │  │
│  │  ● edge width = |weight|                  │  │
│  │  ● edge color = green(+) / red(-)        │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ── NEURON DETAIL (hover any node) ────────   │
│  Neuron H3:                                    │
│  Bias θ = 0.83    │  Time const τ = 2.40      │
│  Inputs: S2(w=1.3), H1(w=-0.7), H4(w=0.9)   │
│                                                │
│  [📊 Weight Matrix Heatmap]                    │
└────────────────────────────────────────────────┘
```

**MVP scope:** Network graph showing topology, weight magnitudes, and excitatory/inhibitory
coloring. Hover for per-neuron detail (bias, tau, input connections). Weight matrix heatmap
for full connectivity overview. **Deferred:** scrolling neuron traces, oscillation analysis,
signal flow animation, trace export. These require per-neuron time series buffering on the
backend and are not needed for initial brain inspection.

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

### 6.9 ⚙ Live Tuning Panel (Deferred)

**Deferred.** Parameter tuning happens through headless sweeps and config file iteration,
not browser sliders. Building a live tuning panel requires a full command protocol from
frontend to backend for every parameter, plus undo/logging. The ROI is low while headless
sweeps exist.

When/if live tuning is added later, the key design principles are:
- Every slider has a sensible range with reset-to-default.
- Changes are logged with tick timestamps for post-hoc correlation.
- Presets let you jump between known-interesting parameter regimes.
- God mode events (meteor, flood, eclipse, inject species) stress-test ecosystem resilience.

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
  nutrient_source_rate: 0.5          # emission per tick (not hard-set)
  nutrient_max_density: 5.0          # per-cell cap
  nutrient_decay_rate: 0.001
  light_intensity_max: 1.0
  light_direction: [1.0, 0.0]
  chemical_diffusion_rate: 0.2
  chemical_decay_rate: 0.05
  season_cycle_ticks: 10000
  epoch_cycle_ticks: 100000
  wall_repulsion_strength: 50.0      # 1/r² soft wall force
  obstacle_count: 0                  # static obstacles (circles)
  obstacle_radius: 20.0

energy:
  basal_cost_per_node: 0.001
  motor_cost_per_torque: 0.0005
  grip_cost: 0.002
  feed_rate: 0.01
  photosynthesis_rate: 0.005
  predation_transfer_efficiency: 0.6
  reproduction_threshold: 100.0
  birth_energy: 50.0
  health_max: 100.0
  health_regen_rate: 0.1             # health restored per tick
  health_regen_cost: 0.002           # energy cost per tick of regen
  bite_health_damage: 5.0            # health lost per bite

creatures:
  max_nodes: 12
  max_population: 500
  min_population: 20
  drag_coeff_range: [0.3, 3.0]       # evolvable per-node drag bounds

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

### Phase 1–4: Core Simulation (COMPLETE)
- [x] Spring-mass overdamped physics, CTRNN brains, genome encoding
- [x] Energy model, feeding, photosynthesis, detritus recycling
- [x] Asexual reproduction, death, population safeguards
- [x] Headless mode, turbo, parameter sweeps, seed-bank promotion
- [x] Gripper mechanics, predation, chemical emission/sensing
- [x] Environmental variation (nutrient shifts, light seasons, drag cycles, epochs)
- [x] Speciation/extinction tracking, phylogeny export
- [x] Structural, node-type, motor-topology, chain-extension, hidden-neuron mutations
- [x] Debug viewers (Tk, HTML), basic browser frontend (`animalcula web`)
- [x] SQLite/JSONL logging, interestingness scoring, dominance detection

### Phase 5: Design Completion (CURRENT)
Implement spec gaps identified in the 2026-03-18 design review:
- [ ] Finite nutrient throughput (rate-based emission + per-cell cap)
- [ ] Bounded world with soft 1/r² repulsive walls (replace toroidal)
- [ ] Health axis separate from energy (bite/environment damage, regen, motor impairment)
- [ ] Dynamic per-morphology brain inputs (resize input weights on mutation)
- [ ] Per-node evolvable drag coefficients
- [ ] Remove-node and remove-edge mutation operators
- [ ] Add-edge mutation operator
- [ ] Self-adaptive mutation rate meta-gene (clamped [0.001, 0.2])
- [ ] Creature heading convention (node 0 = head)
- [ ] Static obstacles with soft repulsion
- [ ] Same-species sexual reproduction (conjugation with crossover)
- [ ] Scallop theorem validation test
- [ ] Numba @njit on hot loops (repulsion, springs, CTRNN step)

### Phase 6: Browser Frontend (PixiJS)
- [ ] Replace inline-HTML canvas with PixiJS/WebGL renderer
- [ ] Microscope aesthetic (shaders, glow, translucency, dark-field background)
- [ ] Creature inspector panel
- [ ] Static brain viewer (network graph, weight heatmap)
- [ ] Genome viewer (structure, diff from parent)
- [ ] Analytics panel (population, energy, phylogeny, trophic charts)
- [ ] Timeline bar with event markers
- [ ] Stabilize WebSocket snapshot format as frontend/backend contract

### Phase 7: Rust Port (when design is stable)
- [ ] Port force computation, CTRNN, spatial hash to Rust
- [ ] PyO3 bindings producing same WebSocket snapshot format
- [ ] Benchmark: target 1000 steps/sec with 300 creatures
- [ ] Checkpoint save/load

### Phase 8: Analytics and Polish (Ongoing)
- [ ] UMAP phenotype space visualization
- [ ] Time machine replay from checkpoints
- [ ] Shader effects (caustics, depth of field)
- [ ] Sound design (map population dynamics to ambient audio)
- [ ] Record and share interesting evolutionary runs

---

## 13. Technology Stack Summary

| Component             | Technology (now)             | Technology (Rust port)           |
|-----------------------|------------------------------|----------------------------------|
| Physics engine        | Python + Numba @njit         | Rust (+ rayon for threading)     |
| Neural nets (CTRNN)   | Python + Numba @njit         | Rust                             |
| Spatial indexing       | Naive O(N²) (adequate now)   | Rust (custom spatial hash)       |
| Python bindings        | N/A (pure Python)           | PyO3 / maturin                   |
| Orchestration          | Python (FastAPI)             | Python (FastAPI)                 |
| Analytics              | Python (NumPy, SciPy, UMAP) | Python (NumPy, SciPy, UMAP)     |
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

### Resolved (2026-03-18 design review)

1. **Toroidal vs. walled world?** → **Walled.** Bounded with soft 1/r² repulsive walls.
   Avoids minimum-image physics complications, adds wall-hugging niche.
2. **Brain inputs: fixed or per-morphology?** → **Per-morphology.** Dynamic input vector
   scales with sensor/photoreceptor/joint/gripper count. Input weight matrix resizes on
   structural mutation.
3. **Health separate from energy?** → **Yes.** Health reduced by bites + environment stress,
   passive regen at energy cost, mild motor impairment at low health.
4. **Sexual reproduction alignment?** → **Same-species only.** Crossover restricted to low
   genome-distance pairs. NEAT-style innovation markings deferred.
5. **Rust port timing?** → **Deferred.** Numba for hot loops now. Rust when design stabilizes.
6. **Per-node drag evolvable?** → **Yes.** In genome, subject to mutation, clamped range.
7. **Self-adaptive mutation rates?** → **Yes.** Per-genome meta-gene, clamped [0.001, 0.2].
8. **Live tuning in browser?** → **Deferred.** Headless sweeps + config files for now.

### Still Open

1. **Continuous vs. discrete node types?** Current spec has discrete types (mouth, gripper,
   sensor). Alternative: continuous capability weights per node. More flexible but harder
   to visualize.

2. **Genome encoding: direct vs. developmental?** Current spec uses direct encoding.
   Alternative: L-system or developmental encoding for complex symmetric body plans.
   Significantly more complex to implement.

3. **2D vs. 2.5D?** Depth layer for predator evasion. Worth exploring later.

4. **Sound?** Mapping ecological dynamics to ambient sound. Low priority, high coolness.

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
