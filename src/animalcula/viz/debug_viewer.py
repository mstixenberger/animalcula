"""Minimal local viewers for simulation inspection."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import ModuleType

from animalcula.sim.world import Snapshot, World

NODE_COLORS = {
    "body": "#9aa5b1",
    "mouth": "#e76f51",
    "gripper": "#2a9d8f",
    "sensor": "#457b9d",
    "photoreceptor": "#f4a261",
}

ROLE_COLORS = {
    "autotroph": "#d4a017",
    "herbivore": "#2a9d8f",
    "predator": "#c1121f",
}


def _rgb_to_css(color_rgb: tuple[int, int, int] | list[int]) -> str:
    red, green, blue = (max(0, min(255, int(channel))) for channel in color_rgb)
    return f"rgb({red}, {green}, {blue})"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Animalcula Debug Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101317;
      --panel: #151a20;
      --panel-2: #1d2430;
      --text: #e5edf5;
      --muted: #9fb0c1;
      --accent: #f4a261;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background:
        radial-gradient(circle at top, rgba(244, 162, 97, 0.16), transparent 28%),
        linear-gradient(180deg, #0d1117, var(--bg));
      color: var(--text);
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
    }}
    .wrap {{
      max-width: {canvas_width}px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      background: #161a1f;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 18px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }}
    .bar {{
      display: grid;
      gap: 10px;
      padding: 14px 16px;
      background: rgba(21, 26, 32, 0.92);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
    }}
    .row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent);
      color: #101317;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    input[type="range"] {{
      flex: 1 1 320px;
      accent-color: var(--accent);
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }}
    .card {{
      padding: 10px 12px;
      background: rgba(29, 36, 48, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 1.05rem;
      font-weight: 700;
    }}
    code {{
      font-family: "Iosevka Term", "SFMono-Regular", monospace;
      color: var(--text);
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <canvas id="sim" width="{canvas_width}" height="{canvas_height}"></canvas>
    <section class="bar">
      <div class="row">
        <button id="toggle">Pause</button>
        <button id="step">Step</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <label class="meta" for="speed">speed</label>
        <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1">
      </div>
      <div class="meta" id="status"></div>
      <div class="stats">
        <div class="card"><span class="label">Population</span><span class="value" id="populationStat"></span></div>
        <div class="card"><span class="label">Energy</span><span class="value" id="energyStat"></span></div>
        <div class="card"><span class="label">Autotrophs</span><span class="value" id="autotrophStat"></span></div>
        <div class="card"><span class="label">Herbivores</span><span class="value" id="herbivoreStat"></span></div>
        <div class="card"><span class="label">Predators</span><span class="value" id="predatorStat"></span></div>
      </div>
      <div class="meta">
        Generated from <code>animalcula view</code> HTML fallback because Tkinter was unavailable.
      </div>
    </section>
  </main>
  <script>
    const snapshots = {snapshots_json};
    const nodeColors = {node_colors_json};
    const roleColors = {role_colors_json};
    const canvas = document.getElementById("sim");
    const ctx = canvas.getContext("2d");
    const toggle = document.getElementById("toggle");
    const step = document.getElementById("step");
    const scrub = document.getElementById("scrub");
    const speed = document.getElementById("speed");
    const status = document.getElementById("status");
    const populationStat = document.getElementById("populationStat");
    const energyStat = document.getElementById("energyStat");
    const autotrophStat = document.getElementById("autotrophStat");
    const herbivoreStat = document.getElementById("herbivoreStat");
    const predatorStat = document.getElementById("predatorStat");
    let frame = 0;
    let running = true;
    let playbackRate = 1;
    let lastTimestamp = 0;
    let frameAccumulator = 0;

    scrub.max = String(Math.max(0, snapshots.length - 1));

    function toCanvas(x, y, snapshot) {{
      return [
        (x / Math.max(snapshot.world_width, 1.0)) * canvas.width,
        (y / Math.max(snapshot.world_height, 1.0)) * canvas.height,
      ];
    }}

    function draw(snapshot) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const creatureRoles = new Map(snapshot.creatures.map((creature) => [creature.creature_id, creature.trophic_role]));
      const creatureColors = new Map(snapshot.creatures.map((creature) => [creature.creature_id, rgbToCss(creature.color_rgb)]));

      for (const edge of snapshot.edges) {{
        const [ax, ay] = toCanvas(edge.ax, edge.ay, snapshot);
        const [bx, by] = toCanvas(edge.bx, edge.by, snapshot);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = edge.has_motor ? "#9fb3c8" : "#62707d";
        ctx.lineWidth = edge.has_motor ? 2 : 1;
        ctx.stroke();
      }}

      for (const node of snapshot.nodes) {{
        const [cx, cy] = toCanvas(node.x, node.y, snapshot);
        const lineageColor = creatureColors.get(node.creature_id) || "#cbd5e1";
        const fill = nodeColors[node.node_type] || "#94a3b8";
        const radius = Math.max(2.0, node.radius * 2.0);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = lineageColor;
        ctx.stroke();
      }}

      status.textContent =
        "frame=" + (frame + 1) + "/" + snapshots.length +
        " tick=" + snapshot.tick +
        " population=" + snapshot.population +
        " total_energy=" + snapshot.total_energy.toFixed(2) +
        " speed=" + playbackRate.toFixed(2) + "x";
      const roleCounts = snapshot.creatures.reduce((counts, creature) => {{
        counts[creature.trophic_role] = (counts[creature.trophic_role] || 0) + 1;
        return counts;
      }}, {{}});
      populationStat.textContent = String(snapshot.population);
      energyStat.textContent = snapshot.total_energy.toFixed(2);
      autotrophStat.textContent = String(roleCounts.autotroph || 0);
      herbivoreStat.textContent = String(roleCounts.herbivore || 0);
      predatorStat.textContent = String(roleCounts.predator || 0);
      scrub.value = String(frame);
    }}

    function renderCurrent() {{
      draw(snapshots[frame]);
    }}

    function advanceFrame() {{
      frame = (frame + 1) % snapshots.length;
      renderCurrent();
    }}

    function rgbToCss(rgb) {{
      if (!Array.isArray(rgb) || rgb.length !== 3) {{
        return "#cbd5e1";
      }}
      return "rgb(" + rgb.map((value) => Math.max(0, Math.min(255, Math.round(value)))).join(", ") + ")";
    }}

    toggle.addEventListener("click", () => {{
      running = !running;
      toggle.textContent = running ? "Pause" : "Play";
    }});

    step.addEventListener("click", () => {{
      running = false;
      toggle.textContent = "Play";
      advanceFrame();
    }});

    scrub.addEventListener("input", (event) => {{
      running = false;
      toggle.textContent = "Play";
      frame = Number(event.target.value);
      renderCurrent();
    }});

    speed.addEventListener("input", (event) => {{
      playbackRate = Number(event.target.value);
      renderCurrent();
    }});

    window.addEventListener("keydown", (event) => {{
      if (event.code === "Space") {{
        event.preventDefault();
        toggle.click();
      }} else if (event.code === "ArrowRight") {{
        event.preventDefault();
        step.click();
      }}
    }});

    function animationLoop(timestamp) {{
      if (!lastTimestamp) {{
        lastTimestamp = timestamp;
      }}
      const delta = timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      if (running) {{
        frameAccumulator += delta * playbackRate;
        while (frameAccumulator >= Math.max(16, {frame_delay_ms})) {{
          advanceFrame();
          frameAccumulator -= Math.max(16, {frame_delay_ms});
        }}
      }}
      window.requestAnimationFrame(animationLoop);
    }}

    renderCurrent();
    window.requestAnimationFrame(animationLoop);
  </script>
</body>
</html>
"""


def _load_tk() -> ModuleType:
    import tkinter as tk

    return tk


def _to_canvas(
    x: float,
    y: float,
    snapshot: Snapshot,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[float, float]:
    return (
        (x / max(snapshot.world_width, 1.0)) * canvas_width,
        (y / max(snapshot.world_height, 1.0)) * canvas_height,
    )


def _default_html_path(world: World) -> Path:
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / f"animalcula_view_seed{world.seed}_tick{world.tick}.html"


def _build_html_viewer(
    world: World,
    *,
    steps_per_frame: int,
    frame_delay_ms: int,
    canvas_width: int,
    canvas_height: int,
    max_frames: int,
) -> str:
    snapshots = [asdict(world.snapshot())]
    for _ in range(max(1, max_frames) - 1):
        world.step(steps_per_frame)
        snapshots.append(asdict(world.snapshot()))
    return HTML_TEMPLATE.format(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        frame_delay_ms=frame_delay_ms,
        snapshots_json=json.dumps(snapshots),
        node_colors_json=json.dumps(NODE_COLORS),
        role_colors_json=json.dumps(ROLE_COLORS),
    )


def write_html_viewer(
    world: World,
    *,
    path: str | Path | None = None,
    steps_per_frame: int = 1,
    frame_delay_ms: int = 33,
    canvas_width: int = 900,
    canvas_height: int = 900,
    max_frames: int = 600,
) -> Path:
    target = Path(path) if path is not None else _default_html_path(world)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _build_html_viewer(
            world,
            steps_per_frame=max(1, steps_per_frame),
            frame_delay_ms=max(1, frame_delay_ms),
            canvas_width=max(200, canvas_width),
            canvas_height=max(200, canvas_height),
            max_frames=max(1, max_frames),
        ),
        encoding="utf-8",
    )
    return target


def _launch_tk_viewer(
    world: World,
    *,
    tk: ModuleType,
    steps_per_frame: int,
    frame_delay_ms: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    root = tk.Tk()
    root.title("Animalcula Debug Viewer")
    root.configure(bg="#111318")

    canvas = tk.Canvas(
        root,
        width=canvas_width,
        height=canvas_height,
        bg="#161a1f",
        highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)

    overlay = tk.StringVar(value="")
    label = tk.Label(
        root,
        textvariable=overlay,
        anchor="w",
        justify="left",
        bg="#111318",
        fg="#f1f5f9",
        font=("Menlo", 11),
        padx=12,
        pady=8,
    )
    label.pack(fill="x")

    running = True
    pending_single_step = False

    def _toggle_running(_: object | None = None) -> None:
        nonlocal running
        running = not running

    def _single_step(_: object | None = None) -> None:
        nonlocal pending_single_step
        pending_single_step = True

    root.bind("<space>", _toggle_running)
    root.bind("<Right>", _single_step)

    def _draw(snapshot: Snapshot) -> None:
        canvas.delete("all")

        creature_roles = {
            creature.creature_id: creature.trophic_role
            for creature in snapshot.creatures
        }
        creature_colors = {
            creature.creature_id: _rgb_to_css(creature.color_rgb)
            for creature in snapshot.creatures
        }
        for edge in snapshot.edges:
            ax, ay = _to_canvas(
                edge.ax,
                edge.ay,
                snapshot,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            bx, by = _to_canvas(
                edge.bx,
                edge.by,
                snapshot,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill="#62707d" if not edge.has_motor else "#9fb3c8",
                width=1 if not edge.has_motor else 2,
            )

        for node in snapshot.nodes:
            cx, cy = _to_canvas(
                node.x,
                node.y,
                snapshot,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            role = creature_roles.get(node.creature_id)
            outline = creature_colors.get(node.creature_id, "#cbd5e1")
            fill = NODE_COLORS.get(node.node_type, "#94a3b8")
            radius = max(2.0, node.radius * 2.0)
            canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=fill,
                outline=outline,
                width=2,
            )
            if role is not None:
                role_radius = max(1.0, radius * 0.35)
                canvas.create_oval(
                    cx - role_radius,
                    cy - role_radius,
                    cx + role_radius,
                    cy + role_radius,
                    fill=ROLE_COLORS.get(role, "#cbd5e1"),
                    outline="",
                )

        overlay.set(
            "\n".join(
                [
                    f"tick={snapshot.tick} population={snapshot.population} total_energy={snapshot.total_energy:.2f}",
                    f"space=play/pause right=step steps_per_frame={steps_per_frame} frame_delay_ms={frame_delay_ms}",
                ]
            )
        )

    def _frame() -> None:
        nonlocal pending_single_step
        if running or pending_single_step:
            world.step(steps_per_frame)
            pending_single_step = False
        _draw(world.snapshot())
        root.after(frame_delay_ms, _frame)

    _draw(world.snapshot())
    _frame()
    root.mainloop()


def launch_viewer(
    world: World,
    *,
    steps_per_frame: int = 1,
    frame_delay_ms: int = 33,
    canvas_width: int = 900,
    canvas_height: int = 900,
    backend: str = "auto",
    html_out_path: str | Path | None = None,
    max_frames: int = 600,
) -> Path | None:
    if backend not in {"auto", "tk", "html"}:
        msg = f"unsupported viewer backend: {backend}"
        raise ValueError(msg)

    if backend == "html":
        return write_html_viewer(
            world,
            path=html_out_path,
            steps_per_frame=steps_per_frame,
            frame_delay_ms=frame_delay_ms,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_frames=max_frames,
        )

    try:
        tk = _load_tk()
    except Exception as exc:  # pragma: no cover - platform-dependent
        if backend == "tk":
            raise RuntimeError(
                "Tkinter is required for `animalcula view --viewer-backend tk` on this machine"
            ) from exc
        return write_html_viewer(
            world,
            path=html_out_path,
            steps_per_frame=steps_per_frame,
            frame_delay_ms=frame_delay_ms,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_frames=max_frames,
        )

    _launch_tk_viewer(
        world,
        tk=tk,
        steps_per_frame=steps_per_frame,
        frame_delay_ms=frame_delay_ms,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    return None
