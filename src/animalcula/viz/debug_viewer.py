"""Minimal local viewers for simulation inspection."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import ModuleType
import webbrowser

from animalcula.sim.world import Snapshot, World
from animalcula.viz.payloads import sample_fields, snapshot_payload

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

FIELD_MODE_SEQUENCE = ("combined", "nutrient", "light", "chemical_a", "chemical_b", "detritus")


def _rgb_to_css(color_rgb: tuple[int, int, int] | list[int]) -> str:
    red, green, blue = (max(0, min(255, int(channel))) for channel in color_rgb)
    return f"rgb({red}, {green}, {blue})"


def _rgb_to_hex(color_rgb: tuple[int, int, int] | list[int]) -> str:
    red, green, blue = (max(0, min(255, int(channel))) for channel in color_rgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _parse_color_rgb(color: str) -> tuple[int, int, int] | None:
    if color.startswith("#") and len(color) == 7:
        try:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        except ValueError:
            return None
    if not color.startswith("rgb(") or not color.endswith(")"):
        return None
    payload = [part.strip() for part in color[4:-1].split(",")]
    if len(payload) != 3:
        return None
    try:
        return tuple(max(0, min(255, int(float(part)))) for part in payload)  # type: ignore[return-value]
    except ValueError:
        return None


def _blend_css_color(css: str, *, alpha: float, background_rgb: tuple[int, int, int] = (22, 26, 31)) -> str:
    parsed = _parse_color_rgb(css)
    if parsed is None:
        return css
    red = round((parsed[0] * alpha) + (background_rgb[0] * (1.0 - alpha)))
    green = round((parsed[1] * alpha) + (background_rgb[1] * (1.0 - alpha)))
    blue = round((parsed[2] * alpha) + (background_rgb[2] * (1.0 - alpha)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _field_rgb(
    mode: str,
    nutrient: float,
    light: float,
    chemical_a: float,
    chemical_b: float,
    detritus: float,
) -> tuple[int, int, int]:
    if mode == "nutrient":
        return (18, max(26, min(255, int(30 + (nutrient * 140)))), 28)
    if mode == "light":
        return (
            max(28, min(255, int(40 + (light * 180)))),
            max(24, min(255, int(36 + (light * 110)))),
            28,
        )
    if mode == "chemical_a":
        return (28, 70, max(40, min(255, int(50 + (chemical_a * 180)))))
    if mode == "chemical_b":
        return (
            max(48, min(255, int(60 + (chemical_b * 120)))),
            36,
            max(60, min(255, int(72 + (chemical_b * 180)))),
        )
    if mode == "detritus":
        detritus_intensity = max(0, min(255, int(55 + (detritus * 140))))
        return (detritus_intensity, max(34, detritus_intensity - 18), 28)
    return (
        max(12, min(255, int(18 + (light * 72) + (chemical_b * 28)))),
        max(20, min(255, int(24 + (nutrient * 110) + (light * 36) + (chemical_a * 42)))),
        max(24, min(255, int(30 + (light * 40) + (chemical_a * 110) + (chemical_b * 90) + (detritus * 30)))),
    )

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
    .history-card {{
      padding: 12px;
    }}
    #historyCanvas {{
      width: 100%;
      height: 160px;
      display: block;
      margin-top: 8px;
      background: rgba(10, 14, 20, 0.56);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .inspector {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
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
        <input id="speed" type="range" min="0.25" max="32" step="0.25" value="1">
        <label class="meta" for="followToggle">follow</label>
        <input id="followToggle" type="checkbox" checked>
        <label class="meta" for="ambientToggle">ambient</label>
        <input id="ambientToggle" type="checkbox">
        <label class="meta" for="zoom">zoom</label>
        <input id="zoom" type="range" min="1" max="12" step="0.5" value="4">
        <label class="meta" for="fieldMode">field</label>
        <select id="fieldMode">
          <option value="combined">combined</option>
          <option value="nutrient">nutrient</option>
          <option value="light">light</option>
          <option value="chemical_a">chemical-a</option>
          <option value="chemical_b">chemical-b</option>
          <option value="detritus">detritus</option>
        </select>
      </div>
      <div class="meta" id="status"></div>
      <div class="stats">
        <div class="card"><span class="label">Population</span><span class="value" id="populationStat"></span></div>
        <div class="card"><span class="label">Species</span><span class="value" id="speciesStat"></span></div>
        <div class="card"><span class="label">Diversity</span><span class="value" id="diversityStat"></span></div>
        <div class="card"><span class="label">Energy</span><span class="value" id="energyStat"></span></div>
        <div class="card"><span class="label">Autotrophs</span><span class="value" id="autotrophStat"></span></div>
        <div class="card"><span class="label">Herbivores</span><span class="value" id="herbivoreStat"></span></div>
        <div class="card"><span class="label">Predators</span><span class="value" id="predatorStat"></span></div>
      </div>
      <div class="stats">
        <div class="card"><span class="label">Recent Births</span><span class="value" id="recentBirthsStat"></span></div>
        <div class="card"><span class="label">Recent Deaths</span><span class="value" id="recentDeathsStat"></span></div>
        <div class="card"><span class="label">Recent Repros</span><span class="value" id="recentReproductionStat"></span></div>
        <div class="card"><span class="label">Recent Kills</span><span class="value" id="recentPredationStat"></span></div>
        <div class="card"><span class="label">Perturbations</span><span class="value" id="recentPerturbationStat"></span></div>
      </div>
      <div class="card history-card">
        <span class="label">Recent Ecology</span>
        <canvas id="historyCanvas" width="{canvas_width}" height="160"></canvas>
      </div>
      <div class="inspector" id="inspector">
        <div class="card"><span class="label">Selected</span><span class="value" id="selectedCreatureId"></span></div>
        <div class="card"><span class="label">Species</span><span class="value" id="selectedSpecies"></span></div>
        <div class="card"><span class="label">Role</span><span class="value" id="selectedRole"></span></div>
        <div class="card"><span class="label">Energy</span><span class="value" id="selectedEnergy"></span></div>
        <div class="card"><span class="label">Speed</span><span class="value" id="selectedSpeed"></span></div>
        <div class="card"><span class="label">Energy Trend</span><span class="value" id="selectedEnergyDelta"></span></div>
        <div class="card"><span class="label">Parent</span><span class="value" id="selectedParent"></span></div>
        <div class="card"><span class="label">Age</span><span class="value" id="selectedAge"></span></div>
        <div class="card"><span class="label">Born</span><span class="value" id="selectedBorn"></span></div>
        <div class="card"><span class="label">Genome Hash</span><span class="value" id="selectedGenomeHash"></span></div>
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
    const followToggle = document.getElementById("followToggle");
    const ambientToggle = document.getElementById("ambientToggle");
    const zoom = document.getElementById("zoom");
    const fieldMode = document.getElementById("fieldMode");
    const status = document.getElementById("status");
    const populationStat = document.getElementById("populationStat");
    const speciesStat = document.getElementById("speciesStat");
    const diversityStat = document.getElementById("diversityStat");
    const energyStat = document.getElementById("energyStat");
    const autotrophStat = document.getElementById("autotrophStat");
    const herbivoreStat = document.getElementById("herbivoreStat");
    const predatorStat = document.getElementById("predatorStat");
    const recentBirthsStat = document.getElementById("recentBirthsStat");
    const recentDeathsStat = document.getElementById("recentDeathsStat");
    const recentReproductionStat = document.getElementById("recentReproductionStat");
    const recentPredationStat = document.getElementById("recentPredationStat");
    const recentPerturbationStat = document.getElementById("recentPerturbationStat");
    const historyCanvas = document.getElementById("historyCanvas");
    const historyCtx = historyCanvas.getContext("2d");
    const selectedCreatureIdStat = document.getElementById("selectedCreatureId");
    const selectedSpeciesStat = document.getElementById("selectedSpecies");
    const selectedRoleStat = document.getElementById("selectedRole");
    const selectedEnergyStat = document.getElementById("selectedEnergy");
    const selectedSpeedStat = document.getElementById("selectedSpeed");
    const selectedEnergyDeltaStat = document.getElementById("selectedEnergyDelta");
    const selectedParentStat = document.getElementById("selectedParent");
    const selectedAgeStat = document.getElementById("selectedAge");
    const selectedBornStat = document.getElementById("selectedBorn");
    const selectedGenomeHashStat = document.getElementById("selectedGenomeHash");
    let frame = 0;
    let running = true;
    let playbackRate = 1;
    let followSelected = true;
    let ambientMode = false;
    let zoomLevel = 4;
    let activeFieldMode = "nutrient";
    let selectedCreatureId = null;
    let lastTimestamp = 0;
    let frameAccumulator = 0;
    let ambientFrameCounter = 0;

    scrub.max = String(Math.max(0, snapshots.length - 1));
    fieldMode.value = activeFieldMode;
    followToggle.checked = followSelected;
    ambientToggle.checked = ambientMode;
    zoom.value = String(zoomLevel);

    function cameraCenter(snapshot) {{
      const selected = snapshot.creatures.find((creature) => creature.creature_id === selectedCreatureId);
      if (followSelected && selected) {{
        return [selected.center_x, selected.center_y];
      }}
      return [snapshot.world_width / 2, snapshot.world_height / 2];
    }}

    function creatureFocusScore(creature) {{
      const speed = Number(creature.mean_speed_recent || 0);
      const energy = Number(creature.energy || 0);
      const age = Number(creature.age_ticks || 0);
      return (speed * 4.0) + Math.min(energy, 20) + (Math.min(age, 600) / 120);
    }}

    function rankedCreatureIds(snapshot) {{
      return snapshot.creatures
        .slice()
        .sort((left, right) => {{
          const scoreDelta = creatureFocusScore(right) - creatureFocusScore(left);
          if (Math.abs(scoreDelta) > 1e-9) {{
            return scoreDelta;
          }}
          return left.creature_id - right.creature_id;
        }})
        .map((creature) => creature.creature_id);
    }}

    function preferredCreatureId(snapshot) {{
      const rankedIds = rankedCreatureIds(snapshot);
      return rankedIds.length ? rankedIds[0] : null;
    }}

    function statsFor(snapshot) {{
      return snapshot.stats || {{}};
    }}

    function recentCounterDelta(frameIndex, key, windowSize = 48) {{
      const currentStats = statsFor(snapshots[frameIndex]);
      const earlierIndex = Math.max(0, frameIndex - Math.max(1, windowSize));
      const earlierStats = statsFor(snapshots[earlierIndex]);
      return (Number(currentStats[key] || 0) - Number(earlierStats[key] || 0));
    }}

    function previousCreatureFrame(frameIndex, creatureId) {{
      for (let index = frameIndex - 1; index >= 0; index -= 1) {{
        const creature = snapshots[index].creatures.find((candidate) => candidate.creature_id === creatureId);
        if (creature) {{
          return [index, creature];
        }}
      }}
      return [null, null];
    }}

    function selectedTrend(frameIndex, creature) {{
      if (!creature) {{
        return {{ speed: null, energyDelta: null }};
      }}
      const [previousIndex, previousCreature] = previousCreatureFrame(frameIndex, creature.creature_id);
      if (previousIndex === null || !previousCreature) {{
        return {{ speed: null, energyDelta: null }};
      }}
      const tickDelta = Math.max(1, snapshots[frameIndex].tick - snapshots[previousIndex].tick);
      const distance = Math.hypot(
        creature.center_x - previousCreature.center_x,
        creature.center_y - previousCreature.center_y,
      );
      return {{
        speed: distance / tickDelta,
        energyDelta: (Number(creature.energy || 0) - Number(previousCreature.energy || 0)) / tickDelta,
      }};
    }}

    function drawHistory(frameIndex) {{
      const series = snapshots.slice(Math.max(0, frameIndex - 159), frameIndex + 1);
      historyCtx.clearRect(0, 0, historyCanvas.width, historyCanvas.height);
      historyCtx.fillStyle = "#0f141a";
      historyCtx.fillRect(0, 0, historyCanvas.width, historyCanvas.height);
      if (!series.length) {{
        return;
      }}
      const plotX = 14;
      const plotY = 18;
      const plotWidth = historyCanvas.width - 28;
      const plotHeight = historyCanvas.height - 36;
      historyCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      historyCtx.lineWidth = 1;
      for (let row = 0; row < 4; row += 1) {{
        const y = plotY + ((plotHeight / 3) * row);
        historyCtx.beginPath();
        historyCtx.moveTo(plotX, y);
        historyCtx.lineTo(plotX + plotWidth, y);
        historyCtx.stroke();
      }}

      const populationValues = series.map((snapshot) => Number(snapshot.population || 0));
      const speciesValues = series.map((snapshot) => Number(statsFor(snapshot).species_count || 0));
      const predatorValues = series.map((snapshot) =>
        snapshot.creatures.reduce((count, creature) => count + (creature.trophic_role === "predator" ? 1 : 0), 0)
      );
      const peakValue = Math.max(1, ...populationValues, ...speciesValues, ...predatorValues);

      function drawSeries(values, color) {{
        historyCtx.beginPath();
        values.forEach((value, index) => {{
          const x = plotX + (plotWidth * (index / Math.max(values.length - 1, 1)));
          const y = plotY + plotHeight - ((value / peakValue) * plotHeight);
          if (index === 0) {{
            historyCtx.moveTo(x, y);
          }} else {{
            historyCtx.lineTo(x, y);
          }}
        }});
        historyCtx.strokeStyle = color;
        historyCtx.lineWidth = 2;
        historyCtx.stroke();
      }}

      drawSeries(populationValues, "#f4a261");
      drawSeries(speciesValues, "#8ecae6");
      drawSeries(predatorValues, "#e63946");
      historyCtx.fillStyle = "#9fb0c1";
      historyCtx.font = '12px "Iosevka Aile", "IBM Plex Sans", sans-serif';
      historyCtx.fillText("pop", plotX, historyCanvas.height - 8);
      historyCtx.fillStyle = "#8ecae6";
      historyCtx.fillText("species", plotX + 54, historyCanvas.height - 8);
      historyCtx.fillStyle = "#e63946";
      historyCtx.fillText("pred", plotX + 126, historyCanvas.height - 8);
    }}

    function hexToRgba(css, alpha) {{
      if (typeof css !== "string" || !css.startsWith("rgb(")) {{
        return "rgba(203, 213, 225, " + alpha.toFixed(3) + ")";
      }}
      return css.replace("rgb(", "rgba(").replace(")", ", " + alpha.toFixed(3) + ")");
    }}

    function toCanvas(x, y, snapshot) {{
      const [focusX, focusY] = cameraCenter(snapshot);
      const scaleX = (canvas.width / Math.max(snapshot.world_width, 1.0)) * zoomLevel;
      const scaleY = (canvas.height / Math.max(snapshot.world_height, 1.0)) * zoomLevel;
      return [
        ((x - focusX) * scaleX) + (canvas.width / 2),
        ((y - focusY) * scaleY) + (canvas.height / 2),
      ];
    }}

    function draw(snapshot) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      syncSelectedCreature(snapshot);
      const creatureRoles = new Map(snapshot.creatures.map((creature) => [creature.creature_id, creature.trophic_role]));
      const creatureColors = new Map(snapshot.creatures.map((creature) => [creature.creature_id, rgbToCss(creature.color_rgb)]));
      const creatureVisuals = new Map(snapshot.creatures.map((creature) => [creature.creature_id, creature]));
      drawFields(snapshot);
      drawCreatureSilhouettes(snapshot, creatureColors, creatureVisuals);

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
        const visual = creatureVisuals.get(node.creature_id);
        const fill = nodeColors[node.node_type] || "#94a3b8";
        const radius = Math.max(2.0, node.radius * 2.0);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = lineageColor;
        ctx.stroke();
        drawNodeGlyph(node, cx, cy, radius, visual ? visual.glyph_scale : 1.0);
      }}

      const selected = snapshot.creatures.find((creature) => creature.creature_id === selectedCreatureId);
      if (selected) {{
        const [sx, sy] = toCanvas(selected.center_x, selected.center_y, snapshot);
        ctx.beginPath();
        ctx.arc(sx, sy, 14, 0, Math.PI * 2);
        ctx.strokeStyle = rgbToCss(selected.color_rgb);
        ctx.lineWidth = 3;
        ctx.stroke();
        drawSelectedLabel(selected, sx, sy);
      }}

      status.textContent =
        "frame=" + (frame + 1) + "/" + snapshots.length +
        " tick=" + snapshot.tick +
        " population=" + snapshot.population +
        " total_energy=" + snapshot.total_energy.toFixed(2) +
        " speed=" + playbackRate.toFixed(2) + "x" +
        " follow=" + (followSelected ? "on" : "off") +
        " ambient=" + (ambientMode ? "on" : "off") +
        " zoom=" + zoomLevel.toFixed(1) + "x" +
        " field=" + activeFieldMode;
      const roleCounts = snapshot.creatures.reduce((counts, creature) => {{
        counts[creature.trophic_role] = (counts[creature.trophic_role] || 0) + 1;
        return counts;
      }}, {{}});
      const snapshotStats = statsFor(snapshot);
      const trend = selectedTrend(frame, selected);
      populationStat.textContent = String(snapshot.population);
      speciesStat.textContent = String(snapshotStats.species_count || 0);
      diversityStat.textContent = Number(snapshotStats.diversity_index || 0).toFixed(2);
      energyStat.textContent = snapshot.total_energy.toFixed(2);
      autotrophStat.textContent = String(roleCounts.autotroph || 0);
      herbivoreStat.textContent = String(roleCounts.herbivore || 0);
      predatorStat.textContent = String(roleCounts.predator || 0);
      recentBirthsStat.textContent = String(recentCounterDelta(frame, "births"));
      recentDeathsStat.textContent = String(recentCounterDelta(frame, "deaths"));
      recentReproductionStat.textContent = String(recentCounterDelta(frame, "reproductions"));
      recentPredationStat.textContent = String(recentCounterDelta(frame, "predation_kills"));
      recentPerturbationStat.textContent = String(recentCounterDelta(frame, "environment_perturbations", 96));
      drawHistory(frame);
      updateInspector(selected, trend);
      scrub.value = String(frame);
    }}

    function renderCurrent() {{
      draw(snapshots[frame]);
    }}

    function drawCreatureSilhouettes(snapshot, creatureColors, creatureVisuals) {{
      const creatureNodes = new Map();
      for (const node of snapshot.nodes) {{
        if (node.creature_id === null || node.creature_id === undefined) {{
          continue;
        }}
        if (!creatureNodes.has(node.creature_id)) {{
          creatureNodes.set(node.creature_id, []);
        }}
        creatureNodes.get(node.creature_id).push(node);
      }}
      for (const [creatureId, nodes] of creatureNodes.entries()) {{
        const visual = creatureVisuals.get(creatureId) || {{ silhouette_scale: 1.0, band_count: 2, band_offset: 0.0 }};
        const silhouetteScale = Number(visual.silhouette_scale || 1.0);
        const fill = hexToRgba(creatureColors.get(creatureId) || "#cbd5e1", 0.18);
        if (nodes.length === 1) {{
          const [cx, cy] = toCanvas(nodes[0].x, nodes[0].y, snapshot);
          const radius = Math.max(6.0, nodes[0].radius * 4.0 * silhouetteScale);
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.fill();
          drawCreatureBands(snapshot, nodes, visual, creatureColors.get(creatureId) || "#cbd5e1");
          continue;
        }}
        if (nodes.length === 2) {{
          const [ax, ay] = toCanvas(nodes[0].x, nodes[0].y, snapshot);
          const [bx, by] = toCanvas(nodes[1].x, nodes[1].y, snapshot);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.strokeStyle = fill;
          ctx.lineWidth = Math.max(8.0, (nodes[0].radius + nodes[1].radius) * 3.5 * silhouetteScale);
          ctx.lineCap = "round";
          ctx.stroke();
          ctx.lineCap = "butt";
          drawCreatureBands(snapshot, nodes, visual, creatureColors.get(creatureId) || "#cbd5e1");
          continue;
        }}
        const centroidX = nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length;
        const centroidY = nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length;
        const ordered = nodes
          .slice()
          .sort((left, right) => Math.atan2(left.y - centroidY, left.x - centroidX) - Math.atan2(right.y - centroidY, right.x - centroidX))
          .map((node) => {{
            const dx = node.x - centroidX;
            const dy = node.y - centroidY;
            const magnitude = Math.hypot(dx, dy) || 1.0;
            const inflate = node.radius * 1.6 * silhouetteScale;
            return [
              node.x + ((dx / magnitude) * inflate),
              node.y + ((dy / magnitude) * inflate),
            ];
          }});
        if (!ordered.length) {{
          continue;
        }}
        ctx.beginPath();
        const [startX, startY] = toCanvas(ordered[0][0], ordered[0][1], snapshot);
        ctx.moveTo(startX, startY);
        for (const [x, y] of ordered.slice(1)) {{
          const [px, py] = toCanvas(x, y, snapshot);
          ctx.lineTo(px, py);
        }}
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        drawCreatureBands(snapshot, nodes, visual, creatureColors.get(creatureId) || "#cbd5e1");
      }}
    }}

    function drawCreatureBands(snapshot, nodes, visual, colorCss) {{
      const bandCount = Math.max(1, Math.min(4, Number(visual.band_count || 1)));
      if (bandCount <= 1) {{
        return;
      }}
      const projected = nodes.map((node) => toCanvas(node.x, node.y, snapshot));
      const xs = projected.map((point) => point[0]);
      const ys = projected.map((point) => point[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const width = maxX - minX;
      const height = maxY - minY;
      const verticalBands = width >= height;
      ctx.save();
      ctx.strokeStyle = hexToRgba(colorCss, 0.28);
      ctx.lineWidth = 1.5;
      for (let index = 0; index < bandCount; index += 1) {{
        const offset = ((index + Number(visual.band_offset || 0.0)) % bandCount) / bandCount;
        if (verticalBands) {{
          const x = minX + (width * offset);
          ctx.beginPath();
          ctx.moveTo(x, minY - 2);
          ctx.lineTo(x, maxY + 2);
          ctx.stroke();
        }} else {{
          const y = minY + (height * offset);
          ctx.beginPath();
          ctx.moveTo(minX - 2, y);
          ctx.lineTo(maxX + 2, y);
          ctx.stroke();
        }}
      }}
      ctx.restore();
    }}

    function drawNodeGlyph(node, cx, cy, radius, glyphScale) {{
      const scaledRadius = radius * Math.max(0.85, glyphScale || 1.0);
      ctx.save();
      ctx.strokeStyle = "rgba(16, 19, 23, 0.88)";
      ctx.fillStyle = "rgba(16, 19, 23, 0.82)";
      ctx.lineWidth = Math.max(1, scaledRadius * 0.28);
      if (node.node_type === "mouth") {{
        ctx.beginPath();
        ctx.moveTo(cx - (scaledRadius * 0.8), cy + (scaledRadius * 0.12));
        ctx.lineTo(cx + (scaledRadius * 0.8), cy + (scaledRadius * 0.12));
        ctx.stroke();
      }} else if (node.node_type === "gripper") {{
        ctx.beginPath();
        ctx.moveTo(cx - (scaledRadius * 0.15), cy - (scaledRadius * 0.15));
        ctx.lineTo(cx + (scaledRadius * 0.95), cy - (scaledRadius * 0.95));
        ctx.moveTo(cx - (scaledRadius * 0.15), cy + (scaledRadius * 0.15));
        ctx.lineTo(cx + (scaledRadius * 0.95), cy + (scaledRadius * 0.95));
        ctx.stroke();
      }} else if (node.node_type === "sensor") {{
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(1.2, scaledRadius * 0.28), 0, Math.PI * 2);
        ctx.fill();
      }} else if (node.node_type === "photoreceptor") {{
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(1.4, scaledRadius * 0.52), 0, Math.PI * 2);
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawSelectedLabel(selected, sx, sy) {{
      const species = (selected.species_id || "species").replace("species-", "s");
      const label = "#" + selected.creature_id + " " + species + " " + selected.trophic_role;
      ctx.save();
      ctx.font = '600 13px "Iosevka Aile", "IBM Plex Sans", sans-serif';
      const width = ctx.measureText(label).width + 14;
      const height = 22;
      const left = sx + 16;
      const top = sy - 30;
      ctx.fillStyle = "rgba(16, 19, 23, 0.82)";
      ctx.fillRect(left, top, width, height);
      ctx.strokeStyle = rgbToCss(selected.color_rgb);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(left, top, width, height);
      ctx.fillStyle = "#e5edf5";
      ctx.textBaseline = "middle";
      ctx.fillText(label, left + 7, top + (height / 2));
      ctx.restore();
    }}

    function advanceFrame() {{
      frame = (frame + 1) % snapshots.length;
      renderCurrent();
    }}

    function drawFields(snapshot) {{
      const fields = snapshot.fields;
      if (!fields) {{
        return;
      }}
      const cellWidth = canvas.width / Math.max(fields.cols, 1);
      const cellHeight = canvas.height / Math.max(fields.rows, 1);
      for (let row = 0; row < fields.rows; row += 1) {{
        for (let col = 0; col < fields.cols; col += 1) {{
          const nutrient = fields.nutrient[row][col] || 0;
          const light = fields.light[row][col] || 0;
          const chemicalA = fields.chemical_a[row][col] || 0;
          const chemicalB = fields.chemical_b[row][col] || 0;
          const detritus = fields.detritus[row][col] || 0;
      const layers = activeFieldMode === "combined"
            ? [
                ["rgba(76, 175, 80, ", Math.min(0.18, nutrient * 0.16)],
                ["rgba(244, 162, 97, ", Math.min(0.12, light * 0.08)],
                ["rgba(74, 144, 226, ", Math.min(0.11, chemicalA * 0.12)],
                ["rgba(181, 93, 255, ", Math.min(0.11, chemicalB * 0.12)],
                ["rgba(143, 98, 65, ", Math.min(0.10, detritus * 0.11)],
              ]
            : [fieldLayer(activeFieldMode, nutrient, light, chemicalA, chemicalB, detritus)];
          for (const [prefix, alpha] of layers) {{
            if (alpha > 0) {{
              ctx.fillStyle = prefix + alpha.toFixed(3) + ")";
              ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth + 1, cellHeight + 1);
            }}
          }}
        }}
      }}
    }}

    function fieldLayer(mode, nutrient, light, chemicalA, chemicalB, detritus) {{
      if (mode === "nutrient") {{
        return ["rgba(76, 175, 80, ", Math.min(0.34, nutrient * 0.28)];
      }}
      if (mode === "light") {{
        return ["rgba(244, 162, 97, ", Math.min(0.26, light * 0.18)];
      }}
      if (mode === "chemical_a") {{
        return ["rgba(74, 144, 226, ", Math.min(0.28, chemicalA * 0.24)];
      }}
      if (mode === "chemical_b") {{
        return ["rgba(181, 93, 255, ", Math.min(0.28, chemicalB * 0.24)];
      }}
      if (mode === "detritus") {{
        return ["rgba(143, 98, 65, ", Math.min(0.24, detritus * 0.24)];
      }}
      return ["rgba(0, 0, 0, ", 0];
    }}

    function syncSelectedCreature(snapshot) {{
      if (!snapshot.creatures.length) {{
        selectedCreatureId = null;
        return;
      }}
      if (!snapshot.creatures.some((creature) => creature.creature_id === selectedCreatureId)) {{
        selectedCreatureId = preferredCreatureId(snapshot);
      }}
    }}

    function updateInspector(creature, trend) {{
      if (!creature) {{
        selectedCreatureIdStat.textContent = "none";
        selectedSpeciesStat.textContent = "-";
        selectedRoleStat.textContent = "-";
        selectedEnergyStat.textContent = "-";
        selectedSpeedStat.textContent = "-";
        selectedEnergyDeltaStat.textContent = "-";
        selectedParentStat.textContent = "-";
        selectedAgeStat.textContent = "-";
        selectedBornStat.textContent = "-";
        selectedGenomeHashStat.textContent = "-";
        return;
      }}
      selectedCreatureIdStat.textContent = "#" + creature.creature_id;
      selectedSpeciesStat.textContent = creature.species_id || "-";
      selectedRoleStat.textContent = creature.trophic_role;
      selectedEnergyStat.textContent = creature.energy.toFixed(2);
      selectedSpeedStat.textContent = trend.speed === null ? "-" : trend.speed.toFixed(2) + "/tick";
      selectedEnergyDeltaStat.textContent =
        trend.energyDelta === null
          ? "-"
          : (trend.energyDelta >= 0 ? "+" : "") + trend.energyDelta.toFixed(2) + "/tick";
      selectedParentStat.textContent = creature.parent_id === null ? "root" : "#" + creature.parent_id;
      selectedAgeStat.textContent = String(creature.age_ticks);
      selectedBornStat.textContent = String(creature.born_tick);
      selectedGenomeHashStat.textContent = creature.genome_hash || "-";
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

    followToggle.addEventListener("input", (event) => {{
      followSelected = Boolean(event.target.checked);
      if (!followSelected) {{
        ambientMode = false;
        ambientToggle.checked = false;
      }}
      renderCurrent();
    }});

    ambientToggle.addEventListener("input", (event) => {{
      ambientMode = Boolean(event.target.checked);
      if (ambientMode) {{
        followSelected = true;
        followToggle.checked = true;
        ambientFrameCounter = 0;
      }}
      renderCurrent();
    }});

    zoom.addEventListener("input", (event) => {{
      zoomLevel = Number(event.target.value);
      renderCurrent();
    }});

    fieldMode.addEventListener("input", (event) => {{
      activeFieldMode = event.target.value;
      renderCurrent();
    }});

    canvas.addEventListener("click", (event) => {{
      const snapshot = snapshots[frame];
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const clickX = (event.clientX - rect.left) * scaleX;
      const clickY = (event.clientY - rect.top) * scaleY;
      let best = null;
      let bestDistance = Infinity;
      for (const creature of snapshot.creatures) {{
        const [cx, cy] = toCanvas(creature.center_x, creature.center_y, snapshot);
        const distance = Math.hypot(clickX - cx, clickY - cy);
        if (distance < bestDistance) {{
          best = creature;
          bestDistance = distance;
        }}
      }}
      if (best && bestDistance <= 28) {{
        selectedCreatureId = best.creature_id;
        ambientMode = false;
        ambientToggle.checked = false;
        followSelected = true;
        followToggle.checked = true;
        renderCurrent();
      }}
    }});

    window.addEventListener("keydown", (event) => {{
      if (event.code === "Space") {{
        event.preventDefault();
        toggle.click();
      }} else if (event.code === "ArrowRight") {{
        event.preventDefault();
        step.click();
      }} else if (event.key.toLowerCase() === "f") {{
        event.preventDefault();
        followSelected = !followSelected;
        followToggle.checked = followSelected;
        if (!followSelected) {{
          ambientMode = false;
          ambientToggle.checked = false;
        }}
        renderCurrent();
      }} else if (event.key.toLowerCase() === "a") {{
        event.preventDefault();
        ambientMode = !ambientMode;
        ambientToggle.checked = ambientMode;
        if (ambientMode) {{
          followSelected = true;
          followToggle.checked = true;
          ambientFrameCounter = 0;
        }}
        renderCurrent();
      }} else if (event.key.toLowerCase() === "g") {{
        event.preventDefault();
        cycleFieldMode();
      }} else if (event.key.toLowerCase() === "c") {{
        event.preventDefault();
        activeFieldMode = activeFieldMode === "chemical_a" ? "chemical_b" : "chemical_a";
        fieldMode.value = activeFieldMode;
        renderCurrent();
      }} else if (event.key === "=" || event.key === "+") {{
        event.preventDefault();
        zoomLevel = Math.min(12, zoomLevel + 0.5);
        zoom.value = String(zoomLevel);
        renderCurrent();
      }} else if (event.key === "-") {{
        event.preventDefault();
        zoomLevel = Math.max(1, zoomLevel - 0.5);
        zoom.value = String(zoomLevel);
        renderCurrent();
      }} else if (event.key === "Home") {{
        event.preventDefault();
        followSelected = false;
        ambientMode = false;
        zoomLevel = 1;
        followToggle.checked = false;
        ambientToggle.checked = false;
        zoom.value = String(zoomLevel);
        renderCurrent();
      }} else if (event.key === "1") {{
        event.preventDefault();
        playbackRate = 1;
        speed.value = "1";
        renderCurrent();
      }} else if (event.key === "2") {{
        event.preventDefault();
        playbackRate = 4;
        speed.value = "4";
        renderCurrent();
      }} else if (event.key === "3") {{
        event.preventDefault();
        playbackRate = 16;
        speed.value = "16";
        renderCurrent();
      }} else if (event.key === "4") {{
        event.preventDefault();
        playbackRate = 32;
        speed.value = "32";
        renderCurrent();
      }}
    }});

    function cycleFieldMode() {{
      const modes = ["combined", "nutrient", "light", "chemical_a", "chemical_b", "detritus"];
      const index = modes.indexOf(activeFieldMode);
      activeFieldMode = modes[(index + 1 + modes.length) % modes.length];
      fieldMode.value = activeFieldMode;
      renderCurrent();
    }}

    function advanceAmbientSelection(snapshot) {{
      if (!ambientMode || snapshot.creatures.length <= 1) {{
        return;
      }}
      ambientFrameCounter += 1;
      if (ambientFrameCounter < 120) {{
        return;
      }}
      ambientFrameCounter = 0;
      const ids = rankedCreatureIds(snapshot).slice(0, Math.min(6, snapshot.creatures.length));
      const currentIndex = ids.indexOf(selectedCreatureId);
      selectedCreatureId = ids[(currentIndex + 1 + ids.length) % ids.length];
    }}

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
      advanceAmbientSelection(snapshots[frame]);
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


def _sample_fields(world: World, *, cols: int = 12, rows: int = 12) -> dict[str, object]:
    return sample_fields(world, cols=cols, rows=rows)


def _history_delta(
    history: list[dict[str, float | int]],
    key: str,
    *,
    window: int,
) -> float:
    if not history:
        return 0.0
    current = float(history[-1].get(key, 0.0))
    earlier_index = max(0, len(history) - 1 - max(1, window))
    earlier = float(history[earlier_index].get(key, 0.0))
    return current - earlier


def _chart_points(
    values: list[float],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [x, y + height, x + width, y + height]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-6)
    points: list[float] = []
    for index, value in enumerate(values):
        px = x + (width * (index / max(len(values) - 1, 1)))
        normalized = (value - min_value) / span
        py = y + height - (normalized * height)
        points.extend((px, py))
    return points


def _draw_creature_bands(
    canvas: object,
    nodes: list[object],
    visual: object | None,
    project: object,
    color_css: str,
) -> None:
    band_count = max(1, min(4, int(getattr(visual, "band_count", 1) or 1)))
    if band_count <= 1:
        return
    projected = [project(node.x, node.y) for node in nodes]
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    band_offset = float(getattr(visual, "band_offset", 0.0) or 0.0)
    vertical_bands = (max_x - min_x) >= (max_y - min_y)
    line_color = _blend_css_color(color_css, alpha=0.28)
    for index in range(band_count):
        offset = ((index + band_offset) % band_count) / band_count
        if vertical_bands:
            x = min_x + ((max_x - min_x) * offset)
            canvas.create_line(x, min_y - 2, x, max_y + 2, fill=line_color, width=1.5)
        else:
            y = min_y + ((max_y - min_y) * offset)
            canvas.create_line(min_x - 2, y, max_x + 2, y, fill=line_color, width=1.5)


def _snapshot_payload(world: World) -> dict[str, object]:
    return snapshot_payload(world)


def _build_html_viewer(
    world: World,
    *,
    steps_per_frame: int,
    frame_delay_ms: int,
    canvas_width: int,
    canvas_height: int,
    max_frames: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    total_frames = max(1, max_frames)
    snapshots = [_snapshot_payload(world)]
    if progress_callback is not None:
        progress_callback(1, total_frames)
    for frame_index in range(total_frames - 1):
        world.step(steps_per_frame)
        snapshots.append(_snapshot_payload(world))
        if progress_callback is not None:
            progress_callback(frame_index + 2, total_frames)
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
    progress_callback: Callable[[int, int], None] | None = None,
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
            progress_callback=progress_callback,
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

    controls = tk.Frame(root, bg="#111318", padx=10, pady=8)
    controls.pack(fill="x")

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
    field_mode_index = 1
    selected_creature_id: int | None = None
    follow_selected = True
    ambient_mode = False
    zoom_level = 4.0
    ambient_frame_counter = 0
    step_stride = max(1, steps_per_frame)
    history: list[dict[str, float | int]] = []
    last_history_tick: int | None = None
    field_status = tk.StringVar(value="")
    camera_status = tk.StringVar(value="")
    speed_status = tk.StringVar(value="")
    play_button_text = tk.StringVar(value="Pause")

    def _refresh_control_status() -> None:
        field_status.set(f"field={FIELD_MODE_SEQUENCE[field_mode_index]}")
        camera_status.set(
            f"follow={'on' if follow_selected else 'off'} ambient={'on' if ambient_mode else 'off'} zoom={zoom_level:.1f}x"
        )
        speed_status.set(f"stride={step_stride}")
        play_button_text.set("Pause" if running else "Play")

    def _toggle_running(_: object | None = None) -> None:
        nonlocal running
        running = not running
        _refresh_control_status()

    def _single_step(_: object | None = None) -> None:
        nonlocal pending_single_step
        pending_single_step = True
        _refresh_control_status()

    def _cycle_field_mode(_: object | None = None) -> None:
        nonlocal field_mode_index
        field_mode_index = (field_mode_index + 1) % len(FIELD_MODE_SEQUENCE)
        _refresh_control_status()

    def _toggle_follow(_: object | None = None) -> None:
        nonlocal ambient_mode, follow_selected
        follow_selected = not follow_selected
        if not follow_selected:
            ambient_mode = False
        _refresh_control_status()

    def _toggle_ambient(_: object | None = None) -> None:
        nonlocal ambient_frame_counter, ambient_mode, follow_selected
        ambient_mode = not ambient_mode
        if ambient_mode:
            follow_selected = True
            ambient_frame_counter = 0
        _refresh_control_status()

    def _zoom_in(_: object | None = None) -> None:
        nonlocal zoom_level
        zoom_level = min(12.0, zoom_level + 0.5)
        _refresh_control_status()

    def _zoom_out(_: object | None = None) -> None:
        nonlocal zoom_level
        zoom_level = max(1.0, zoom_level - 0.5)
        _refresh_control_status()

    def _reset_camera(_: object | None = None) -> None:
        nonlocal ambient_mode, follow_selected, zoom_level
        follow_selected = False
        ambient_mode = False
        zoom_level = 1.0
        _refresh_control_status()

    def _slow_down(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = max(1, step_stride // 2)
        _refresh_control_status()

    def _speed_up(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = min(256, step_stride * 2)
        _refresh_control_status()

    def _set_stride_1(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = 1
        _refresh_control_status()

    def _set_stride_4(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = 4
        _refresh_control_status()

    def _set_stride_16(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = 16
        _refresh_control_status()

    def _set_stride_64(_: object | None = None) -> None:
        nonlocal step_stride
        step_stride = 64
        _refresh_control_status()

    def _select_creature(event: object) -> None:
        nonlocal selected_creature_id
        snapshot = world.snapshot()
        if not snapshot.creatures:
            selected_creature_id = None
            return
        click_x = float(getattr(event, "x", 0.0))
        click_y = float(getattr(event, "y", 0.0))
        selected = next(
            (creature for creature in snapshot.creatures if creature.creature_id == selected_creature_id),
            None,
        )

        def _project(x: float, y: float) -> tuple[float, float]:
            focus_x = selected.center_x if (follow_selected and selected is not None) else (snapshot.world_width / 2)
            focus_y = selected.center_y if (follow_selected and selected is not None) else (snapshot.world_height / 2)
            scale_x = (canvas_width / max(snapshot.world_width, 1.0)) * zoom_level
            scale_y = (canvas_height / max(snapshot.world_height, 1.0)) * zoom_level
            return (
                ((x - focus_x) * scale_x) + (canvas_width / 2),
                ((y - focus_y) * scale_y) + (canvas_height / 2),
            )

        node_positions: dict[int, list[tuple[float, float]]] = {}
        for node in snapshot.nodes:
            if node.creature_id is None:
                continue
            node_positions.setdefault(node.creature_id, []).append(_project(node.x, node.y))

        best_creature = None
        best_distance = float("inf")
        for creature in snapshot.creatures:
            candidate_points = [ _project(creature.center_x, creature.center_y), *node_positions.get(creature.creature_id, []) ]
            distance = min(math.hypot(click_x - px, click_y - py) for px, py in candidate_points)
            if distance < best_distance:
                best_distance = distance
                best_creature = creature
        click_threshold = max(40.0, 12.0 * zoom_level)
        if best_creature is not None and best_distance <= click_threshold:
            selected_creature_id = best_creature.creature_id
            ambient_mode = False
            follow_selected = True
            _refresh_control_status()

    root.bind("<space>", _toggle_running)
    root.bind("<Right>", _single_step)
    root.bind("f", _toggle_follow)
    root.bind("F", _toggle_follow)
    root.bind("a", _toggle_ambient)
    root.bind("A", _toggle_ambient)
    root.bind("g", _cycle_field_mode)
    root.bind("G", _cycle_field_mode)
    root.bind("+", _zoom_in)
    root.bind("=", _zoom_in)
    root.bind("-", _zoom_out)
    root.bind("<Home>", _reset_camera)
    root.bind("[", _slow_down)
    root.bind("]", _speed_up)
    root.bind("1", _set_stride_1)
    root.bind("2", _set_stride_4)
    root.bind("3", _set_stride_16)
    root.bind("4", _set_stride_64)
    canvas.bind("<Button-1>", _select_creature)

    def _button(*, text: str | None = None, textvariable: object | None = None, command: object) -> object:
        kwargs: dict[str, object] = {
            "bg": "#1d2430",
            "fg": "#e5edf5",
            "activebackground": "#2a3442",
            "activeforeground": "#e5edf5",
            "relief": "flat",
            "padx": 8,
            "pady": 4,
            "command": command,
        }
        if text is not None:
            kwargs["text"] = text
        if textvariable is not None:
            kwargs["textvariable"] = textvariable
        button = tk.Button(controls, **kwargs)
        button.pack(side="left", padx=4)
        return button

    _button(textvariable=play_button_text, command=_toggle_running)
    _button(text="Step", command=_single_step)
    _button(text="Follow", command=_toggle_follow)
    _button(text="Ambient", command=_toggle_ambient)
    _button(text="Field", command=_cycle_field_mode)
    _button(text="Zoom +", command=_zoom_in)
    _button(text="Zoom -", command=_zoom_out)
    _button(text="Speed -", command=_slow_down)
    _button(text="Speed +", command=_speed_up)
    _button(text="Overview", command=_reset_camera)

    for variable in (field_status, camera_status, speed_status):
        tk.Label(
            controls,
            textvariable=variable,
            anchor="w",
            justify="left",
            bg="#111318",
            fg="#9fb0c1",
            font=("Menlo", 10),
            padx=6,
        ).pack(side="left")
    _refresh_control_status()

    def _creature_focus_score(creature: object) -> tuple[float, int]:
        speed = float(getattr(creature, "mean_speed_recent", 0.0))
        energy = float(getattr(creature, "energy", 0.0))
        age = int(getattr(creature, "age_ticks", 0))
        return ((speed * 4.0) + min(energy, 20.0) + (min(age, 600) / 120.0), -int(getattr(creature, "creature_id", 0)))

    def _ranked_creature_ids(snapshot: Snapshot) -> list[int]:
        ranked = sorted(snapshot.creatures, key=_creature_focus_score, reverse=True)
        return [creature.creature_id for creature in ranked]

    def _preferred_creature_id(snapshot: Snapshot) -> int | None:
        ranked_ids = _ranked_creature_ids(snapshot)
        return ranked_ids[0] if ranked_ids else None

    def _silhouette_color(css: str) -> str:
        return _blend_css_color(css, alpha=0.18)

    def _append_history(stats: object) -> None:
        nonlocal last_history_tick
        tick = int(getattr(stats, "tick", 0))
        if last_history_tick == tick:
            return
        last_history_tick = tick
        history.append(
            {
                "population": int(getattr(stats, "population", 0)),
                "species": int(getattr(stats, "species_count", 0)),
                "predators": int(getattr(stats, "predator_count", 0)),
                "births": int(getattr(stats, "births", 0)),
                "deaths": int(getattr(stats, "deaths", 0)),
                "reproductions": int(getattr(stats, "reproductions", 0)),
                "predation_kills": int(getattr(stats, "predation_kills", 0)),
                "environment_perturbations": int(getattr(stats, "environment_perturbations", 0)),
                "diversity": float(getattr(stats, "diversity_index", 0.0)),
            }
        )
        del history[:-160]

    def _draw_hud(*, snapshot: Snapshot, stats: object, selected: object | None) -> None:
        panel_left = 14
        panel_top = 14
        panel_width = 294
        panel_height = 156
        canvas.create_rectangle(
            panel_left,
            panel_top,
            panel_left + panel_width,
            panel_top + panel_height,
            fill="#0f141a",
            outline="#233142",
            width=1,
        )
        summary_lines = [
            (
                f"pop={snapshot.population} species={getattr(stats, 'species_count', 0)} "
                f"diversity={float(getattr(stats, 'diversity_index', 0.0)):.2f}"
            ),
            (
                f"auto={getattr(stats, 'autotroph_count', 0)} herb={getattr(stats, 'herbivore_count', 0)} "
                f"pred={getattr(stats, 'predator_count', 0)}"
            ),
            (
                f"recent births={int(_history_delta(history, 'births', window=24))} "
                f"deaths={int(_history_delta(history, 'deaths', window=24))} "
                f"repr={int(_history_delta(history, 'reproductions', window=24))} "
                f"kills={int(_history_delta(history, 'predation_kills', window=24))}"
            ),
        ]
        if selected is not None:
            summary_lines.append(
                f"selected=#{selected.creature_id} role={selected.trophic_role} energy={selected.energy:.2f} age={selected.age_ticks}"
            )
        for index, line in enumerate(summary_lines):
            canvas.create_text(
                panel_left + 10,
                panel_top + 14 + (index * 18),
                anchor="w",
                text=line,
                fill="#e5edf5",
                font=("Menlo", 10),
            )
        chart_left = panel_left + 10
        chart_top = panel_top + 84
        chart_width = panel_width - 20
        chart_height = 52
        canvas.create_rectangle(
            chart_left,
            chart_top,
            chart_left + chart_width,
            chart_top + chart_height,
            fill="#131b24",
            outline="",
        )
        population_points = _chart_points(
            [float(entry["population"]) for entry in history],
            x=chart_left,
            y=chart_top,
            width=chart_width,
            height=chart_height,
        )
        species_points = _chart_points(
            [float(entry["species"]) for entry in history],
            x=chart_left,
            y=chart_top,
            width=chart_width,
            height=chart_height,
        )
        predator_points = _chart_points(
            [float(entry["predators"]) for entry in history],
            x=chart_left,
            y=chart_top,
            width=chart_width,
            height=chart_height,
        )
        if len(population_points) >= 4:
            canvas.create_line(*population_points, fill="#f4a261", width=2)
        if len(species_points) >= 4:
            canvas.create_line(*species_points, fill="#8ecae6", width=2)
        if len(predator_points) >= 4:
            canvas.create_line(*predator_points, fill="#e63946", width=2)
        canvas.create_text(
            chart_left,
            chart_top + chart_height + 10,
            anchor="w",
            text="pop",
            fill="#f4a261",
            font=("Menlo", 9),
        )
        canvas.create_text(
            chart_left + 42,
            chart_top + chart_height + 10,
            anchor="w",
            text="species",
            fill="#8ecae6",
            font=("Menlo", 9),
        )
        canvas.create_text(
            chart_left + 112,
            chart_top + chart_height + 10,
            anchor="w",
            text="pred",
            fill="#e63946",
            font=("Menlo", 9),
        )

    def _draw(snapshot: Snapshot) -> None:
        nonlocal ambient_frame_counter, selected_creature_id
        canvas.delete("all")
        field_samples = _sample_fields(world)
        stats = world.stats()
        _append_history(stats)
        cell_width = canvas_width / max(int(field_samples["cols"]), 1)
        cell_height = canvas_height / max(int(field_samples["rows"]), 1)
        field_mode = FIELD_MODE_SEQUENCE[field_mode_index]
        if snapshot.creatures and not any(creature.creature_id == selected_creature_id for creature in snapshot.creatures):
            selected_creature_id = _preferred_creature_id(snapshot)
        selected = next(
            (creature for creature in snapshot.creatures if creature.creature_id == selected_creature_id),
            None,
        )
        if ambient_mode and len(snapshot.creatures) > 1:
            ambient_frame_counter += 1
            if ambient_frame_counter >= 120:
                ambient_frame_counter = 0
                creature_ids = _ranked_creature_ids(snapshot)[: min(6, len(snapshot.creatures))]
                current_index = creature_ids.index(selected_creature_id) if selected_creature_id in creature_ids else -1
                selected_creature_id = creature_ids[(current_index + 1 + len(creature_ids)) % len(creature_ids)]
                selected = next(
                    (creature for creature in snapshot.creatures if creature.creature_id == selected_creature_id),
                    None,
                )

        def _project(x: float, y: float) -> tuple[float, float]:
            focus_x = selected.center_x if (follow_selected and selected is not None) else (snapshot.world_width / 2)
            focus_y = selected.center_y if (follow_selected and selected is not None) else (snapshot.world_height / 2)
            scale_x = (canvas_width / max(snapshot.world_width, 1.0)) * zoom_level
            scale_y = (canvas_height / max(snapshot.world_height, 1.0)) * zoom_level
            return (
                ((x - focus_x) * scale_x) + (canvas_width / 2),
                ((y - focus_y) * scale_y) + (canvas_height / 2),
            )

        for row_index, rows in enumerate(
            zip(
                field_samples["nutrient"],
                field_samples["light"],
                field_samples["chemical_a"],
                field_samples["chemical_b"],
                field_samples["detritus"],
                strict=True,
            )
        ):
            nutrient_row, light_row, chemical_a_row, chemical_b_row, detritus_row = rows
            for col_index, values in enumerate(
                zip(nutrient_row, light_row, chemical_a_row, chemical_b_row, detritus_row, strict=True)
            ):
                nutrient, light, chemical_a, chemical_b, detritus = values
                red, green, blue = _field_rgb(field_mode, nutrient, light, chemical_a, chemical_b, detritus)
                canvas.create_rectangle(
                    col_index * cell_width,
                    row_index * cell_height,
                    (col_index + 1) * cell_width,
                    (row_index + 1) * cell_height,
                    fill=f"#{red:02x}{green:02x}{blue:02x}",
                    outline="",
                )

        creature_roles = {
            creature.creature_id: creature.trophic_role
            for creature in snapshot.creatures
        }
        creature_colors = {
            creature.creature_id: _rgb_to_hex(creature.color_rgb)
            for creature in snapshot.creatures
        }
        creature_visuals = {
            creature.creature_id: creature
            for creature in snapshot.creatures
        }
        creature_nodes: dict[int, list[object]] = {}
        for node in snapshot.nodes:
            if node.creature_id is None:
                continue
            creature_nodes.setdefault(node.creature_id, []).append(node)
        for creature_id, nodes in creature_nodes.items():
            visual = creature_visuals.get(creature_id)
            silhouette_scale = visual.silhouette_scale if visual is not None else 1.0
            fill = _silhouette_color(creature_colors.get(creature_id, "#cbd5e1"))
            if len(nodes) == 1:
                cx, cy = _project(nodes[0].x, nodes[0].y)
                radius = max(6.0, nodes[0].radius * 4.0 * silhouette_scale)
                canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=fill,
                    outline="",
                )
                _draw_creature_bands(canvas, nodes, visual, _project, creature_colors.get(creature_id, "#cbd5e1"))
                continue
            if len(nodes) == 2:
                ax, ay = _project(nodes[0].x, nodes[0].y)
                bx, by = _project(nodes[1].x, nodes[1].y)
                canvas.create_line(
                    ax,
                    ay,
                    bx,
                    by,
                    fill=fill,
                    width=max(8.0, (nodes[0].radius + nodes[1].radius) * 3.5 * silhouette_scale),
                    capstyle="round",
                )
                _draw_creature_bands(canvas, nodes, visual, _project, creature_colors.get(creature_id, "#cbd5e1"))
                continue
            centroid_x = sum(node.x for node in nodes) / len(nodes)
            centroid_y = sum(node.y for node in nodes) / len(nodes)
            ordered_nodes = sorted(
                nodes,
                key=lambda node: math.atan2(node.y - centroid_y, node.x - centroid_x),
            )
            polygon_points: list[float] = []
            for node in ordered_nodes:
                dx = node.x - centroid_x
                dy = node.y - centroid_y
                magnitude = math.hypot(dx, dy) or 1.0
                inflate = node.radius * 1.6 * silhouette_scale
                px, py = _project(
                    node.x + ((dx / magnitude) * inflate),
                    node.y + ((dy / magnitude) * inflate),
                )
                polygon_points.extend((px, py))
            if polygon_points:
                canvas.create_polygon(*polygon_points, fill=fill, outline="")
                _draw_creature_bands(canvas, nodes, visual, _project, creature_colors.get(creature_id, "#cbd5e1"))
        for edge in snapshot.edges:
            ax, ay = _project(edge.ax, edge.ay)
            bx, by = _project(edge.bx, edge.by)
            canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill="#62707d" if not edge.has_motor else "#9fb3c8",
                width=1 if not edge.has_motor else 2,
            )

        for node in snapshot.nodes:
            cx, cy = _project(node.x, node.y)
            role = creature_roles.get(node.creature_id)
            outline = creature_colors.get(node.creature_id, "#cbd5e1")
            visual = creature_visuals.get(node.creature_id)
            glyph_scale = visual.glyph_scale if visual is not None else 1.0
            fill = NODE_COLORS.get(node.node_type, "#94a3b8")
            radius = max(2.0, node.radius * 2.0)
            scaled_radius = radius * max(0.85, glyph_scale)
            canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=fill,
                outline=outline,
                width=2,
            )
            if node.node_type == "mouth":
                canvas.create_line(
                    cx - (scaled_radius * 0.8),
                    cy + (scaled_radius * 0.12),
                    cx + (scaled_radius * 0.8),
                    cy + (scaled_radius * 0.12),
                    fill="#101317",
                    width=max(1.0, scaled_radius * 0.3),
                )
            elif node.node_type == "gripper":
                canvas.create_line(
                    cx - (scaled_radius * 0.15),
                    cy - (scaled_radius * 0.15),
                    cx + (scaled_radius * 0.95),
                    cy - (scaled_radius * 0.95),
                    fill="#101317",
                    width=max(1.0, scaled_radius * 0.3),
                )
                canvas.create_line(
                    cx - (scaled_radius * 0.15),
                    cy + (scaled_radius * 0.15),
                    cx + (scaled_radius * 0.95),
                    cy + (scaled_radius * 0.95),
                    fill="#101317",
                    width=max(1.0, scaled_radius * 0.3),
                )
            elif node.node_type == "sensor":
                glyph_radius = max(1.2, scaled_radius * 0.28)
                canvas.create_oval(
                    cx - glyph_radius,
                    cy - glyph_radius,
                    cx + glyph_radius,
                    cy + glyph_radius,
                    fill="#101317",
                    outline="",
                )
            elif node.node_type == "photoreceptor":
                glyph_radius = max(1.4, scaled_radius * 0.52)
                canvas.create_oval(
                    cx - glyph_radius,
                    cy - glyph_radius,
                    cx + glyph_radius,
                    cy + glyph_radius,
                    outline="#101317",
                    width=max(1.0, scaled_radius * 0.22),
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

        if selected is not None:
            sx, sy = _project(selected.center_x, selected.center_y)
            canvas.create_oval(
                sx - 14,
                sy - 14,
                sx + 14,
                sy + 14,
                outline=_rgb_to_hex(selected.color_rgb),
                width=3,
            )
            species_label = (selected.species_id or "species").replace("species-", "s")
            label = f"#{selected.creature_id} {species_label} {selected.trophic_role}"
            left = sx + 16
            top = sy - 30
            text_id = canvas.create_text(
                left + 7,
                top + 11,
                anchor="w",
                text=label,
                fill="#e5edf5",
                font=("Iosevka Aile", 12, "bold"),
            )
            bbox = canvas.bbox(text_id)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                pad = 4
                canvas.create_rectangle(
                    x0 - pad,
                    y0 - pad,
                    x1 + pad,
                    y1 + pad,
                    fill="#101317",
                    outline=_rgb_to_hex(selected.color_rgb),
                    width=1,
                )
                canvas.tag_raise(text_id)

        _draw_hud(snapshot=snapshot, stats=stats, selected=selected)

        overlay.set(
            "\n".join(
                [
                    (
                        f"tick={snapshot.tick} population={snapshot.population} species={stats.species_count} "
                        f"total_energy={snapshot.total_energy:.2f} diversity={stats.diversity_index:.2f}"
                    ),
                    (
                        f"selected=#{selected.creature_id} species={selected.species_id} role={selected.trophic_role} "
                        f"energy={selected.energy:.2f} age={selected.age_ticks}"
                        if selected is not None
                        else "selected=none"
                    ),
                    (
                        f"recent births={int(_history_delta(history, 'births', window=24))} "
                        f"deaths={int(_history_delta(history, 'deaths', window=24))} "
                        f"repr={int(_history_delta(history, 'reproductions', window=24))} "
                        f"kills={int(_history_delta(history, 'predation_kills', window=24))}"
                    ),
                    (
                        f"follow={'on' if follow_selected else 'off'} ambient={'on' if ambient_mode else 'off'} zoom={zoom_level:.1f}x "
                        f"field={field_mode} stride={step_stride}"
                    ),
                    f"space=play/pause right=step click=inspect f=follow a=ambient g=field +/-=zoom [/] or 1-4=speed home=overview frame_delay_ms={frame_delay_ms}",
                ]
            )
        )

    def _frame() -> None:
        nonlocal pending_single_step
        if running or pending_single_step:
            world.step(step_stride)
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
    open_html_in_browser: bool = True,
    html_progress_callback: Callable[[int, int], None] | None = None,
) -> Path | None:
    if backend not in {"auto", "tk", "html"}:
        msg = f"unsupported viewer backend: {backend}"
        raise ValueError(msg)

    if backend == "html":
        html_path = write_html_viewer(
            world,
            path=html_out_path,
            steps_per_frame=steps_per_frame,
            frame_delay_ms=frame_delay_ms,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_frames=max_frames,
            progress_callback=html_progress_callback,
        )
        _open_html_viewer(html_path, enabled=open_html_in_browser)
        return html_path

    try:
        tk = _load_tk()
    except Exception as exc:  # pragma: no cover - platform-dependent
        if backend == "tk":
            raise RuntimeError(
                "Tkinter is required for `animalcula view --viewer-backend tk` on this machine"
            ) from exc
        html_path = write_html_viewer(
            world,
            path=html_out_path,
            steps_per_frame=steps_per_frame,
            frame_delay_ms=frame_delay_ms,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_frames=max_frames,
            progress_callback=html_progress_callback,
        )
        _open_html_viewer(html_path, enabled=open_html_in_browser)
        return html_path

    _launch_tk_viewer(
        world,
        tk=tk,
        steps_per_frame=steps_per_frame,
        frame_delay_ms=frame_delay_ms,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    return None


def _open_html_viewer(path: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:
        return
