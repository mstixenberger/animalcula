"""Live browser frontend for Animalcula."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from animalcula.sim.world import World
from animalcula.viz.payloads import snapshot_payload

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


def build_web_index_html(*, websocket_path: str = "/ws") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Animalcula Live Frontend</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #071017;
      --panel: rgba(11, 18, 25, 0.86);
      --panel-strong: rgba(16, 24, 33, 0.96);
      --line: rgba(255, 255, 255, 0.08);
      --text: #e8f0f7;
      --muted: #9cb2c4;
      --accent: #f6bd60;
      --accent-2: #84dcc6;
      --danger: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(246, 189, 96, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(78, 205, 196, 0.10), transparent 24%),
        linear-gradient(180deg, #04070b, var(--bg));
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
    }}
    .app {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
      gap: 12px;
      padding: 14px;
    }}
    .topbar, .timeline, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      backdrop-filter: blur(14px);
    }}
    .topbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      padding: 12px 14px;
    }}
    .button, select, input[type="range"] {{
      font: inherit;
    }}
    .button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 9px 13px;
      background: #1a2632;
      color: var(--text);
      cursor: pointer;
    }}
    .button.primary {{
      background: var(--accent);
      color: #071017;
      font-weight: 700;
    }}
    .status {{
      color: var(--muted);
      font-size: 0.95rem;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-left: auto;
    }}
    .stage {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 12px;
      min-height: 0;
    }}
    .viewport {{
      position: relative;
      min-height: 72vh;
      background: rgba(5, 10, 15, 0.72);
      border: 1px solid var(--line);
      border-radius: 24px;
      overflow: hidden;
      box-shadow: 0 30px 120px rgba(0, 0, 0, 0.38);
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background: #0b1118;
    }}
    .hud {{
      position: absolute;
      inset: 14px auto auto 14px;
      width: min(360px, calc(100% - 28px));
      display: grid;
      gap: 8px;
      pointer-events: none;
    }}
    .hud-card {{
      background: rgba(11, 18, 25, 0.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .hud-grid, .metric-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      padding: 8px 10px;
      background: rgba(19, 29, 39, 0.82);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 1rem;
      font-weight: 700;
    }}
    .panel {{
      padding: 12px;
      display: grid;
      gap: 10px;
      align-content: start;
    }}
    .panel h2 {{
      margin: 2px 0 0;
      font-size: 1rem;
    }}
    .panel canvas {{
      height: 140px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .timeline {{
      padding: 10px 14px;
      display: grid;
      gap: 8px;
    }}
    .sparkline {{
      width: 100%;
      height: 92px;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 1080px) {{
      .stage {{
        grid-template-columns: 1fr;
      }}
      .viewport {{
        min-height: 58vh;
      }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <section class="topbar">
      <button class="button primary" id="playPause">Pause</button>
      <button class="button" id="step">Step</button>
      <button class="button" id="follow">Follow</button>
      <button class="button" id="ambient">Ambient</button>
      <button class="button" id="field">Field</button>
      <button class="button" id="zoomOut">-</button>
      <button class="button" id="zoomIn">+</button>
      <label class="muted" for="speed">Speed</label>
      <input id="speed" type="range" min="1" max="128" step="1" value="4">
      <span id="speedLabel" class="muted">x4</span>
      <div class="status">
        <span id="connection">connecting</span>
        <span id="tickStat">tick -</span>
        <span id="popStat">pop -</span>
        <span id="speciesStat">species -</span>
      </div>
    </section>
    <section class="stage">
      <section class="viewport">
        <canvas id="world" width="1280" height="900"></canvas>
        <div class="hud">
          <div class="hud-card">
            <div class="metric-grid">
              <div class="metric"><span class="label">Population</span><span class="value" id="hudPopulation">-</span></div>
              <div class="metric"><span class="label">Species</span><span class="value" id="hudSpecies">-</span></div>
              <div class="metric"><span class="label">Diversity</span><span class="value" id="hudDiversity">-</span></div>
              <div class="metric"><span class="label">Births</span><span class="value" id="hudBirths">-</span></div>
              <div class="metric"><span class="label">Deaths</span><span class="value" id="hudDeaths">-</span></div>
              <div class="metric"><span class="label">Kills</span><span class="value" id="hudKills">-</span></div>
            </div>
          </div>
        </div>
      </section>
      <aside class="panel">
        <h2>Inspector</h2>
        <div class="metric-grid">
          <div class="metric"><span class="label">Selected</span><span class="value" id="selectedId">none</span></div>
          <div class="metric"><span class="label">Role</span><span class="value" id="selectedRole">-</span></div>
          <div class="metric"><span class="label">Species</span><span class="value" id="selectedSpecies">-</span></div>
          <div class="metric"><span class="label">Energy</span><span class="value" id="selectedEnergy">-</span></div>
          <div class="metric"><span class="label">Age</span><span class="value" id="selectedAge">-</span></div>
          <div class="metric"><span class="label">Parent</span><span class="value" id="selectedParent">-</span></div>
        </div>
        <canvas id="inspectorHistory" width="320" height="140"></canvas>
        <div class="muted" id="inspectorMeta">click a creature to inspect, scroll to zoom, drag to pan, double-click to follow</div>
      </aside>
    </section>
    <section class="timeline">
      <div class="muted" id="timelineMeta">live ecology timeline</div>
      <canvas class="sparkline" id="timelineCanvas" width="1280" height="92"></canvas>
    </section>
  </main>
  <script>
    const nodeColors = {json.dumps(NODE_COLORS)};
    const roleColors = {json.dumps(ROLE_COLORS)};
    const fieldModes = {json.dumps(FIELD_MODE_SEQUENCE)};
    const socketProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(socketProtocol + "//" + window.location.host + {json.dumps(websocket_path)});
    const canvas = document.getElementById("world");
    const ctx = canvas.getContext("2d");
    const timelineCanvas = document.getElementById("timelineCanvas");
    const timelineCtx = timelineCanvas.getContext("2d");
    const inspectorHistory = document.getElementById("inspectorHistory");
    const inspectorCtx = inspectorHistory.getContext("2d");
    const connection = document.getElementById("connection");
    const playPause = document.getElementById("playPause");
    const stepButton = document.getElementById("step");
    const followButton = document.getElementById("follow");
    const ambientButton = document.getElementById("ambient");
    const fieldButton = document.getElementById("field");
    const zoomInButton = document.getElementById("zoomIn");
    const zoomOutButton = document.getElementById("zoomOut");
    const speedSlider = document.getElementById("speed");
    const speedLabel = document.getElementById("speedLabel");
    const tickStat = document.getElementById("tickStat");
    const popStat = document.getElementById("popStat");
    const speciesStat = document.getElementById("speciesStat");
    const hudPopulation = document.getElementById("hudPopulation");
    const hudSpecies = document.getElementById("hudSpecies");
    const hudDiversity = document.getElementById("hudDiversity");
    const hudBirths = document.getElementById("hudBirths");
    const hudDeaths = document.getElementById("hudDeaths");
    const hudKills = document.getElementById("hudKills");
    const selectedId = document.getElementById("selectedId");
    const selectedRole = document.getElementById("selectedRole");
    const selectedSpecies = document.getElementById("selectedSpecies");
    const selectedEnergy = document.getElementById("selectedEnergy");
    const selectedAge = document.getElementById("selectedAge");
    const selectedParent = document.getElementById("selectedParent");
    const inspectorMeta = document.getElementById("inspectorMeta");
    const timelineMeta = document.getElementById("timelineMeta");

    const state = {{
      snapshot: null,
      paused: false,
      stepStride: 4,
      selectedCreatureId: null,
      followSelected: false,
      ambientMode: false,
      zoomLevel: 1.0,
      fieldMode: "nutrient",
      timeline: [],
      inspectorTimeline: [],
      panX: 0,
      panY: 0,
      dragging: false,
      dragOrigin: null,
    }};

    function rgbToCss(rgb) {{
      if (!Array.isArray(rgb) || rgb.length !== 3) return "#cbd5e1";
      return "rgb(" + rgb.map((value) => Math.max(0, Math.min(255, Math.round(value)))).join(", ") + ")";
    }}

    function blendCssColor(css, alpha, background = [22, 26, 31]) {{
      const match = /^rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)$/.exec(css);
      if (!match) return css;
      const values = match.slice(1).map((value) => Number(value));
      const red = Math.round((values[0] * alpha) + (background[0] * (1 - alpha)));
      const green = Math.round((values[1] * alpha) + (background[1] * (1 - alpha)));
      const blue = Math.round((values[2] * alpha) + (background[2] * (1 - alpha)));
      return `rgb(${{red}}, ${{green}}, ${{blue}})`;
    }}

    function fieldColor(mode, nutrient, light, chemicalA, chemicalB, detritus) {{
      if (mode === "nutrient") return `rgba(76, 175, 80, ${{Math.min(0.34, nutrient * 0.28)}})`;
      if (mode === "light") return `rgba(244, 162, 97, ${{Math.min(0.26, light * 0.18)}})`;
      if (mode === "chemical_a") return `rgba(74, 144, 226, ${{Math.min(0.28, chemicalA * 0.24)}})`;
      if (mode === "chemical_b") return `rgba(181, 93, 255, ${{Math.min(0.28, chemicalB * 0.24)}})`;
      if (mode === "detritus") return `rgba(143, 98, 65, ${{Math.min(0.24, detritus * 0.24)}})`;
      return null;
    }}

    function sendCommand(action, value = null) {{
      if (socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({{ action, value }}));
    }}

    function cameraCenter(snapshot) {{
      if (state.followSelected) {{
        const selected = snapshot.creatures.find((creature) => creature.creature_id === state.selectedCreatureId);
        if (selected) return [selected.center_x, selected.center_y];
      }}
      return [(snapshot.world_width / 2) - state.panX, (snapshot.world_height / 2) - state.panY];
    }}

    function toCanvas(x, y, snapshot) {{
      const [focusX, focusY] = cameraCenter(snapshot);
      const scaleX = (canvas.width / Math.max(snapshot.world_width, 1.0)) * state.zoomLevel;
      const scaleY = (canvas.height / Math.max(snapshot.world_height, 1.0)) * state.zoomLevel;
      return [
        ((x - focusX) * scaleX) + (canvas.width / 2),
        ((y - focusY) * scaleY) + (canvas.height / 2),
      ];
    }}

    function creatureFocusScore(creature) {{
      return (Number(creature.energy || 0) * 0.5) + Math.min(Number(creature.age_ticks || 0), 600) / 120;
    }}

    function syncSelection(snapshot) {{
      if (!snapshot.creatures.length) {{
        state.selectedCreatureId = null;
        return;
      }}
      if (!snapshot.creatures.some((creature) => creature.creature_id === state.selectedCreatureId)) {{
        state.selectedCreatureId = snapshot.creatures.slice().sort((a, b) => creatureFocusScore(b) - creatureFocusScore(a))[0].creature_id;
      }}
      if (state.ambientMode && snapshot.creatures.length > 1 && snapshot.tick % 90 === 0) {{
        const ranked = snapshot.creatures.slice().sort((a, b) => creatureFocusScore(b) - creatureFocusScore(a)).slice(0, 6);
        const ids = ranked.map((creature) => creature.creature_id);
        const currentIndex = ids.indexOf(state.selectedCreatureId);
        state.selectedCreatureId = ids[(currentIndex + 1 + ids.length) % ids.length];
      }}
    }}

    function drawFields(snapshot) {{
      const fields = snapshot.fields;
      if (!fields) return;
      const cellWidth = canvas.width / Math.max(fields.cols, 1);
      const cellHeight = canvas.height / Math.max(fields.rows, 1);
      for (let row = 0; row < fields.rows; row += 1) {{
        for (let col = 0; col < fields.cols; col += 1) {{
          const nutrient = fields.nutrient[row][col] || 0;
          const light = fields.light[row][col] || 0;
          const chemicalA = fields.chemical_a[row][col] || 0;
          const chemicalB = fields.chemical_b[row][col] || 0;
          const detritus = fields.detritus[row][col] || 0;
          if (state.fieldMode === "combined") {{
            const layers = [
              ["rgba(76, 175, 80, ", Math.min(0.16, nutrient * 0.12)],
              ["rgba(244, 162, 97, ", Math.min(0.12, light * 0.08)],
              ["rgba(74, 144, 226, ", Math.min(0.12, chemicalA * 0.12)],
              ["rgba(181, 93, 255, ", Math.min(0.12, chemicalB * 0.12)],
              ["rgba(143, 98, 65, ", Math.min(0.10, detritus * 0.10)]];
            for (const [prefix, alpha] of layers) {{
              if (alpha <= 0) continue;
              ctx.fillStyle = prefix + alpha.toFixed(3) + ")";
              ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth + 1, cellHeight + 1);
            }}
            continue;
          }}
          const fill = fieldColor(state.fieldMode, nutrient, light, chemicalA, chemicalB, detritus);
          if (!fill) continue;
          ctx.fillStyle = fill;
          ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth + 1, cellHeight + 1);
        }}
      }}
    }}

    function drawCreatureSilhouettes(snapshot, creatureColors, creatureVisuals) {{
      const creatureNodes = new Map();
      for (const node of snapshot.nodes) {{
        if (node.creature_id === null || node.creature_id === undefined) continue;
        if (!creatureNodes.has(node.creature_id)) creatureNodes.set(node.creature_id, []);
        creatureNodes.get(node.creature_id).push(node);
      }}
      for (const [creatureId, nodes] of creatureNodes.entries()) {{
        const visual = creatureVisuals.get(creatureId) || {{ silhouette_scale: 1.0, band_count: 2, band_offset: 0.0 }};
        const fill = blendCssColor(creatureColors.get(creatureId) || "rgb(203, 213, 225)", 0.18);
        const silhouetteScale = Number(visual.silhouette_scale || 1.0);
        if (nodes.length === 1) {{
          const [cx, cy] = toCanvas(nodes[0].x, nodes[0].y, snapshot);
          const radius = Math.max(6.0, nodes[0].radius * 4.0 * silhouetteScale);
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.fill();
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
          continue;
        }}
        const centroidX = nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length;
        const centroidY = nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length;
        const ordered = nodes.slice().sort((left, right) => Math.atan2(left.y - centroidY, left.x - centroidX) - Math.atan2(right.y - centroidY, right.x - centroidX));
        ctx.beginPath();
        ordered.forEach((node, index) => {{
          const dx = node.x - centroidX;
          const dy = node.y - centroidY;
          const magnitude = Math.hypot(dx, dy) || 1.0;
          const inflate = node.radius * 1.6 * silhouetteScale;
          const [px, py] = toCanvas(
            node.x + ((dx / magnitude) * inflate),
            node.y + ((dy / magnitude) * inflate),
            snapshot,
          );
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }}
    }}

    function drawHistory(canvasCtx, canvasEl, values, labels) {{
      canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      canvasCtx.fillStyle = "#0b1118";
      canvasCtx.fillRect(0, 0, canvasEl.width, canvasEl.height);
      const plotX = 12;
      const plotY = 12;
      const plotWidth = canvasEl.width - 24;
      const plotHeight = canvasEl.height - 28;
      const peak = Math.max(1, ...values.flatMap((series) => series.values));
      labels.forEach((label, index) => {{
        const series = values[index];
        canvasCtx.beginPath();
        series.values.forEach((value, pointIndex) => {{
          const x = plotX + (plotWidth * (pointIndex / Math.max(series.values.length - 1, 1)));
          const y = plotY + plotHeight - ((value / peak) * plotHeight);
          if (pointIndex === 0) canvasCtx.moveTo(x, y);
          else canvasCtx.lineTo(x, y);
        }});
        canvasCtx.strokeStyle = series.color;
        canvasCtx.lineWidth = 2;
        canvasCtx.stroke();
        canvasCtx.fillStyle = series.color;
        canvasCtx.fillText(label, plotX + (index * 56), canvasEl.height - 6);
      }});
    }}

    function updateInspector(snapshot) {{
      const selected = snapshot.creatures.find((creature) => creature.creature_id === state.selectedCreatureId);
      if (!selected) {{
        selectedId.textContent = "none";
        selectedRole.textContent = "-";
        selectedSpecies.textContent = "-";
        selectedEnergy.textContent = "-";
        selectedAge.textContent = "-";
        selectedParent.textContent = "-";
        inspectorMeta.textContent = "click a creature to inspect, scroll to zoom, drag to pan, double-click to follow";
        return;
      }}
      selectedId.textContent = "#" + selected.creature_id;
      selectedRole.textContent = selected.trophic_role;
      selectedSpecies.textContent = selected.species_id || "-";
      selectedEnergy.textContent = Number(selected.energy || 0).toFixed(2);
      selectedAge.textContent = String(selected.age_ticks);
      selectedParent.textContent = selected.parent_id === null ? "root" : "#" + selected.parent_id;
      state.inspectorTimeline.push(Number(selected.energy || 0));
      state.inspectorTimeline = state.inspectorTimeline.slice(-120);
      drawHistory(inspectorCtx, inspectorHistory, [
        {{ values: state.inspectorTimeline, color: "#f6bd60" }},
      ], ["energy"]);
      inspectorMeta.textContent = `genome ${{selected.genome_hash}}`;
    }}

    function drawSnapshot() {{
      const snapshot = state.snapshot;
      if (!snapshot) return;
      syncSelection(snapshot);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawFields(snapshot);
      const creatureColors = new Map(snapshot.creatures.map((creature) => [creature.creature_id, rgbToCss(creature.color_rgb)]));
      const creatureVisuals = new Map(snapshot.creatures.map((creature) => [creature.creature_id, creature]));
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
        const fill = nodeColors[node.node_type] || "#94a3b8";
        const outline = creatureColors.get(node.creature_id) || "#cbd5e1";
        const radius = Math.max(2, node.radius * 2.1);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = outline;
        ctx.stroke();
      }}
      const selected = snapshot.creatures.find((creature) => creature.creature_id === state.selectedCreatureId);
      if (selected) {{
        const [sx, sy] = toCanvas(selected.center_x, selected.center_y, snapshot);
        ctx.beginPath();
        ctx.arc(sx, sy, 14, 0, Math.PI * 2);
        ctx.strokeStyle = rgbToCss(selected.color_rgb);
        ctx.lineWidth = 3;
        ctx.stroke();
      }}

      const stats = snapshot.stats || {{}};
      tickStat.textContent = `tick ${{snapshot.tick}}`;
      popStat.textContent = `pop ${{snapshot.population}}`;
      speciesStat.textContent = `species ${{stats.species_count ?? 0}}`;
      hudPopulation.textContent = String(snapshot.population);
      hudSpecies.textContent = String(stats.species_count ?? 0);
      hudDiversity.textContent = Number(stats.diversity_index || 0).toFixed(2);
      hudBirths.textContent = String(stats.births ?? 0);
      hudDeaths.textContent = String(stats.deaths ?? 0);
      hudKills.textContent = String(stats.predation_kills ?? 0);
      state.timeline.push({{
        population: Number(snapshot.population || 0),
        species: Number(stats.species_count || 0),
        predators: Number(stats.predator_count || 0),
      }});
      state.timeline = state.timeline.slice(-180);
      drawHistory(timelineCtx, timelineCanvas, [
        {{ values: state.timeline.map((entry) => entry.population), color: "#f6bd60" }},
        {{ values: state.timeline.map((entry) => entry.species), color: "#84dcc6" }},
        {{ values: state.timeline.map((entry) => entry.predators), color: "#ff6b6b" }},
      ], ["pop", "species", "pred"]);
      timelineMeta.textContent = `field=${{state.fieldMode}} follow=${{state.followSelected ? "on" : "off"}} ambient=${{state.ambientMode ? "on" : "off"}} zoom=${{state.zoomLevel.toFixed(1)}}x`;
      updateInspector(snapshot);
    }}

    socket.addEventListener("open", () => {{
      connection.textContent = "live";
      connection.style.color = "#84dcc6";
      sendCommand("set_speed", Number(speedSlider.value));
    }});

    socket.addEventListener("close", () => {{
      connection.textContent = "offline";
      connection.style.color = "#ff6b6b";
    }});

    socket.addEventListener("message", (event) => {{
      const message = JSON.parse(event.data);
      if (message.type === "hello") {{
        state.stepStride = message.controls.step_stride;
        state.paused = message.controls.paused;
        speedSlider.value = String(state.stepStride);
        speedLabel.textContent = `x${{state.stepStride}}`;
        playPause.textContent = state.paused ? "Play" : "Pause";
        return;
      }}
      if (message.type !== "frame") return;
      state.snapshot = message.snapshot;
      state.paused = message.controls.paused;
      state.stepStride = message.controls.step_stride;
      speedSlider.value = String(state.stepStride);
      speedLabel.textContent = `x${{state.stepStride}}`;
      playPause.textContent = state.paused ? "Play" : "Pause";
      drawSnapshot();
    }});

    playPause.addEventListener("click", () => sendCommand("toggle_pause"));
    stepButton.addEventListener("click", () => sendCommand("step"));
    followButton.addEventListener("click", () => {{ state.followSelected = !state.followSelected; drawSnapshot(); }});
    ambientButton.addEventListener("click", () => {{ state.ambientMode = !state.ambientMode; if (state.ambientMode) state.followSelected = true; drawSnapshot(); }});
    fieldButton.addEventListener("click", () => {{
      const index = fieldModes.indexOf(state.fieldMode);
      state.fieldMode = fieldModes[(index + 1 + fieldModes.length) % fieldModes.length];
      drawSnapshot();
    }});
    zoomInButton.addEventListener("click", () => {{ state.zoomLevel = Math.min(10, state.zoomLevel + 0.5); drawSnapshot(); }});
    zoomOutButton.addEventListener("click", () => {{ state.zoomLevel = Math.max(0.5, state.zoomLevel - 0.5); drawSnapshot(); }});
    speedSlider.addEventListener("input", (event) => {{
      const value = Number(event.target.value);
      speedLabel.textContent = `x${{value}}`;
      sendCommand("set_speed", value);
    }});
    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      state.zoomLevel = Math.max(0.5, Math.min(10, state.zoomLevel + (event.deltaY < 0 ? 0.25 : -0.25)));
      drawSnapshot();
    }}, {{ passive: false }});
    canvas.addEventListener("dblclick", () => {{
      state.followSelected = true;
      drawSnapshot();
    }});
    canvas.addEventListener("mousedown", (event) => {{
      state.dragging = true;
      state.dragOrigin = [event.clientX, event.clientY];
    }});
    window.addEventListener("mouseup", () => {{
      state.dragging = false;
      state.dragOrigin = null;
    }});
    window.addEventListener("mousemove", (event) => {{
      if (!state.dragging || !state.dragOrigin || state.followSelected || !state.snapshot) return;
      const dx = event.clientX - state.dragOrigin[0];
      const dy = event.clientY - state.dragOrigin[1];
      state.dragOrigin = [event.clientX, event.clientY];
      const scaleX = (canvas.width / Math.max(state.snapshot.world_width, 1.0)) * state.zoomLevel;
      const scaleY = (canvas.height / Math.max(state.snapshot.world_height, 1.0)) * state.zoomLevel;
      state.panX -= dx / Math.max(scaleX, 1e-6);
      state.panY -= dy / Math.max(scaleY, 1e-6);
      drawSnapshot();
    }});
    canvas.addEventListener("click", (event) => {{
      if (!state.snapshot) return;
      const rect = canvas.getBoundingClientRect();
      const clickX = (event.clientX - rect.left) * (canvas.width / rect.width);
      const clickY = (event.clientY - rect.top) * (canvas.height / rect.height);
      let best = null;
      let bestDistance = Infinity;
      for (const creature of state.snapshot.creatures) {{
        const points = [toCanvas(creature.center_x, creature.center_y, state.snapshot)];
        for (const node of state.snapshot.nodes) {{
          if (node.creature_id === creature.creature_id) points.push(toCanvas(node.x, node.y, state.snapshot));
        }}
        const distance = Math.min(...points.map(([px, py]) => Math.hypot(clickX - px, clickY - py)));
        if (distance < bestDistance) {{
          best = creature;
          bestDistance = distance;
        }}
      }}
      if (best && bestDistance <= Math.max(42, 12 * state.zoomLevel)) {{
        state.selectedCreatureId = best.creature_id;
        state.followSelected = true;
        state.ambientMode = false;
        drawSnapshot();
      }}
    }});
  </script>
</body>
</html>"""


@dataclass(slots=True)
class LiveControls:
    paused: bool
    step_stride: int
    pending_step: bool = False


def create_web_app(
    world: World,
    *,
    target_fps: int = 30,
    default_speed: int = 4,
) -> FastAPI:
    controls = LiveControls(paused=False, step_stride=max(1, default_speed))
    app = FastAPI(title="Animalcula Live Frontend")

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True, "tick": world.tick, "population": len(world.creatures)})

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(build_web_index_html())

    @app.websocket("/ws")
    async def websocket_frames(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "hello",
                "controls": {"paused": controls.paused, "step_stride": controls.step_stride},
            }
        )
        frame_delay = 1.0 / max(1, target_fps)
        try:
            while True:
                if not controls.paused or controls.pending_step:
                    world.step(controls.step_stride)
                    controls.pending_step = False
                await websocket.send_json(
                    {
                        "type": "frame",
                        "snapshot": snapshot_payload(world),
                        "controls": {"paused": controls.paused, "step_stride": controls.step_stride},
                    }
                )
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=frame_delay)
                except TimeoutError:
                    continue
                action = message.get("action")
                value = message.get("value")
                if action == "toggle_pause":
                    controls.paused = not controls.paused
                elif action == "pause":
                    controls.paused = True
                elif action == "play":
                    controls.paused = False
                elif action == "step":
                    controls.pending_step = True
                    controls.paused = True
                elif action == "set_speed":
                    try:
                        controls.step_stride = max(1, min(256, int(value)))
                    except (TypeError, ValueError):
                        continue
        except WebSocketDisconnect:
            return

    return app


def run_web_frontend(
    world: World,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    target_fps: int = 30,
    default_speed: int = 4,
    open_browser: bool = True,
) -> str:
    url = f"http://{host}:{port}/"

    if open_browser:
        def _open_browser() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(url)
            except Exception:
                return

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        create_web_app(world, target_fps=target_fps, default_speed=default_speed),
        host=host,
        port=port,
        log_level="warning",
    )
    return url
