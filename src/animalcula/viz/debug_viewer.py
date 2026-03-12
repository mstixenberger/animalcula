"""Tiny Tkinter debug viewer for local simulation inspection."""

from __future__ import annotations

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


def launch_viewer(
    world: World,
    *,
    steps_per_frame: int = 1,
    frame_delay_ms: int = 33,
    canvas_width: int = 900,
    canvas_height: int = 900,
) -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - platform-dependent
        raise RuntimeError("Tkinter is required for `animalcula view` on this machine") from exc

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

    def _to_canvas(x: float, y: float, snapshot: Snapshot) -> tuple[float, float]:
        return (
            (x / max(snapshot.world_width, 1.0)) * canvas_width,
            (y / max(snapshot.world_height, 1.0)) * canvas_height,
        )

    def _draw(snapshot: Snapshot) -> None:
        canvas.delete("all")

        creature_roles = {
            creature.creature_id: creature.trophic_role
            for creature in snapshot.creatures
        }
        for edge in snapshot.edges:
            ax, ay = _to_canvas(edge.ax, edge.ay, snapshot)
            bx, by = _to_canvas(edge.bx, edge.by, snapshot)
            canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill="#62707d" if not edge.has_motor else "#9fb3c8",
                width=1 if not edge.has_motor else 2,
            )

        for node in snapshot.nodes:
            cx, cy = _to_canvas(node.x, node.y, snapshot)
            role = creature_roles.get(node.creature_id)
            outline = ROLE_COLORS.get(role, "#cbd5e1")
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
