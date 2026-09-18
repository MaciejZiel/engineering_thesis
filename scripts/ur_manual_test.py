"""Desktop panel for camera-free, synchronized UR7e joint testing."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_robot_arm.robot.manual_test import ManualArmTestSession, ManualTestSettings  # noqa: E402
from vision_robot_arm.robot.targets import JOINT_NAMES  # noqa: E402

BG, PANEL, RAISED, BORDER = "#0d0f11", "#15181b", "#1c2024", "#2a2f34"
TEXT, MUTED, ACCENT = "#eef0f2", "#9299a2", "#dda879"
SUCCESS, DANGER = "#83bfa1", "#e18282"
FONT, MONO = "DejaVu Sans", "DejaVu Sans Mono"
DEFAULT_ROBOT_HOST = "10.20.3.20"
PHASE_LABELS = {
    "disconnected": "DISCONNECTED", "monitoring": "READ-ONLY",
    "prepared": "CONTROL READY", "armed": "ARMED", "fault": "FAULT",
}
JOINT_LABELS = {
    "base": "Base", "shoulder": "Shoulder", "elbow": "Elbow",
    "wrist_1": "Wrist 1", "wrist_2": "Wrist 2", "wrist_3": "Wrist 3",
}


class ManualTestWindow:
    def __init__(self, root: tk.Tk, host: str, side: str) -> None:
        self.root = root
        self.session = ManualArmTestSession()
        self.host, self.side = tk.StringVar(value=host), tk.StringVar(value=side)
        self.excursion = tk.StringVar(value="80.0")
        self.phase = tk.StringVar(value=PHASE_LABELS["disconnected"])
        self.hint = tk.StringVar(value=self.session.action_hint)
        self.detail = tk.StringVar(value="Connect read-only to inspect the controller.")
        self.joint_values = {name: tk.StringVar(value="—") for name in JOINT_NAMES}
        self.joint_speeds = {name: tk.StringVar(value="30.0") for name in JOINT_NAMES}
        self.joint_directions = {name: 0 for name in JOINT_NAMES}
        self.direction_buttons = {}
        self.speed_controls, self.locked_controls = [], []
        self._moving = self._closed = False
        self._configure_window()
        self._build()
        self._refresh_ui()
        self.root.after(50, self._tick)

    def _configure_window(self) -> None:
        self.root.title("Motion Twin · UR7e Multi-Joint Control")
        self.root.geometry("1400x860")
        self.root.minsize(1370, 800)
        self.root.configure(background=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Control.TCombobox", fieldbackground=RAISED, background=RAISED,
                        foreground=TEXT, bordercolor=BORDER, arrowcolor=MUTED,
                        padding=8, font=(FONT, 10))
        style.map("Control.TCombobox", fieldbackground=[("readonly", RAISED)])
        style.configure("Control.TSpinbox", fieldbackground=RAISED, background=RAISED,
                        foreground=TEXT, bordercolor=BORDER, arrowcolor=MUTED,
                        padding=7, font=(MONO, 10))
        self.root.bind("<Escape>", lambda _event: self._stop())
        self.root.bind("<KeyPress-space>", self._space_down)
        self.root.bind("<KeyRelease-space>", lambda _event: self._end_jog())

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=BG, padx=26, pady=22)
        shell.pack(fill="both", expand=True)
        header = tk.Frame(shell, bg=BG)
        header.pack(fill="x")
        title = tk.Frame(header, bg=BG)
        title.pack(side="left")
        tk.Label(title, text="UR7e joint control", bg=BG, fg=TEXT,
                 font=(FONT, 22, "bold")).pack(anchor="w")
        tk.Label(title, text="Direct multi-axis commissioning · camera and tracking disabled",
                 bg=BG, fg=MUTED, font=(FONT, 10)).pack(anchor="w", pady=(4, 0))
        status = tk.Frame(header, bg=RAISED, padx=13, pady=8)
        status.pack(side="right")
        self.status_dot = tk.Label(status, text="●", bg=RAISED, fg=MUTED, font=(FONT, 9))
        self.status_dot.pack(side="left", padx=(0, 7))
        tk.Label(status, textvariable=self.phase, bg=RAISED, fg=TEXT,
                 font=(FONT, 9, "bold")).pack(side="left")
        content = tk.Frame(shell, bg=BG)
        content.pack(fill="both", expand=True, pady=(20, 0))
        sidebar, workspace = self._panel(content, 300), self._panel(content)
        sidebar.pack(side="left", fill="y", padx=(0, 14))
        workspace.pack(side="left", fill="both", expand=True)
        self._build_connection(sidebar)
        self._build_workspace(workspace)

    def _build_connection(self, parent: tk.Widget) -> None:
        self._section(parent, "Connection", "One controller at a time")
        form = tk.Frame(parent, bg=PANEL, padx=18)
        form.pack(fill="x")
        self._label(form, "Robot IP")
        host = tk.Entry(form, textvariable=self.host, bg=RAISED, fg=TEXT,
                        insertbackground=TEXT, relief="flat", highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=ACCENT, font=(MONO, 10))
        host.pack(fill="x", ipady=8, pady=(5, 12))
        self.locked_controls.append(host)
        self._label(form, "Robot label")
        side = ttk.Combobox(form, textvariable=self.side, values=("right", "left"),
                            state="readonly", style="Control.TCombobox")
        side.pack(fill="x", pady=(5, 12))
        self.locked_controls.append(side)
        self._label(form, "Maximum excursion from captured pose")
        excursion = ttk.Spinbox(form, textvariable=self.excursion, from_=0.1, to=80.0,
                                increment=1.0, format="%.1f", style="Control.TSpinbox")
        excursion.pack(fill="x", pady=(5, 4))
        self.locked_controls.append(excursion)
        tk.Label(form, text="degrees · shared by all six axes", bg=PANEL, fg=MUTED,
                 font=(FONT, 8)).pack(anchor="w")
        actions = tk.Frame(parent, bg=PANEL, padx=18, pady=20)
        actions.pack(fill="x")
        self.connect_button = self._button(actions, "1   Connect read-only", self._connect)
        self.prepare_button = self._button(actions, "2   Prepare control", self._prepare)
        self.arm_button = self._button(actions, "3   Capture pose & arm", self._arm,
                                       bg=ACCENT, fg="#241a13", active="#efbd90")
        for button, pady in ((self.connect_button, (0, 6)), (self.prepare_button, 6),
                             (self.arm_button, (6, 0))):
            button.pack(fill="x", pady=pady)
        info = tk.Frame(parent, bg=PANEL, padx=18, pady=4)
        info.pack(fill="both", expand=True)
        tk.Label(info, text="CONTROLLER", bg=PANEL, fg=MUTED,
                 font=(FONT, 8, "bold")).pack(anchor="w", pady=(8, 7))
        tk.Label(info, textvariable=self.detail, bg=RAISED, fg=MUTED, justify="left",
                 anchor="nw", wraplength=250, padx=12, pady=11,
                 font=(MONO, 8)).pack(fill="both", expand=True, pady=(0, 18))

    def _build_workspace(self, parent: tk.Widget) -> None:
        self._section(parent, "Joint mixer",
                      "Set direction and speed per axis, then hold the global dead-man")
        table = tk.Frame(parent, bg=PANEL, padx=18)
        table.pack(fill="x")
        for text, column in (("AXIS", 0), ("LIVE POSITION", 2), ("DIRECTION", 3), ("SPEED", 6)):
            tk.Label(table, text=text, bg=PANEL, fg=MUTED,
                     font=(FONT, 8, "bold")).grid(row=0, column=column, sticky="w", pady=(0, 8))
        table.columnconfigure(1, weight=1)
        table.columnconfigure(2, minsize=120)
        table.columnconfigure(6, minsize=105)
        for index, name in enumerate(JOINT_NAMES, 1):
            self._joint_row(table, index, name)
        motion = tk.Frame(parent, bg=PANEL, padx=18, pady=18)
        motion.pack(fill="x")
        self.move_button = self._button(motion, "HOLD TO MOVE SELECTED JOINTS   ·   SPACE",
                                        None, bg=ACCENT, fg="#241a13",
                                        active="#efbd90", pady=15)
        self.move_button.pack(fill="x")
        self.move_button.bind("<ButtonPress-1>", lambda _event: self._begin_multi_jog())
        self.move_button.bind("<ButtonRelease-1>", lambda _event: self._end_jog())
        self.move_button.bind("<Leave>", lambda _event: self._end_jog())
        footer = tk.Frame(parent, bg=PANEL, padx=18, pady=10)
        footer.pack(fill="x")
        tk.Label(footer, textvariable=self.hint, bg=PANEL, fg=MUTED,
                 font=(FONT, 9)).pack(side="left")
        self.stop_button = self._button(footer, "STOP & DISCONNECT   ESC", self._stop,
                                        bg="#352124", fg=DANGER, active="#48282c")
        self.stop_button.pack(side="right")

    def _joint_row(self, table: tk.Frame, row: int, name: str) -> None:
        bg = PANEL if row % 2 else "#181b1f"
        tk.Label(table, text=f"J{row}", bg=bg, fg=ACCENT, width=4, anchor="w",
                 font=(MONO, 9, "bold")).grid(row=row, column=0, sticky="nsew", pady=2, ipady=11)
        tk.Label(table, text=JOINT_LABELS[name], bg=bg, fg=TEXT, anchor="w",
                 font=(FONT, 10, "bold")).grid(row=row, column=1, sticky="nsew", pady=2)
        tk.Label(table, textvariable=self.joint_values[name], bg=bg, fg=TEXT, anchor="e",
                 padx=14, font=(MONO, 10)).grid(row=row, column=2, sticky="nsew", pady=2)
        buttons = (
            self._dir_button(table, "−", lambda n=name: self._select_direction(n, -1)),
            self._dir_button(table, "■", lambda n=name: self._select_direction(n, 0)),
            self._dir_button(table, "+", lambda n=name: self._select_direction(n, 1)),
        )
        for column, button in enumerate(buttons, 3):
            button.grid(row=row, column=column, padx=2, pady=7, sticky="nsew")
        self.direction_buttons[name] = buttons
        speed = ttk.Spinbox(table, textvariable=self.joint_speeds[name], from_=0.1, to=30.0,
                            increment=0.5, format="%.1f", width=8, style="Control.TSpinbox")
        speed.grid(row=row, column=6, sticky="ew", padx=(10, 0), pady=7)
        self.speed_controls.append(speed)
        tk.Label(table, text="°/s", bg=bg, fg=MUTED, font=(FONT, 8)).grid(
            row=row, column=7, sticky="w", padx=(6, 0))
        self._paint_direction(name)

    @staticmethod
    def _panel(parent, width=None):
        panel = tk.Frame(parent, bg=PANEL, width=width or 1, highlightthickness=1,
                         highlightbackground=BORDER)
        if width:
            panel.pack_propagate(False)
        return panel

    @staticmethod
    def _section(parent, title, subtitle):
        frame = tk.Frame(parent, bg=PANEL, padx=18, pady=17)
        frame.pack(fill="x")
        tk.Label(frame, text=title, bg=PANEL, fg=TEXT,
                 font=(FONT, 12, "bold")).pack(anchor="w")
        tk.Label(frame, text=subtitle, bg=PANEL, fg=MUTED,
                 font=(FONT, 8)).pack(anchor="w", pady=(3, 0))

    @staticmethod
    def _label(parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=MUTED,
                 font=(FONT, 8, "bold")).pack(anchor="w")

    @staticmethod
    def _button(parent, text, command, *, bg=RAISED, fg=TEXT,
                active="#292e33", pady=10):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=active, activeforeground=fg,
                         disabledforeground="#60666d", relief="flat", bd=0,
                         cursor="hand2", padx=13, pady=pady, font=(FONT, 9, "bold"))

    @staticmethod
    def _dir_button(parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=RAISED, fg=MUTED,
                         activebackground="#33271e", activeforeground=ACCENT,
                         relief="flat", bd=0, cursor="hand2", width=4,
                         font=(FONT, 10, "bold"))

    def _speed_for(self, joint):
        speed = float(self.joint_speeds[joint].get().replace(",", "."))
        if not 0 < speed <= 30:
            raise ValueError(f"{JOINT_LABELS[joint]} speed must be between 0 and 30°/s.")
        return speed

    def _settings(self):
        try:
            excursion = float(self.excursion.get().replace(",", "."))
            max_speed = max(self._speed_for(name) for name in JOINT_NAMES)
        except ValueError as error:
            raise ValueError(str(error) or "Speeds and excursion must be valid numbers.") from error
        return ManualTestSettings(host=self.host.get(), side=self.side.get(), joint="shoulder",
                                  speed_deg_s=max_speed, excursion_deg=excursion)

    def _selected_velocities(self):
        return {name: direction * self._speed_for(name)
                for name, direction in self.joint_directions.items() if direction}

    def _select_direction(self, joint, direction):
        if self.session.can_jog:
            self.joint_directions[joint] = direction
            self._paint_direction(joint)

    def _paint_direction(self, joint):
        for direction, button in zip((-1, 0, 1), self.direction_buttons[joint]):
            active = direction == self.joint_directions[joint]
            button.configure(bg="#33271e" if active else RAISED,
                             fg=ACCENT if active else MUTED)

    def _connect(self): self._perform(lambda: self.session.connect_monitor(self._settings()))
    def _prepare(self): self._perform(self.session.prepare_control)
    def _arm(self): self._perform(self.session.arm)

    def _begin_multi_jog(self):
        if not self.session.can_jog or self._moving:
            return
        try:
            self.session.begin_multi_jog(self._selected_velocities())
            self._moving = True
            self.move_button.configure(bg=SUCCESS, activebackground=SUCCESS)
        except (ValueError, RuntimeError) as error:
            self.detail.set(str(error))

    def _end_jog(self):
        if self._moving:
            self._moving = False
            self.session.end_jog()
            self.move_button.configure(bg=ACCENT, activebackground="#efbd90")
            self._refresh_ui()

    def _space_down(self, event):
        if not isinstance(event.widget, (tk.Entry, ttk.Spinbox, ttk.Combobox)):
            self._begin_multi_jog()

    def _perform(self, action):
        try:
            action()
        except (ValueError, RuntimeError) as error:
            self.detail.set(str(error))
        self._refresh_ui()

    def _tick(self):
        if self._closed:
            return
        state = self.session.tick()
        side = self.session.settings.side if self.session.settings else self.side.get()
        arm = state.arm(side) if state is not None else None
        for name, value in self.joint_values.items():
            angle = arm.joints.get(name) if arm is not None else None
            value.set("—" if angle is None else f"{angle:+8.2f}°")
        if self.session.phase != "disconnected":
            self.detail.set("\n".join(self.session.status_lines()))
        self._refresh_ui()
        self.root.after(50, self._tick)

    def _refresh_ui(self):
        phase = self.session.phase
        self.phase.set(PHASE_LABELS[phase])
        self.hint.set(self.session.action_hint)
        self.status_dot.configure(fg={"disconnected": MUTED, "monitoring": SUCCESS,
                                      "prepared": ACCENT, "armed": SUCCESS,
                                      "fault": DANGER}[phase])
        if self.session.error:
            self.detail.set(self.session.error)
        editable = phase == "disconnected"
        for control in self.locked_controls:
            if isinstance(control, ttk.Combobox):
                control.configure(state="readonly" if editable else "disabled")
            else:
                control.configure(state="normal" if editable else "disabled")
        speed_state = "normal" if phase in ("disconnected", "prepared", "armed") else "disabled"
        for control in self.speed_controls:
            control.configure(state=speed_state)
        self.connect_button.configure(state="normal" if phase == "disconnected" else "disabled")
        self.prepare_button.configure(state="normal" if phase == "monitoring" else "disabled")
        self.arm_button.configure(state="normal" if phase == "prepared" else "disabled")
        jog_state = "normal" if self.session.can_jog else "disabled"
        self.move_button.configure(state=jog_state)
        for buttons in self.direction_buttons.values():
            for button in buttons:
                button.configure(state=jog_state)
        self.stop_button.configure(state="normal" if phase != "disconnected" else "disabled")

    def _stop(self):
        self._moving = False
        try:
            self.session.stop_and_disconnect()
        except (Exception, SystemExit) as error:
            self.detail.set(f"Shutdown error: {error}")
        for name in JOINT_NAMES:
            self.joint_directions[name] = 0
            self._paint_direction(name)
        self._refresh_ui()

    def _close(self):
        self._closed = True
        self.session.close()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Synchronized UR7e joint commissioning panel.")
    parser.add_argument("--host", default=DEFAULT_ROBOT_HOST)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    ManualTestWindow(root, args.host, args.side)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
