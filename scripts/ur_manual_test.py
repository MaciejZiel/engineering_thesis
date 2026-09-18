"""Standalone UI for cautious, single-joint UR7e connection tests.

This launcher does not open a camera, load MediaPipe, or consume pose tracking.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_robot_arm.robot.manual_test import (  # noqa: E402
    ManualArmTestSession,
    ManualTestSettings,
)
from vision_robot_arm.robot.targets import JOINT_NAMES  # noqa: E402

BACKGROUND = "#111315"
SURFACE = "#191c1f"
SURFACE_RAISED = "#222629"
BORDER = "#303438"
TEXT = "#e9ebed"
MUTED = "#959ba3"
ACCENT = "#dfac80"
SUCCESS = "#96c4ac"
DANGER = "#e88f8f"
DEFAULT_ROBOT_HOST = "10.20.3.20"

PHASE_LABELS = {
    "disconnected": "DISCONNECTED",
    "monitoring": "READ-ONLY MONITOR",
    "prepared": "CONTROL PREPARED · NO MOTION",
    "armed": "MANUAL CONTROL ARMED",
    "fault": "FAULT · RESTART REQUIRED",
}


class ManualTestWindow:
    def __init__(self, root: tk.Tk, host: str, side: str) -> None:
        self.root = root
        self.session = ManualArmTestSession()
        self.host = tk.StringVar(value=host)
        self.side = tk.StringVar(value=side)
        self.joint = tk.StringVar(value="shoulder")
        self.speed = tk.StringVar(value="0.5")
        self.excursion = tk.StringVar(value="0.5")
        self.phase = tk.StringVar(value=PHASE_LABELS["disconnected"])
        self.hint = tk.StringVar(value=self.session.action_hint)
        self.detail = tk.StringVar(value="No robot connection.")
        self.joint_values = {name: tk.StringVar(value="—") for name in JOINT_NAMES}
        self._controls: list[tk.Widget] = []
        self._closed = False
        self._configure_window()
        self._build()
        self.speed.trace_add("write", self._on_speed_changed)
        self._refresh_ui()
        self.root.after(50, self._tick)

    def _configure_window(self) -> None:
        self.root.title("Motion Twin · UR7e Manual Connection Test")
        self.root.geometry("920x650")
        self.root.minsize(820, 610)
        self.root.configure(background=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=SURFACE_RAISED,
            background=SURFACE_RAISED,
            foreground=TEXT,
            bordercolor=BORDER,
            arrowcolor=MUTED,
            padding=7,
        )
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE_RAISED)])
        style.configure(
            "TSpinbox",
            fieldbackground=SURFACE_RAISED,
            background=SURFACE_RAISED,
            foreground=TEXT,
            bordercolor=BORDER,
            arrowcolor=MUTED,
            padding=7,
        )

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=BACKGROUND, padx=30, pady=24)
        header.pack(fill="x")
        tk.Label(
            header,
            text="UR7e manual connection test",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 21, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Camera-free commissioning · one robot · one joint",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(5, 0))

        body = tk.Frame(self.root, bg=BACKGROUND, padx=30)
        body.pack(fill="both", expand=True)
        settings = self._panel(body)
        settings.pack(side="left", fill="both", expand=True, padx=(0, 8))
        state = self._panel(body)
        state.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._build_settings(settings)
        self._build_state(state)

        footer = tk.Frame(self.root, bg=BACKGROUND, padx=30, pady=24)
        footer.pack(fill="x")
        self.stop_button = self._button(
            footer,
            "STOP AND DISCONNECT",
            self._stop,
            bg="#382426",
            fg=DANGER,
            active="#492b2e",
        )
        self.stop_button.pack(side="right")
        tk.Label(
            footer,
            text="Releasing a jog button sends stopj immediately.\nThe 150 ms watchdog remains active.",
            bg=BACKGROUND,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 9),
        ).pack(side="left")

        self.root.bind("<KeyPress-bracketleft>", lambda _event: self._begin_jog(-1))
        self.root.bind("<KeyRelease-bracketleft>", lambda _event: self._end_jog())
        self.root.bind("<KeyPress-bracketright>", lambda _event: self._begin_jog(1))
        self.root.bind("<KeyRelease-bracketright>", lambda _event: self._end_jog())
        self.root.bind("<Escape>", lambda _event: self._stop())

    def _build_settings(self, parent: tk.Frame) -> None:
        self._section_title(parent, "Connection & limits", "Values lock after connection.")
        grid = tk.Frame(parent, bg=SURFACE)
        grid.pack(fill="x", padx=22)
        grid.columnconfigure(1, weight=1)
        self._field_label(grid, "Robot IP", 0)
        host = tk.Entry(
            grid,
            textvariable=self.host,
            bg=SURFACE_RAISED,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            font=("Segoe UI", 10),
        )
        host.grid(row=0, column=1, sticky="ew", pady=7, ipady=8)
        self._controls.append(host)

        self._field_label(grid, "Robot side", 1)
        side = ttk.Combobox(
            grid, textvariable=self.side, values=("right", "left"), state="readonly"
        )
        side.grid(row=1, column=1, sticky="ew", pady=7)
        self._controls.append(side)

        self._field_label(grid, "Joint", 2)
        joint = ttk.Combobox(
            grid, textvariable=self.joint, values=JOINT_NAMES, state="readonly"
        )
        joint.grid(row=2, column=1, sticky="ew", pady=7)
        self._controls.append(joint)

        self._field_label(grid, "Speed", 3)
        speed = ttk.Spinbox(
            grid,
            textvariable=self.speed,
            from_=0.1,
            to=30.0,
            increment=0.1,
            format="%.1f",
        )
        speed.grid(row=3, column=1, sticky="ew", pady=7)
        self._unit(grid, "°/s", 3)
        self._controls.append(speed)
        self.speed_control = speed

        self._field_label(grid, "Max excursion", 4)
        excursion = ttk.Spinbox(
            grid,
            textvariable=self.excursion,
            from_=0.1,
            to=80.0,
            increment=1.0,
            format="%.1f",
        )
        excursion.grid(row=4, column=1, sticky="ew", pady=7)
        self._unit(grid, "± °", 4)
        self._controls.append(excursion)

        actions = tk.Frame(parent, bg=SURFACE, padx=22, pady=22)
        actions.pack(fill="x")
        self.connect_button = self._button(
            actions, "1  CONNECT READ-ONLY", self._connect
        )
        self.connect_button.pack(fill="x", pady=(0, 8))
        self.prepare_button = self._button(
            actions, "2  PREPARE MANUAL CONTROL", self._prepare
        )
        self.prepare_button.pack(fill="x", pady=8)
        self.arm_button = self._button(
            actions,
            "3  CAPTURE CURRENT POSE & ARM",
            self._arm,
            bg=ACCENT,
            fg="#211b16",
            active="#edbd94",
        )
        self.arm_button.pack(fill="x", pady=(8, 0))

    def _build_state(self, parent: tk.Frame) -> None:
        phase_row = tk.Frame(parent, bg=SURFACE, padx=22, pady=18)
        phase_row.pack(fill="x")
        tk.Label(
            phase_row,
            textvariable=self.phase,
            bg=SURFACE,
            fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            phase_row,
            textvariable=self.hint,
            bg=SURFACE,
            fg=TEXT,
            wraplength=355,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 0))

        joints = tk.Frame(parent, bg=SURFACE, padx=22)
        joints.pack(fill="x")
        for index, name in enumerate(JOINT_NAMES):
            row = tk.Frame(joints, bg=SURFACE, pady=5)
            row.pack(fill="x")
            tk.Label(
                row,
                text=f"J{index + 1}  {name.replace('_', ' ').title()}",
                bg=SURFACE,
                fg=MUTED,
                font=("Segoe UI", 9),
            ).pack(side="left")
            tk.Label(
                row,
                textvariable=self.joint_values[name],
                bg=SURFACE,
                fg=TEXT,
                font=("Consolas", 11, "bold"),
            ).pack(side="right")

        jog = tk.Frame(parent, bg=SURFACE, padx=22, pady=20)
        jog.pack(fill="x")
        self.negative = self._button(jog, "HOLD  −    [", None)
        self.negative.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.positive = self._button(jog, "HOLD  +    ]", None)
        self.positive.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self._bind_jog(self.negative, -1)
        self._bind_jog(self.positive, 1)

        detail = tk.Label(
            parent,
            textvariable=self.detail,
            bg=SURFACE_RAISED,
            fg=MUTED,
            wraplength=355,
            justify="left",
            anchor="nw",
            padx=14,
            pady=13,
            font=("Consolas", 8),
        )
        detail.pack(fill="both", expand=True, padx=22, pady=(0, 22))

    @staticmethod
    def _panel(parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )

    @staticmethod
    def _section_title(parent: tk.Widget, title: str, subtitle: str) -> None:
        frame = tk.Frame(parent, bg=SURFACE, padx=22, pady=19)
        frame.pack(fill="x")
        tk.Label(
            frame,
            text=title,
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame,
            text=subtitle,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(3, 0))

    @staticmethod
    def _field_label(parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(
            parent, text=text, bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)
        ).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=7)

    @staticmethod
    def _unit(parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(
            parent, text=text, bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)
        ).grid(row=row, column=2, sticky="w", padx=(7, 0))

    @staticmethod
    def _button(
        parent: tk.Widget,
        text: str,
        command,
        *,
        bg: str = SURFACE_RAISED,
        fg: str = TEXT,
        active: str = "#303438",
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            disabledforeground="#60656b",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=11,
            font=("Segoe UI", 9, "bold"),
        )

    def _settings(self) -> ManualTestSettings:
        try:
            speed = float(self.speed.get().replace(",", "."))
            excursion = float(self.excursion.get().replace(",", "."))
        except ValueError as error:
            raise ValueError("Speed and excursion must be numbers.") from error
        return ManualTestSettings(
            host=self.host.get(),
            side=self.side.get(),
            joint=self.joint.get(),
            speed_deg_s=speed,
            excursion_deg=excursion,
        )

    def _connect(self) -> None:
        self._perform(lambda: self.session.connect_monitor(self._settings()))

    def _prepare(self) -> None:
        self._perform(self.session.prepare_control)

    def _arm(self) -> None:
        self._perform(self.session.arm)

    def _begin_jog(self, direction: int) -> None:
        if not self.session.can_jog:
            return
        self._apply_speed()
        self._perform(lambda: self.session.begin_jog(direction))

    def _apply_speed(self) -> None:
        try:
            speed = float(self.speed.get().replace(",", "."))
        except ValueError as error:
            raise ValueError("Speed must be a number.") from error
        self.session.set_speed(speed)

    def _on_speed_changed(self, *_args) -> None:
        if self.session.phase not in ("prepared", "armed"):
            return
        try:
            self._apply_speed()
        except (ValueError, RuntimeError):
            # Partial values are normal while editing. The complete value is
            # validated again before the next jog command starts.
            pass

    def _end_jog(self) -> None:
        self.session.end_jog()
        self._refresh_ui()

    def _bind_jog(self, button: tk.Button, direction: int) -> None:
        button.bind("<ButtonPress-1>", lambda _event: self._begin_jog(direction))
        button.bind("<ButtonRelease-1>", lambda _event: self._end_jog())
        button.bind("<Leave>", lambda _event: self._end_jog())

    def _perform(self, action) -> None:
        try:
            action()
        except (ValueError, RuntimeError) as error:
            self.detail.set(str(error))
        self._refresh_ui()

    def _tick(self) -> None:
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

    def _refresh_ui(self) -> None:
        phase = self.session.phase
        self.phase.set(PHASE_LABELS[phase])
        self.hint.set(self.session.action_hint)
        if self.session.error:
            self.detail.set(self.session.error)
        editable = phase == "disconnected"
        for control in self._controls:
            if isinstance(control, ttk.Combobox):
                control.configure(state="readonly" if editable else "disabled")
            else:
                control.configure(state="normal" if editable else "disabled")
        # Speed is deliberately adjustable between jogs after the connection
        # settings and excursion limit have been locked.
        self.speed_control.configure(
            state="normal" if phase in ("disconnected", "prepared", "armed") else "disabled"
        )
        self.connect_button.configure(
            state="normal" if phase == "disconnected" else "disabled"
        )
        self.prepare_button.configure(
            state="normal" if phase == "monitoring" else "disabled"
        )
        self.arm_button.configure(
            state="normal" if phase == "prepared" else "disabled"
        )
        jog_state = "normal" if self.session.can_jog else "disabled"
        self.negative.configure(state=jog_state)
        self.positive.configure(state=jog_state)
        self.stop_button.configure(
            state="normal" if phase != "disconnected" else "disabled"
        )

    def _stop(self) -> None:
        try:
            self.session.stop_and_disconnect()
        except (Exception, SystemExit) as error:
            self.detail.set(f"Shutdown error: {error}")
        self._refresh_ui()

    def _close(self) -> None:
        self._closed = True
        self.session.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-free, single-joint UR7e connection and commissioning test."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_ROBOT_HOST,
        help=f"Prefill the robot IP address (default: {DEFAULT_ROBOT_HOST}).",
    )
    parser.add_argument(
        "--side", choices=("right", "left"), default="right", help="Robot label."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    ManualTestWindow(root, args.host, args.side)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
