"""Small Tkinter front end for the interactive mobile-robot vision demo.

The GUI starts :mod:`room_navigation_vision` as a child process.  The child
initializes Genesis and its cameras first, then pauses at a stdin handshake so
the instruction field is enabled only after the simulation is ready.  Keeping
the simulation in its own process preserves the existing Genesis/OpenCV window
threading behavior and leaves the original command-line entry point unchanged.

Example::

    python examples/mobile_robot/room_navigation_gui.py \
        --scenario vision_route_showcase \
        --perception-mode open_vocab \
        --open-vocab-backend yolo-world \
        --vision-model models/mobile_robot/open_vocab/yolov8s-world.pt \
        --open-vocab-device auto \
        --open-vocab-infer-conf 0.001 \
        --open-vocab-decision-conf 0.05 \
        --target-lock-conf 0.001 \
        --calibrated-depth \
        --vis --robot-view --annotated-view \
        --save-vision \
        --output-dir out/mobile_robot_nl_gui

The options after the GUI script are the same options accepted by
``room_navigation_vision.py``.  The instruction is entered in the window
instead of being passed with ``--instruction``.
"""

from __future__ import annotations

import argparse
from collections import deque
import queue
from pathlib import Path
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

try:
    from . import room_navigation_vision
except ImportError:  # pragma: no cover - direct script execution
    import room_navigation_vision  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_SCRIPT = Path(__file__).resolve().with_name("room_navigation_vision.py")
READY_MARKER = "GENESIS_READY_FOR_INSTRUCTION"


def build_arg_parser() -> argparse.ArgumentParser:
    """Reuse the vision demo's CLI options and add only GUI presentation options."""

    parser = room_navigation_vision.build_arg_parser()
    parser.description = __doc__
    parser.add_argument(
        "--window-title",
        default="Genesis Mobile Robot - Natural Language Instruction",
        help="GUI window title.",
    )
    parser.add_argument("--window-width", type=int, default=640, help=argparse.SUPPRESS)
    parser.add_argument("--window-height", type=int, default=420, help=argparse.SUPPRESS)
    return parser


def _child_command(args: argparse.Namespace) -> list[str]:
    """Turn the shared Namespace into a child CLI command.

    ``instruction`` and ``vision_prompt`` are intentionally omitted: the GUI
    always supplies one instruction through stdin after the ready handshake.
    """

    omitted = {
        "instruction",
        "vision_prompt",
        "wait_for_instruction",
        "window_title",
        "window_width",
        "window_height",
    }
    command = [sys.executable, str(VISION_SCRIPT)]
    for name, value in vars(args).items():
        if name in omitted or value is None or value is False:
            continue
        option = f"--{name.replace('_', '-')}"
        if value is True:
            command.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                command.extend((option, str(item)))
        else:
            command.extend((option, str(value)))
    command.append("--wait-for-instruction")
    return command


class InstructionWindow:
    """Tkinter window and lifecycle bridge for one Genesis episode."""

    def __init__(self, args: argparse.Namespace):
        if args.window_width <= 0 or args.window_height <= 0:
            raise ValueError("--window-width and --window-height must be positive")

        self.args = args
        self.root = tk.Tk()
        self.root.title(args.window_title)
        self.root.geometry(f"{args.window_width}x{args.window_height}")
        self.root.minsize(520, 320)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="正在启动 Genesis 环境…")
        self.instruction_var = tk.StringVar(value=args.instruction or "")
        self._messages: deque[str] = deque(maxlen=500)
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._closed = False
        self._submitted = False
        self._ready = False

        self._build_widgets()
        self.process = self._start_process()
        self.root.after(100, self._poll_events)

    def _build_widgets(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(frame, text="Genesis 小车自然语言导航", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="环境初始化完成后，在下方输入 instruction，再点击执行。",
        ).grid(row=1, column=0, sticky="w", pady=(6, 10))

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(input_frame, textvariable=self.instruction_var, state="disabled")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", self._submit_event)
        self.submit_button = ttk.Button(
            input_frame,
            text="执行指令",
            command=self.submit,
            state="disabled",
        )
        self.submit_button.grid(row=0, column=1, padx=(0, 8))
        self.reset_button = ttk.Button(
            input_frame,
            text="重置环境",
            command=self.reset,
            state="disabled",
        )
        self.reset_button.grid(row=0, column=2)

        self.log = tk.Text(frame, height=12, state="disabled", wrap=tk.WORD)
        self.log.grid(row=3, column=0, sticky="nsew", pady=(12, 8))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.log.yview)
        scrollbar.grid(row=3, column=1, sticky="ns", pady=(12, 8))
        self.log.configure(yscrollcommand=scrollbar.set)
        ttk.Label(frame, textvariable=self.status_var).grid(row=4, column=0, sticky="w")

    def _start_process(self) -> subprocess.Popen[str]:
        command = _child_command(self.args)
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self._events.put(("error", f"无法启动仿真进程：{exc}"))
            raise

        thread = threading.Thread(target=self._read_output, args=(process,), daemon=True)
        thread.start()
        return process

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._events.put(("output", line.rstrip()))
        self._events.put(("exit", process.wait()))

    def _poll_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "output":
                    line = str(payload)
                    if line == READY_MARKER:
                        self._ready = True
                        self.status_var.set("Genesis 环境已初始化，请输入 instruction。")
                        self.entry.configure(state="normal")
                        self.submit_button.configure(state="normal")
                        self.reset_button.configure(state="disabled")
                        self.entry.focus_set()
                    elif line:
                        self._append_log(line)
                elif event == "exit":
                    return_code = int(payload)
                    if not self._closed:
                        self._ready = False
                        self.status_var.set(
                            "执行完成，请点击“重置环境”后再次执行。"
                            if return_code == 0
                            else f"仿真进程已退出（返回码 {return_code}），可点击“重置环境”重试。"
                        )
                        self.entry.configure(state="disabled")
                        self.submit_button.configure(state="disabled")
                        self.reset_button.configure(state="normal")
                elif event == "error":
                    self.status_var.set(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _append_log(self, line: str) -> None:
        self._messages.append(line)
        self.log.configure(state="normal")
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _submit_event(self, _event: tk.Event) -> str:
        self.submit()
        return "break"

    def submit(self) -> None:
        instruction = self.instruction_var.get().strip()
        if not self._ready or self._submitted:
            return
        if not instruction:
            self.status_var.set("请输入一条非空 instruction。")
            self.entry.focus_set()
            return
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(instruction + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.status_var.set(f"无法发送 instruction：{exc}")
            return
        self._submitted = True
        self.status_var.set("已发送 instruction，小车执行中…")
        self.entry.configure(state="disabled")
        self.submit_button.configure(state="disabled")

    def reset(self) -> None:
        """Start a fresh Genesis process after the previous episode has ended."""

        if self._closed or self.process.poll() is None:
            return
        self._ready = False
        self._submitted = False
        self.instruction_var.set("")
        self.entry.configure(state="disabled")
        self.submit_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.status_var.set("正在重置 Genesis 环境…")
        self._append_log("----- 重置环境，开始新的 episode -----")
        try:
            self.process = self._start_process()
        except OSError as exc:
            self.status_var.set(f"无法重新启动仿真进程：{exc}")
            self.reset_button.configure(state="normal")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

    def show(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.vision_prompt:
        raise SystemExit("GUI 模式通过输入框提供 instruction，请不要同时使用 --vision-prompt。")
    try:
        window = InstructionWindow(args)
    except tk.TclError as exc:
        raise SystemExit(f"无法创建 Tkinter 界面：{exc}") from exc
    window.show()


if __name__ == "__main__":
    main()
