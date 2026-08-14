"""Parent-side handle for the sandbox worker.

Owns the subprocess lifecycle: start, execute with a wall-clock deadline, kill
and restart on timeout or crash. State (`df`, intermediate variables) lives in
the worker and survives across calls — that persistence is what makes follow-up
questions like "now break that down by region" work without re-reading the file.

A restart wipes that state, so the kernel keeps a bootstrap snippet (typically
the data load) and replays it automatically when it has to come back up.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

WORKER_MODULE = "analyst.sandbox.worker"
STARTUP_TIMEOUT = 30.0


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    result_repr: str | None = None
    error: dict[str, str] | None = None
    new_files: list[str] = field(default_factory=list)
    duration_ms: int = 0
    timed_out: bool = False
    kernel_restarted: bool = False

    @property
    def produced_output(self) -> bool:
        """Did this run actually surface anything to reason from?

        A block of assignments raises no exception and reveals nothing. Treating
        that as a successful computation is what let a model run the same silent
        code five times and then invent an answer.
        """
        return bool(self.stdout.strip() or self.result_repr)

    def to_model_text(self) -> str:
        """Render for the model — the exact text fed back into the repair loop."""
        if self.timed_out:
            return (
                f"EXECUTION TIMED OUT after {self.duration_ms / 1000:.1f}s. The kernel was "
                "restarted and the data reloaded; all other variables are gone. Rewrite "
                "the code to do less work (sample the data, or avoid the expensive step)."
            )
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(f"STDOUT:\n{self.stdout.rstrip()}")
        if self.result_repr:
            parts.append(f"VALUE:\n{self.result_repr}")
        if self.new_files:
            parts.append("FILES WRITTEN: " + ", ".join(self.new_files))
        if self.error:
            parts.append(f"ERROR:\n{self.error['traceback']}")
        if parts:
            return "\n\n".join(parts)
        return (
            "The code ran without error but printed nothing, so it told you "
            "nothing. Assigning to a variable does not show you its value — wrap "
            "what you need in print(), or leave it as the last line on its own."
        )


class SandboxTimeout(RuntimeError):
    pass


class Kernel:
    """A persistent, isolated Python namespace."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        timeout: float = 60.0,
        max_memory_mb: int | None = None,
        max_cpu_seconds: int = 60,
        max_file_mb: int = 64,
        block_network: bool = True,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.timeout = timeout
        self._env_extra = {
            "ANALYST_WORKDIR": str(self.workdir),
            "ANALYST_MAX_MEMORY_MB": str(max_memory_mb or 0),
            "ANALYST_MAX_CPU_SECONDS": str(max_cpu_seconds),
            "ANALYST_MAX_FILE_MB": str(max_file_mb),
            "ANALYST_BLOCK_NETWORK": "1" if block_network else "0",
        }
        self._bootstrap: str | None = None
        self._proc: subprocess.Popen | None = None
        self._result_r: int | None = None
        self._buffer = b""

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        self.workdir.mkdir(parents=True, exist_ok=True)
        result_r, result_w = os.pipe()

        env = {**os.environ, **self._env_extra, "ANALYST_RESULT_FD": str(result_w)}
        src_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_root, env.get("PYTHONPATH", "")]))

        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-m", WORKER_MODULE],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(result_w,),
            env=env,
            cwd=str(self.workdir),
            start_new_session=True,  # own process group, so we can kill the whole tree
        )
        os.close(result_w)
        self._result_r = result_r
        self._buffer = b""

        handshake = self._read_message(deadline=time.monotonic() + STARTUP_TIMEOUT)
        if not handshake or handshake.get("event") != "ready":
            stderr = self._proc.stderr.read().decode("utf-8", "replace") if self._proc.stderr else ""
            self.shutdown()
            raise RuntimeError(f"sandbox worker failed to start.\n{stderr}")

        if self._bootstrap:
            self._exec_once(self._bootstrap, self.timeout)

    def set_bootstrap(self, code: str) -> ExecResult:
        """Run `code` now and replay it on every restart. Use this to load data."""
        self.start()
        result = self._exec_once(code, self.timeout)
        if result.ok:
            self._bootstrap = code
        return result

    def shutdown(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(b'{"op": "shutdown"}\n')
                    proc.stdin.flush()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                self._kill()
        if self._result_r is not None:
            try:
                os.close(self._result_r)
            except OSError:
                pass
            self._result_r = None
        self._proc = None

    def _kill(self) -> None:
        if not self._proc:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self._proc.kill()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def __enter__(self) -> Kernel:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    # ----------------------------------------------------------------- exec

    def execute(self, code: str, timeout: float | None = None) -> ExecResult:
        self.start()
        result = self._exec_once(code, timeout or self.timeout)
        if result.timed_out or self._proc is None or self._proc.poll() is not None:
            self._restart_after_failure(result)
        return result

    def _restart_after_failure(self, result: ExecResult) -> None:
        self.shutdown()
        self.start()  # replays the bootstrap
        result.kernel_restarted = True

    def _exec_once(self, code: str, timeout: float) -> ExecResult:
        assert self._proc is not None and self._proc.stdin is not None
        request = json.dumps({"op": "exec", "code": code, "id": uuid.uuid4().hex})
        started = time.monotonic()

        try:
            self._proc.stdin.write(request.encode() + b"\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return ExecResult(
                ok=False,
                error={"type": "KernelDied", "message": "worker exited", "traceback": "worker exited"},
                duration_ms=0,
            )

        message = self._read_message(deadline=started + timeout)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        if message is None:
            self._kill()
            return ExecResult(ok=False, timed_out=True, duration_ms=elapsed_ms)

        return ExecResult(
            ok=message.get("ok", False),
            stdout=message.get("stdout", ""),
            result_repr=message.get("result_repr"),
            error=message.get("error"),
            new_files=message.get("new_files", []),
            duration_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ io

    def _read_message(self, deadline: float) -> dict | None:
        """Read one newline-delimited JSON message, or None if the deadline passes."""
        assert self._result_r is not None
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._result_r], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self._result_r, 65536)
            if not chunk:  # worker closed the pipe
                return None
            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)
