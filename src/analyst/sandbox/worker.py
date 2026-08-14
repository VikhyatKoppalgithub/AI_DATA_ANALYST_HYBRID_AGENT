"""Sandbox worker — runs as a subprocess, executes code in a persistent namespace.

Protocol: newline-delimited JSON requests on stdin, responses on a dedicated
result fd handed down by the parent (ANALYST_RESULT_FD). The result fd exists so
that anything the analysed code prints cannot corrupt the protocol stream —
fds 1 and 2 are redirected to a capture file for the duration of each exec.

Never import this module in-process. It calls os.chdir, rewrites fds, and
installs resource limits on the process it runs in.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import linecache
import os
import resource
import sys
import traceback
from pathlib import Path
from typing import Any

WORKER_FILENAME = "<analysis>"
MAX_REPR_CHARS = 4000
MAX_STDOUT_CHARS = 12000


def _apply_limits(max_memory_mb: int | None, max_cpu_seconds: int, max_file_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds + 5))
    file_bytes = max_file_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    # RLIMIT_AS is opt-in: numpy and pandas reserve large virtual address ranges
    # at import time, so an aggressive cap breaks the interpreter before any user
    # code runs. Only meaningful as a backstop against runaway allocation.
    if max_memory_mb:
        mem_bytes = max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass


def _block_network() -> None:
    """Neutralise outbound sockets.

    Defence in depth, not a security boundary: code that reaches for ctypes or
    spawns a subprocess can get around it. The real boundary is running this
    worker inside a container with no network interface.
    """
    import socket

    def _refuse(*_args: Any, **_kwargs: Any):
        raise PermissionError("network access is disabled in the analysis sandbox")

    socket.socket = _refuse  # type: ignore[assignment]
    socket.create_connection = _refuse  # type: ignore[assignment]
    socket.getaddrinfo = _refuse  # type: ignore[assignment]


def _build_namespace() -> dict[str, Any]:
    """Preload the libraries the agent is told it can use."""
    ns: dict[str, Any] = {"__name__": "__analysis__", "__builtins__": builtins}
    preamble = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.max_columns', 60)\n"
    )
    exec(compile(preamble, "<preamble>", "exec"), ns)
    try:
        exec(compile("import duckdb\n", "<preamble>", "exec"), ns)
    except ImportError:
        pass
    return ns


def _register_source(code: str) -> None:
    """Make the executed code visible to linecache.

    Without this, tracebacks for `<analysis>` render with a blank source line,
    because linecache cannot read a filename that is not on disk. The repair
    loop depends on the model seeing the line that actually failed.
    """
    linecache.cache[WORKER_FILENAME] = (
        len(code),
        None,  # mtime None => linecache.checkcache will not evict this entry
        code.splitlines(keepends=True),
        WORKER_FILENAME,
    )


def _clean_traceback(exc: BaseException) -> str:
    """Traceback with worker frames removed, so the model sees only its own code."""
    if isinstance(exc, SyntaxError):
        # Raised at compile time, so there are no useful frames. The standard
        # rendering already carries the file, line, and a caret at the offset.
        return "".join(traceback.format_exception_only(type(exc), exc)).rstrip()

    entries = traceback.extract_tb(exc.__traceback__)
    user_frames = [f for f in entries if f.filename == WORKER_FILENAME]
    lines = ["Traceback (most recent call last):"]
    lines += [
        f'  File "{f.filename}", line {f.lineno}\n    {(f.line or "").strip()}'
        for f in (user_frames or entries[-3:])
    ]
    lines += traceback.format_exception_only(type(exc), exc)
    return "".join(l if l.endswith("\n") else l + "\n" for l in lines).rstrip()


def _split_trailing_expression(code: str) -> tuple[ast.Module, ast.Expression | None]:
    """Split off a trailing bare expression so it can be echoed like a REPL.

    Returns AST nodes rather than source text. Round-tripping through
    ast.unparse would renumber the lines, so a traceback would point at the
    wrong line of the code the model actually wrote. Compiling the AST directly
    keeps the original line numbers intact. Raises SyntaxError on bad input;
    the caller executes this inside its error-capturing block.
    """
    tree = ast.parse(code, filename=WORKER_FILENAME)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        return tree, ast.Expression(body=last.value)
    return tree, None


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Map each file to (mtime_ns, size).

    Comparing names alone misses a rewrite: asking the same chart question twice
    produces the same filename, the set difference is empty, and the second
    chart is silently never surfaced. Identity has to include content, so
    mtime and size travel with the name.
    """
    out: dict[str, tuple[int, int]] = {}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:  # vanished mid-walk
            continue
        out[str(path.relative_to(directory))] = (stat.st_mtime_ns, stat.st_size)
    return out


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n... [{omitted:,} chars omitted] ...\n{text[-half:]}"


def _execute(code: str, ns: dict[str, Any], workdir: Path) -> dict[str, Any]:
    before = _snapshot(workdir)
    capture = io.BytesIO()

    # Redirect at the fd level so output from C extensions and subprocesses is
    # captured too, not just writes that go through sys.stdout.
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    tmp_fd = os.memfd_create("capture") if hasattr(os, "memfd_create") else None
    if tmp_fd is None:
        import tempfile

        tmp = tempfile.TemporaryFile()
        tmp_fd = tmp.fileno()

    error: dict[str, Any] | None = None
    result_repr: str | None = None
    _register_source(code)
    try:
        os.dup2(tmp_fd, 1)
        os.dup2(tmp_fd, 2)
        try:
            body, trailing = _split_trailing_expression(code)
            if body.body:
                exec(compile(body, WORKER_FILENAME, "exec"), ns)
            if trailing is not None:
                value = eval(compile(trailing, WORKER_FILENAME, "eval"), ns)
                if value is not None:
                    result_repr = _truncate(repr(value), MAX_REPR_CHARS)
        except BaseException as exc:  # noqa: BLE001 — surfaced to the model verbatim
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": _clean_traceback(exc)}
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.lseek(tmp_fd, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(tmp_fd, 65536):
            chunks.append(chunk)
        capture.write(b"".join(chunks))
        os.close(tmp_fd)

    # "new" means created *or* rewritten during this execution.
    after = _snapshot(workdir)
    new_files = sorted(name for name, meta in after.items() if before.get(name) != meta)
    return {
        "ok": error is None,
        "stdout": _truncate(capture.getvalue().decode("utf-8", "replace"), MAX_STDOUT_CHARS),
        "result_repr": result_repr,
        "error": error,
        "new_files": new_files,
    }


def main() -> None:
    result_fd = int(os.environ["ANALYST_RESULT_FD"])
    workdir = Path(os.environ.get("ANALYST_WORKDIR", ".")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    _apply_limits(
        max_memory_mb=int(os.environ.get("ANALYST_MAX_MEMORY_MB", "0")) or None,
        max_cpu_seconds=int(os.environ.get("ANALYST_MAX_CPU_SECONDS", "60")),
        max_file_mb=int(os.environ.get("ANALYST_MAX_FILE_MB", "64")),
    )

    namespace = _build_namespace()
    if os.environ.get("ANALYST_BLOCK_NETWORK", "1") == "1":
        _block_network()

    out = os.fdopen(result_fd, "w", buffering=1)
    out.write(json.dumps({"event": "ready"}) + "\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        op = request.get("op")

        if op == "shutdown":
            break
        if op == "ping":
            out.write(json.dumps({"event": "pong"}) + "\n")
            continue
        if op == "exec":
            response = _execute(request["code"], namespace, workdir)
            response["event"] = "result"
            response["id"] = request.get("id")
            out.write(json.dumps(response) + "\n")
            continue

        out.write(json.dumps({"event": "error", "message": f"unknown op: {op}"}) + "\n")


if __name__ == "__main__":
    main()
