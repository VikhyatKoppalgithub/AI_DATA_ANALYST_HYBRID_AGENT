"""Sandbox behaviour that the agent loop depends on."""

from __future__ import annotations

import pytest

from analyst.sandbox import Kernel


@pytest.fixture
def kernel(tmp_path):
    k = Kernel(tmp_path, timeout=10)
    k.start()
    yield k
    k.shutdown()


def test_namespace_persists_across_calls(kernel):
    kernel.execute("x = 41")
    result = kernel.execute("x + 1")
    assert result.ok
    assert result.result_repr == "42"


def test_stdout_is_captured(kernel):
    result = kernel.execute("print('hello'); print('world')")
    assert result.ok
    assert result.stdout.splitlines() == ["hello", "world"]


def test_trailing_expression_is_echoed_but_statements_are_not(kernel):
    assert kernel.execute("1 + 1").result_repr == "2"
    assert kernel.execute("y = 5").result_repr is None


def test_traceback_reports_the_correct_line_and_source(kernel):
    """The repair loop is only as good as the traceback it feeds back."""
    result = kernel.execute("a = 1\nb = 2\nraise ValueError('boom')")
    assert not result.ok
    assert result.error["type"] == "ValueError"
    assert 'line 3' in result.error["traceback"]
    assert "raise ValueError('boom')" in result.error["traceback"]


def test_syntax_error_is_reported_without_worker_frames(kernel):
    result = kernel.execute("x = = 1")
    assert not result.ok
    assert result.error["type"] == "SyntaxError"
    assert "worker.py" not in result.error["traceback"]


def test_worker_frames_are_stripped_from_tracebacks(kernel):
    result = kernel.execute("1 / 0")
    assert "worker.py" not in result.error["traceback"]
    assert "<analysis>" in result.error["traceback"]


def test_error_does_not_destroy_the_namespace(kernel):
    kernel.execute("keep = 'safe'")
    kernel.execute("1 / 0")
    assert kernel.execute("keep").result_repr == "'safe'"


def test_new_files_are_detected(kernel):
    result = kernel.execute("open('chart.png', 'w').write('x')")
    assert result.new_files == ["chart.png"]


def test_network_is_blocked(kernel):
    result = kernel.execute("import socket; socket.socket()")
    assert not result.ok
    assert result.error["type"] == "PermissionError"


def test_timeout_kills_and_restarts_the_kernel(kernel):
    result = kernel.execute("import time; time.sleep(30)", timeout=2)
    assert result.timed_out
    assert result.kernel_restarted
    assert kernel.execute("1 + 1").result_repr == "2"  # usable again


def test_bootstrap_is_replayed_after_a_restart(kernel):
    """State is lost on restart; the data load has to come back automatically."""
    kernel.set_bootstrap("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})")
    kernel.execute("scratch = 'gone after restart'")

    kernel.execute("import time; time.sleep(30)", timeout=2)

    assert kernel.execute("len(df)").result_repr == "3"  # bootstrap replayed
    assert not kernel.execute("scratch").ok  # other state genuinely gone


def test_large_output_is_truncated_not_dropped(kernel):
    result = kernel.execute("print('x' * 50_000)")
    assert result.ok
    assert "omitted" in result.stdout
    assert len(result.stdout) < 20_000


def test_rewriting_an_existing_file_is_reported(kernel):
    """Name-only comparison misses a rewrite, so a repeated chart vanishes."""
    write = "import time; time.sleep(0.01); open('chart.png', 'w').write('x')"
    assert kernel.execute(write).new_files == ["chart.png"]
    assert kernel.execute(write).new_files == ["chart.png"]


def test_an_untouched_file_is_not_reported(kernel):
    kernel.execute("open('kept.txt', 'w').write('x')")
    assert kernel.execute("y = 1").new_files == []
