from __future__ import annotations

import io
from pathlib import Path

from scripts.verify_local_files import probe_file


def test_probe_file_marks_complete_read_as_ok(tmp_path):
    path = tmp_path / "complete.bin"
    path.write_bytes(b"abcdef\n")

    report = probe_file(path)

    assert report.exists is True
    assert report.ok is True
    assert report.stat_size == 7
    assert report.readable_size == 7
    assert report.line_count == 1


def test_probe_file_detects_short_read(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"abcdef")

    report = probe_file(
        path,
        stat_size=6,
        open_binary=lambda _path: io.BytesIO(b"abc"),
    )

    assert report.exists is True
    assert report.ok is False
    assert report.stat_size == 6
    assert report.readable_size == 3
    assert "readable bytes differ" in report.message
