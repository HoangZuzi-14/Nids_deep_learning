from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

import yaml


OpenBinary = Callable[[Path], BinaryIO]


@dataclass
class FileProbeReport:
    path: Path
    exists: bool
    ok: bool
    stat_size: int = 0
    readable_size: int = 0
    line_count: int = 0
    message: str = ""


def _mb(size: int) -> float:
    return size / 1024 / 1024


def probe_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    stat_size: int | None = None,
    open_binary: OpenBinary | None = None,
) -> FileProbeReport:
    path = Path(path)
    if not path.exists():
        return FileProbeReport(path=path, exists=False, ok=False, message="missing")

    try:
        expected_size = int(path.stat().st_size if stat_size is None else stat_size)
        readable_size = 0
        line_count = 0
        opener = open_binary or (lambda p: p.open("rb"))
        with opener(path) as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                readable_size += len(chunk)
                line_count += chunk.count(b"\n")
    except OSError as exc:
        return FileProbeReport(path=path, exists=True, ok=False, message=f"read error: {exc}")

    ok = readable_size == expected_size
    message = (
        "ok"
        if ok
        else (
            "readable bytes differ from filesystem stat size; "
            "the file is likely incomplete, cloud-placeholdered, or sync-corrupted"
        )
    )
    return FileProbeReport(
        path=path,
        exists=True,
        ok=ok,
        stat_size=expected_size,
        readable_size=readable_size,
        line_count=line_count,
        message=message,
    )


def _resolve_project_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def iter_dataset_files(root: Path) -> Iterable[Path]:
    config_path = root / "config" / "datasets.yaml"
    if not config_path.exists():
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for dataset_cfg in (config.get("datasets") or {}).values():
        cache_path = _resolve_project_path(root, dataset_cfg.get("cache_path"))
        if cache_path is not None:
            yield cache_path

        raw_dir = _resolve_project_path(root, dataset_cfg.get("raw_dir"))
        if raw_dir is not None:
            if raw_dir.exists():
                yield from sorted(raw_dir.glob("*.csv"))
            else:
                yield raw_dir


def iter_artifact_files(root: Path) -> Iterable[Path]:
    artifact_dir = root / "artifacts"
    if not artifact_dir.exists():
        return

    extensions = {".json", ".onnx", ".pkl", ".pt", ".pth"}
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions or path.name.endswith(".onnx.data"):
            yield path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_report(report: FileProbeReport, root: Path) -> None:
    status = "OK" if report.ok else "FAIL"
    details = (
        f"stat={_mb(report.stat_size):.3f}MB "
        f"read={_mb(report.readable_size):.3f}MB "
        f"lines={report.line_count}"
        if report.exists
        else "missing"
    )
    print(f"[{status}] {_display_path(report.path, root)} | {details} | {report.message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that local dataset and artifact files are fully readable."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-artifacts", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    paths = list(iter_dataset_files(root))
    if args.include_artifacts:
        paths.extend(iter_artifact_files(root))

    seen: set[Path] = set()
    reports: list[FileProbeReport] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        report = probe_file(resolved)
        reports.append(report)
        print_report(report, root)

    failed = [report for report in reports if not report.ok]
    if failed:
        print(f"\nFound {len(failed)} unreadable or incomplete file(s).")
        return 1

    print(f"\nVerified {len(reports)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
