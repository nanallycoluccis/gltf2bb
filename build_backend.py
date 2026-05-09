from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path


NAME = "gltf2bb"
VERSION = "0.1.0"
SUMMARY = "glTF to Blockbench preparation tools"
REQUIRES_PYTHON = ">=3.14"
ENTRY_POINT = "gltf2bb = src.inspect:main"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
WHEEL_NAME = f"{NAME}-{VERSION}-py3-none-any.whl"
PROJECT_ROOT = Path(__file__).resolve().parent


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _write_metadata(Path(metadata_directory))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _write_metadata(Path(metadata_directory))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_path = Path(wheel_directory) / WHEEL_NAME
    files = {
        path.relative_to(PROJECT_ROOT).as_posix(): path.read_bytes()
        for path in sorted((PROJECT_ROOT / "src").glob("*.py"))
    }
    files.update(
        {
        f"{DIST_INFO}/METADATA": _metadata_text().encode(),
        f"{DIST_INFO}/WHEEL": _wheel_text().encode(),
        f"{DIST_INFO}/entry_points.txt": _entry_points_text().encode(),
        }
    )
    _write_wheel(wheel_path, files)
    return wheel_path.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_path = Path(wheel_directory) / WHEEL_NAME
    files = {
        f"{NAME}.pth": f"{PROJECT_ROOT}\n".encode(),
        f"{DIST_INFO}/METADATA": _metadata_text().encode(),
        f"{DIST_INFO}/WHEEL": _wheel_text().encode(),
        f"{DIST_INFO}/entry_points.txt": _entry_points_text().encode(),
    }
    _write_wheel(wheel_path, files)
    return wheel_path.name


def _write_metadata(metadata_directory: Path) -> str:
    dist_info = metadata_directory / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(_entry_points_text(), encoding="utf-8")
    return DIST_INFO


def _write_wheel(wheel_path: Path, files: dict[str, bytes]) -> None:
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_path, content in files.items():
            wheel.writestr(archive_path, content)
            records.append((archive_path, _hash(content), str(len(content))))

        record_path = f"{DIST_INFO}/RECORD"
        record_lines = [f"{path},{digest},{size}" for path, digest, size in records]
        record_lines.append(f"{record_path},,")
        wheel.writestr(record_path, "\n".join(record_lines).encode())


def _metadata_text() -> str:
    return "\n".join(
        [
            "Metadata-Version: 2.4",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            f"Summary: {SUMMARY}",
            f"Requires-Python: {REQUIRES_PYTHON}",
            "",
        ]
    )


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: gltf2bb-build-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    return "\n".join(["[console_scripts]", ENTRY_POINT, ""])


def _hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"
