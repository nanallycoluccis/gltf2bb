from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_DIR = Path("/tmp/gltf2bb-quality")
DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / ".omo" / "evidence" / "task-2-exp-benchmarks.json"
DEFAULT_MODELS = (
    Path("exp") / "pmx-acacia.glb",
    Path("exp") / "pmx-silverwolf.glb",
    Path("exp") / "pmx-lacrimosa.glb",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run exp-model conversion smoke benchmarks and write compact evidence.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        type=Path,
        help="Model paths to convert. Defaults to the selected exp/ PMX GLBs.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="Evidence path. JSON is written for .json, concise text otherwise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_OUTPUT_DIR,
        help="Directory for raw .bbmodel and report JSON outputs.",
    )
    args = parser.parse_args(argv)

    model_paths = tuple(args.models) if args.models else DEFAULT_MODELS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_one(model_path, args.output_dir) for model_path in model_paths]
    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "results": results,
    }
    write_evidence(args.evidence, evidence)

    failed = [result for result in results if result["return_code"] != 0]
    if failed:
        print(
            f"Benchmark smoke failed for {len(failed)} model(s); evidence: {args.evidence}",
            file=sys.stderr,
        )
        return 1

    print(f"Benchmark smoke passed for {len(results)} model(s); evidence: {args.evidence}")
    return 0


def run_one(model_path: Path, output_dir: Path) -> dict[str, Any]:
    stem = model_path.stem
    output_path = output_dir / f"{stem}.bbmodel"
    report_path = output_dir / f"{stem}-report.json"
    command = [
        "uv",
        "run",
        "gltf2bb",
        "convert",
        str(model_path),
        "--mode",
        "hybrid",
        "--report",
        str(report_path),
        "-o",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    summary: dict[str, Any] = {
        "model": str(model_path),
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
        "output_path": str(output_path),
        "output_exists": output_path.exists(),
        "report_path": str(report_path),
        "report_exists": report_path.exists(),
        "cube_count": None,
        "report_quality_keys": [],
        "report_totals_present": False,
        "report_totals_keys": [],
        "budget_warnings": [],
    }

    if output_path.exists():
        summary["cube_count"] = read_cube_count(output_path)
    if report_path.exists():
        report = read_json_object(report_path)
        quality = report.get("quality", {}) if isinstance(report, dict) else {}
        totals = report.get("totals", {}) if isinstance(report, dict) else {}
        summary["report_quality_keys"] = sorted(quality) if isinstance(quality, dict) else []
        summary["report_totals_present"] = isinstance(totals, dict)
        summary["report_totals_keys"] = sorted(totals) if isinstance(totals, dict) else []
        summary["budget_warnings"] = collect_budget_warnings(report)

    return summary


def read_cube_count(path: Path) -> int | None:
    data = read_json_object(path)
    elements = data.get("elements") if isinstance(data, dict) else None
    if isinstance(elements, list):
        return len(elements)
    return None


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def collect_budget_warnings(data: Any) -> list[Any]:
    warnings: list[Any] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = key.lower()
            if "budget" in key_lower and "warning" in key_lower:
                if isinstance(value, list):
                    warnings.extend(value)
                else:
                    warnings.append(value)
            else:
                warnings.extend(collect_budget_warnings(value))
    elif isinstance(data, list):
        for item in data:
            warnings.extend(collect_budget_warnings(item))
    return warnings


def tail_lines(text: str, limit: int = 8) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    lines = [
        f"generated_at: {evidence['generated_at']}",
        f"output_dir: {evidence['output_dir']}",
    ]
    for result in evidence["results"]:
        lines.extend(
            [
                "",
                f"model: {result['model']}",
                f"return_code: {result['return_code']}",
                f"output_exists: {result['output_exists']}",
                f"report_exists: {result['report_exists']}",
                f"cube_count: {result['cube_count']}",
                f"quality_keys: {', '.join(result['report_quality_keys'])}",
                f"totals_present: {result['report_totals_present']}",
                f"budget_warnings: {json.dumps(result['budget_warnings'], ensure_ascii=False)}",
            ]
        )
        if result["stderr_tail"]:
            lines.append("stderr_tail: " + " | ".join(result["stderr_tail"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
