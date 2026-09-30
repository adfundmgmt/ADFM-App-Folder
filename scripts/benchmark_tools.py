"""Measure cold/warm default-page renders in isolated processes.

Run `python scripts/benchmark_tools.py --all --output /tmp/adfm-benchmark.json`.
These are local AppTest measurements, not browser/network latency or a load
test. Every page gets a separate process and a hard timeout. Failure and
explicit provider-unavailable output remain visible in the report.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKER = "ADFM_BENCHMARK="


def parse_worker_output(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            payload = json.loads(line[len(MARKER):])
            if not isinstance(payload, dict) or "status" not in payload:
                raise ValueError("Malformed benchmark result")
            return payload
    raise ValueError("Worker did not return a benchmark result")


def benchmark_page(page: str, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", page,
             "--timeout", str(timeout)],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout * 2 + 15,
            check=False,
        )
        output = parse_worker_output(result.stdout)
        if result.returncode and output.get("status") == "ok":
            output["status"] = "error"
        return output
    except subprocess.TimeoutExpired:
        return {"page": page, "status": "timeout",
                "elapsed_seconds": round(time.perf_counter() - started, 3)}
    except (ValueError, json.JSONDecodeError) as exc:
        return {"page": page, "status": "error", "errors": [str(exc)]}


def run_worker(page: str, timeout: float) -> dict:
    from streamlit.testing.v1 import AppTest

    from adfm_core.catalog import tool_definitions
    from adfm_core.observability import performance_events

    allowed = {tool.page_filename for tool in tool_definitions()}
    if page not in allowed and page != "Home.py":
        raise ValueError("Choose an existing catalog page")
    path = ROOT / "Home.py" if page == "Home.py" else ROOT / "pages" / page
    started = time.perf_counter()
    app = AppTest.from_file(str(path), default_timeout=timeout).run()
    cold = time.perf_counter() - started
    cold_errors = [str(item.message) for item in app.exception]
    cold_provider_errors = len(app.error)
    cold_provider_warnings = len(app.warning)
    started = time.perf_counter()
    if not cold_errors:
        app.run()
    warm = time.perf_counter() - started
    errors = cold_errors or [str(item.message) for item in app.exception]
    # Access the actual app state; an outside bare-mode performance_events()
    # call cannot see this AppTest session's recorded events.
    try:
        events = app.session_state["adfm_performance_events"]
    except KeyError:
        events = []
    del performance_events
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_mib = peak / (1024 if sys.platform != "darwin" else 1024 * 1024)
    return {
        "page": page, "status": "error" if errors else "ok",
        "cold_seconds": round(cold, 4), "warm_seconds": round(warm, 4),
        "peak_process_memory_mib": round(memory_mib, 2),
        "runtime_errors": len(errors), "errors": errors,
        "provider_errors": cold_provider_errors,
        "provider_warnings": cold_provider_warnings,
        "tables": len(app.dataframe), "telemetry_events": len(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--page", action="append", default=[])
    parser.add_argument("--worker")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            result = run_worker(args.worker, args.timeout)
        except Exception as exc:
            result = {"page": args.worker, "status": "error",
                      "errors": [f"{type(exc).__name__}: {exc}"]}
        print(MARKER + json.dumps(result))
        return int(result["status"] != "ok")
    from adfm_core.catalog import tool_definitions

    pages = (["Home.py", *(tool.page_filename for tool in tool_definitions())]
             if args.all else args.page or ["Home.py"])
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda page: benchmark_page(page, args.timeout), pages))
    report = {"measurement": "Local isolated AppTest; not production/browser latency",
              "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(any(result["status"] != "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
