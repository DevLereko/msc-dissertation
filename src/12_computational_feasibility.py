"""
12_computational_feasibility.py
Records timing and memory for the framework.
"""

import json
import platform
import socket
import psutil
import os
from pathlib import Path

ARTIFACTS = Path("results/artifacts")

def cpu_model():
    try:
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "Unknown CPU"

def main():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    with open(ARTIFACTS / "training_times.json") as f:
        training_times = json.load(f)

    report = {
        "training_times_seconds": training_times,
        "memory_usage_mb": round(memory_mb, 2),
        "cpu_count": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_model": cpu_model(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
    }
    # Normalise to dissertation-friendly keys
    report["hardware_note"] = (
        f"Consumer-grade {report['cpu_model']} ({report['cpu_physical']} physical / "
        f"{report['cpu_count']} logical cores)"
    )
    with open(ARTIFACTS / "computational_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()