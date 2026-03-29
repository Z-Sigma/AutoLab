"""Run training scripts in subprocess (optional Docker)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def run_python(
    script_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    use_docker: bool = False,
    docker_image: str = "autoresearcher-sandbox:latest",
) -> SandboxResult:
    """
    Execute `python script_path` with optional timeout.
    Docker path is optional (requires image build); default is local venv python.
    """
    script_path = Path(script_path).resolve()
    cwd = Path(cwd or script_path.parent).resolve()
    full_env = {**os.environ, **(env or {})}

    if use_docker:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:{cwd}",
            "-w",
            str(cwd),
            docker_image,
            "python",
            str(script_path),
        ]
    else:
        cmd = [sys.executable, str(script_path)]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        dur = time.perf_counter() - t0
        return SandboxResult(
            proc.returncode,
            proc.stdout or "",
            proc.stderr or "",
            dur,
        )
    except subprocess.TimeoutExpired as e:
        dur = time.perf_counter() - t0
        out = (e.stdout or "").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or "").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return SandboxResult(-1, out, err + "\n[timeout]", dur)
