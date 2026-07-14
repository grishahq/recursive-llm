"""Executable example smoke tests."""

import subprocess
import sys
from pathlib import Path


def test_demo_script_runs_from_the_repository_root() -> None:
    """Test the local demo through the same entry point used by a user."""
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository_root / "examples" / "demo.py")],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Total revenue: $13.2M" in result.stdout
    assert "Extracted answer: The total revenue is $6.6M" in result.stdout
