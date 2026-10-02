"""Spawned Python workers do not need to import a model-provider stack."""

import subprocess
import sys
import textwrap


def test_repl_import_and_spawn_do_not_load_litellm(tmp_path):
    script = tmp_path / "standalone_repl.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys

            class RejectProviderImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in {"litellm", "rlm.core"}:
                        raise AssertionError(f"REPL-only worker imported {fullname}")

            sys.meta_path.insert(0, RejectProviderImports())
            import rlm
            from rlm.repl import REPLExecutor

            assert "RLM" in rlm.__all__ and "RLM" in dir(rlm)
            assert "rlm.core" not in sys.modules and "litellm" not in sys.modules

            if __name__ == "__main__":
                with REPLExecutor() as repl:
                    assert repl.execute("6 * 7", {}) == "42"
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_lazy_public_model_export_preserves_import_api():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import rlm
                from rlm import RLM
                from rlm.core import RLM as CoreRLM

                assert rlm.RLM is RLM is CoreRLM
                exports = {}
                exec("from rlm import *", exports)
                assert all(name in exports for name in rlm.__all__)
                assert exports["RLM"] is RLM
                try:
                    rlm.missing_public_name
                except AttributeError as exc:
                    assert "missing_public_name" in str(exc)
                else:
                    raise AssertionError("Unknown package attribute did not raise")
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
