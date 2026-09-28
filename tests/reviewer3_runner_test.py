from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", ["run_reviewer3_suite.py", "run_reviewer3_cpu_followup.py"])
def test_run_reuses_only_an_exact_command_signature(tmp_path: Path, script_name: str) -> None:
    module = _load_script(script_name)
    done_path = tmp_path / "done.txt"
    counter_path = tmp_path / "counter.txt"

    def command(value: str) -> list[str]:
        code = (
            "from pathlib import Path; "
            f"p=Path({str(counter_path)!r}); "
            "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1'); "
            f"Path({str(done_path)!r}).write_text({value!r})"
        )
        return [sys.executable, "-c", code]

    first = command("first")
    keyword_args = {"done_path": done_path, "reuse_existing": True}
    module.run(first, **keyword_args)
    module.run(first, **keyword_args)
    assert counter_path.read_text() == "1"

    module.run(command("changed"), **keyword_args)
    assert counter_path.read_text() == "2"
    assert done_path.read_text() == "changed"


@pytest.mark.parametrize("script_name", ["run_reviewer3_suite.py", "run_reviewer3_cpu_followup.py"])
def test_run_treats_scratch_symlink_aliases_as_the_same_command(
    tmp_path: Path, script_name: str
) -> None:
    module = _load_script(script_name)
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    done_path = real_root / "done.txt"
    done_path.write_text("complete")
    counter_path = real_root / "counter.txt"
    code = f"from pathlib import Path; Path({str(counter_path)!r}).write_text('ran')"
    prior_command = [sys.executable, "-c", code, str(alias_root / "input.pt")]
    signature_path = done_path.with_name(f"{done_path.name}.command.json")
    signature_path.write_text(json.dumps({"command": prior_command}))

    current_command = [sys.executable, "-c", code, str(real_root / "input.pt")]
    if script_name == "run_reviewer3_suite.py":
        module.run(
            current_command,
            done_path=done_path,
            reuse_existing=True,
        )
    else:
        module.run(current_command, done_path, True)

    assert not counter_path.exists()


def test_reviewer3_suite_dispatches_gbdt_confounding_and_structural_holdout(tmp_path: Path, monkeypatch) -> None:
    module = _load_script("run_reviewer3_suite.py")
    captured: list[list[str]] = []

    def capture(command: list[str], *, done_path: Path | None, reuse_existing: bool) -> None:
        captured.append(command)

    monkeypatch.setattr(module, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reviewer3_suite.py",
            "--output-root",
            str(tmp_path / "runs"),
            "--seed",
            "42",
            "--snapshot-path",
            str(tmp_path / "snapshot"),
            "--device",
            "cpu",
            "--structural-holdout-rotation",
            "3",
        ],
    )

    module.main()

    command_text = [" ".join(command) for command in captured]
    assert any("compute_structural_confounding.py" in command for command in command_text)
    assert any("--method gbdt" in command for command in command_text)
    assert sum("run_structural_holdout_probes.py" in command for command in command_text) == 2
    assert all("--rotation 3" in command for command in command_text if "run_structural_holdout_probes.py" in command)
