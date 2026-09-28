from __future__ import annotations

from toto_interp.fev_tasks import get_fev_task, list_fev_tasks


def test_official_fev_task_registry_exposes_safe_and_unsafe_tasks():
    safe_names = {task.dataset_config for task in list_fev_tasks(safe_only=True)}
    all_names = {task.dataset_config for task in list_fev_tasks()}

    assert "entsoe_15T" in safe_names
    assert "m5_1D" not in safe_names
    assert "m5_1D" in all_names


def test_official_fev_task_registry_contains_target_and_exogenous_fields():
    task = get_fev_task("rohlik_orders_1D")

    assert task is not None
    assert task.target_fields == ("orders",)
    assert "holiday" in task.known_dynamic_columns
    assert "shutdown" in task.exogenous_fields
