"""Tests for the resume runner (scripts/resume_finetune_local.py).

Fast: every heavy function the runner calls is mocked, so no model loads.
The runner must reuse run_all_local's functions and run ONLY the remaining
steps (fine-tune answers, fine-tune traces, generalization, aggregate) --
never baseline or CoT.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "scripts" / "resume_finetune_local.py"


def _load_runner():
    """Load scripts/resume_finetune_local.py as a module (scripts is not a package)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("resume_finetune_local", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patched_run(argv):
    """Run the runner's main with argv, mocking every heavy boundary function.

    Returns the dict of mocks plus a recorder list capturing call order.
    """
    runner = _load_runner()
    order = []

    def rec(name):
        def _f(*args, **kwargs):
            order.append((name, args, kwargs))
        return _f

    m_ft = mock.Mock(side_effect=rec("run_finetune"))
    m_gen = mock.Mock(side_effect=rec("run_generalization"))
    m_agg = mock.Mock(side_effect=rec("aggregate"))
    m_base = mock.Mock(side_effect=rec("run_baseline"))
    m_cot = mock.Mock(side_effect=rec("run_cot"))
    m_getdev = mock.Mock(return_value="sentinel-device")

    # Patch where the runner looks them up: on the run_all_local module object,
    # and config.get_device on the runner's config reference.
    ral = runner.run_all_local
    with mock.patch.object(ral, "run_finetune", m_ft), \
         mock.patch.object(ral, "run_generalization", m_gen), \
         mock.patch.object(ral, "aggregate", m_agg), \
         mock.patch.object(ral, "run_baseline", m_base), \
         mock.patch.object(ral, "run_cot", m_cot), \
         mock.patch.object(runner.config, "get_device", m_getdev):
        with mock.patch.object(sys, "argv", ["resume_finetune_local.py", *argv]):
            runner.main()

    return {
        "run_finetune": m_ft,
        "run_generalization": m_gen,
        "aggregate": m_agg,
        "run_baseline": m_base,
        "run_cot": m_cot,
        "get_device": m_getdev,
        "order": order,
    }


def test_runs_only_remaining_steps_in_order():
    r = _patched_run(["--scale", "quick"])

    # fine-tune called exactly twice, answers first then traces.
    # condition may be passed positionally or by keyword.
    assert r["run_finetune"].call_count == 2
    first, second = r["run_finetune"].call_args_list

    def _condition(call):
        return call.kwargs.get("condition", call.args[2] if len(call.args) > 2 else None)

    assert _condition(first) == "finetune_answers"
    assert first.kwargs.get("use_traces") is False
    assert _condition(second) == "finetune_traces"
    assert second.kwargs.get("use_traces") is True

    # generalization + aggregate each called once
    assert r["run_generalization"].call_count == 1
    assert r["aggregate"].call_count == 1

    # baseline and CoT must NOT run (their JSONs already exist)
    assert r["run_baseline"].call_count == 0
    assert r["run_cot"].call_count == 0

    # order: answers -> traces -> generalization -> aggregate
    names = [n for (n, _a, _k) in r["order"]]
    assert names == [
        "run_finetune",
        "run_finetune",
        "run_generalization",
        "aggregate",
    ]


def test_no_device_uses_config_get_device():
    r = _patched_run(["--scale", "quick"])
    r["get_device"].assert_called_once()
    # sentinel device propagates into the heavy calls
    for call in r["run_finetune"].call_args_list:
        assert "sentinel-device" in call.args or call.kwargs.get("device") == "sentinel-device"
    gen_call = r["run_generalization"].call_args
    assert "sentinel-device" in gen_call.args or gen_call.kwargs.get("device") == "sentinel-device"


def test_device_cpu_override_propagates():
    r = _patched_run(["--device", "cpu"])
    # get_device must NOT be consulted when --device is given
    r["get_device"].assert_not_called()
    for call in r["run_finetune"].call_args_list:
        assert "cpu" in call.args or call.kwargs.get("device") == "cpu"
    gen_call = r["run_generalization"].call_args
    assert "cpu" in gen_call.args or gen_call.kwargs.get("device") == "cpu"
