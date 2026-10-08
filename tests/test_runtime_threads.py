import os
import subprocess
import sys

from utils.runtime_threads import NUMERICAL_THREADS, candidate_thread_environment, configure_controller_threads


def test_controller_does_not_consume_candidate_thread_budget():
    env = {name: "18" for name in NUMERICAL_THREADS}
    env["unrelated"] = "preserved"
    configure_controller_threads(env)
    assert all(env[name] == "2" for name in NUMERICAL_THREADS)
    child = candidate_thread_environment(env)
    assert all(child[name] == "18" for name in NUMERICAL_THREADS)
    assert child["unrelated"] == "preserved"
    assert all(env[name] == "2" for name in NUMERICAL_THREADS)


def test_low_existing_budget_is_preserved_and_repeated_setup_is_stable():
    env = {name: "1" for name in NUMERICAL_THREADS}
    configure_controller_threads(env)
    configure_controller_threads(env)
    assert all(env[name] == "1" for name in NUMERICAL_THREADS)
    assert all(candidate_thread_environment(env)[name] == "1" for name in NUMERICAL_THREADS)


def test_search_import_keeps_controller_small_without_loading_torch():
    environment = dict(os.environ)
    environment.update({name: "18" for name in NUMERICAL_THREADS})
    code = "import run,sys,os; assert 'torch' not in sys.modules; assert os.environ['OPENBLAS_NUM_THREADS']=='2'; from utils.runtime_threads import candidate_thread_environment; assert candidate_thread_environment(os.environ)['OPENBLAS_NUM_THREADS']=='18'"
    child = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
