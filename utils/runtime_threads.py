"""Keep orchestration lightweight while preserving candidate numerical thread budgets."""
import os

NUMERICAL_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PREFIX = "ALGOEVOLVE_CANDIDATE_"


def configure_controller_threads(environment=None):
    env = os.environ if environment is None else environment
    controller_count = max(1, int(env.get("ALGOEVOLVE_CONTROLLER_NUM_THREADS", "2")))
    for name in NUMERICAL_THREADS:
        original = env.get(name)
        if original is not None:
            env.setdefault(PREFIX + name, original)
        try:
            controller = min(controller_count, max(1, int(original))) if original else controller_count
        except ValueError:
            controller = controller_count
        env[name] = str(controller)


def candidate_thread_environment(environment):
    result = dict(environment)
    for name in NUMERICAL_THREADS:
        saved = result.get(PREFIX + name)
        if saved is not None:
            result[name] = saved
    return result
