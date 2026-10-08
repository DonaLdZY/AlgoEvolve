from types import SimpleNamespace

from agents.prompt_policy import infer_method_family, method_family_for_node


def test_task_prose_and_negative_rl_mentions_are_not_algorithm_evidence():
    code = '''from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
description = "This is prediction, not reinforcement learning or MCTS."
model = LGBMRegressor()
'''
    assert infer_method_family("MCTS search", code=code) == "tree_boosting"
    node = SimpleNamespace(code=code, method_family="reinforcement_learning")
    assert method_family_for_node(node, task_family="prediction") == "tree_boosting"


def test_real_ensemble_and_rl_symbols_remain_distinguishable():
    ensemble = '''from lightgbm import LGBMRegressor
class _WeightedSeasonalEnsemble:
    pass
model = _WeightedSeasonalEnsemble()
'''
    assert infer_method_family("", code=ensemble) == "ensemble"
    assert infer_method_family("", code="from stable_baselines3 import PPO\nmodel = PPO('MlpPolicy', env)", task_family="decision") == "reinforcement_learning"
