# AlgoEvolve Generated Solution Interface v1

AlgoEvolve exports one `solution_manifest.json` beside each best/Top-K solution. The
manifest uses `interface_version: algoevolve.solution.v1` and records the task kind,
statefulness, artifact path, method family, and callable entrypoints.

## Prediction

```python
def train(data, artifact_dir): ...
def predict(model_path, data): ...
```

`train` saves the fitted model and preprocessing state. `predict` loads them and
performs inference without retraining.

## Decision Solver

```python
def solve(model_path, data): ...
```

Stateless heuristics, mathematical optimization, and search methods may accept
`model_path=None`. The returned solution must be replayed through the task's
deterministic validator and scorer.

## Reinforcement Learning Or Hybrid Decision

```python
def train_policy(data, artifact_dir): ...
def rollout(model_path, data): ...
```

`rollout` must load the saved policy and return the evaluated decision artifact
without retraining. Static optimization tasks may use this interface when their
instances define a credible state/action/transition/reward process.

Generated code is checked immediately after generation/review. Syntax errors,
missing required functions, `eval`, `exec`, `os.system`, and subprocess calls with
`shell=True` are rejected before execution.
