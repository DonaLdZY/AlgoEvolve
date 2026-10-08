"""Environment/package prompt."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=1)
def _installed_packages():
    candidates = (
        "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "xgboost",
        "lightgbm", "catboost", "joblib", "ortools", "openpyxl", "xlrd",
        "torch", "torchvision", "torch-geometric", "bayesian-optimization", "timm",
        "transformers", "sentence-transformers", "opencv-python", "Pillow",
    )
    installed = []
    for name in sorted(candidates):
        try:
            installed.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    return tuple(installed)


def get_prompt_environment():
    """Stable metadata from this interpreter; package names never imply GPU access."""
    return {
        "Installed Packages": (
            "Detected relevant distributions in the search interpreter: " + ", ".join(_installed_packages()) + ". "
            "This is metadata, not an import or hardware compatibility test. Check the selected dependency before use. "
            "Follow the task's CPU/GPU, memory, network and installation limits; an installed GPU-capable package does not grant GPU access. "
            "Other libraries are not guaranteed available. For neural networks prefer PyTorch when compatible."
        )
    }
