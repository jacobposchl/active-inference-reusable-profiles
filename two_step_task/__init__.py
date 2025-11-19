"""two_step_task package: two-step task experiment utilities.

Structure:
- `data_loader` : data import & preprocessing helpers
- `generative`  : A/B matrix builders
- `models`      : M1/M2/M3 model classes
- `agent`       : TwoStepAgent implementation
- `fitting`     : fitting scaffolds (grid + optimization)
- `scripts`     : small command-line utilities (e.g., CSV inspector)

Keep this package lightweight: import submodules as needed.
"""

__all__ = [
    "data_loader",
    "generative",
    "models",
    "agent",
    "fitting",
]

__version__ = "0.1"
