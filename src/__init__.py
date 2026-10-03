# macOS: torch (pulled in by tabpfn / tabpfn-client) bundles its own libomp. If torch
# loads first, a later XGBoost fit segfaults. Loading xgboost first avoids it.
import xgboost  # noqa: F401
