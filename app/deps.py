from typing import List, Optional
import os, joblib, numpy as np
from pathlib import Path

_default = Path(__file__).resolve().parent.parent / "models"
ARTIF_DIR = Path(os.environ.get("MODEL_DIR", _default))

# Variants defined for the clinical model type.
# Mirrors FEATURES["clinical"]["variants"] in src/config.py.
_CLINICAL_VARIANTS = ["full", "lpa", "crp", "base"]

# ---------------------------------------------------------------------------
# Graceful loader: returns None if the file does not exist
# ---------------------------------------------------------------------------
def _try_load(path: Path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Clinical models and preprocessors: (variant, sex) → artefact
# sex is int: 1=men, 0=women; sex_label is str: "men"/"women"
# ---------------------------------------------------------------------------
def _sex_label(sex: int) -> str:
    return "men" if sex == 1 else "women"


CLINICAL_MODELS: dict = {}
CLINICAL_PREPS: dict  = {}

for _variant in _CLINICAL_VARIANTS:
    for _sex in (1, 0):
        _label = _sex_label(_sex)
        CLINICAL_MODELS[(_variant, _sex)] = _try_load(
            ARTIF_DIR / f"PRED-CAD_DSM_clinical_{_variant}_model_{_label}.joblib"
        )
        CLINICAL_PREPS[(_variant, _sex)] = _try_load(
            ARTIF_DIR / f"scaling_parameters_clinical_{_variant}_{_label}.joblib"
        )

# ---------------------------------------------------------------------------
# Combined (sex-agnostic) clinical models and preprocessors: variant → artefact
# ---------------------------------------------------------------------------
COMBINED_MODELS: dict = {}
COMBINED_PREPS: dict  = {}

for _variant in _CLINICAL_VARIANTS:
    COMBINED_MODELS[_variant] = _try_load(
        ARTIF_DIR / f"PRED-CAD_DSM_clinical_sex_combined_{_variant}_model_combined.joblib"
    )
    COMBINED_PREPS[_variant] = _try_load(
        ARTIF_DIR / f"scaling_parameters_clinical_sex_combined_{_variant}_combined.joblib"
    )


# ---------------------------------------------------------------------------
# Variant selection helper
# ---------------------------------------------------------------------------

def select_variant(lpa: Optional[float], crp: Optional[float]) -> str:
    """
    Determine which clinical model variant to use based on optional feature availability.

    Variant mapping:
        "full"  — both Lpa and CRP are present
        "lpa"   — only Lpa is present (CRP absent)
        "crp"   — only CRP is present (Lpa absent)
        "base"  — neither Lpa nor CRP is present

    Args:
        lpa: Patient's Lpa value, or None if not available.
        crp: Patient's CRP value, or None if not available.

    Returns:
        Variant name string.
    """
    has_lpa = lpa is not None
    has_crp = crp is not None

    if has_lpa and has_crp:
        return "full"
    elif has_lpa:
        return "lpa"
    elif has_crp:
        return "crp"
    else:
        return "base"


# ---------------------------------------------------------------------------
# Time grid helper (unchanged)
# ---------------------------------------------------------------------------

def times_from_age_to_80(age_current: float) -> List[int]:
    """
    Annual grid (days) from current age (>=40) to 80 years.
    If age_current < 40, we start at 40 (aligned with training data).
    """
    start_age = max(40.0, age_current)
    horizon_years = max(0, int(np.floor(80 - start_age)))
    return [y * 365 for y in range(0, horizon_years + 1)]
