from typing import List, Optional
import os, joblib, numpy as np
from pathlib import Path

_default = Path(__file__).resolve().parent.parent / "models"
ARTIF_DIR = Path(os.environ.get("MODEL_DIR", _default))

# Variants defined for the clinical model type.
# Mirrors _CLINICAL_VARIANTS in src/config.py — 16 variants across 4 optional dimensions:
#   waist_circumference (wc), Lpa (lpa), CRP (crp), ApoB (apob).
# ApoB path variants exclude non_HDL from the model input (ApoB substitutes for the HDL+non_HDL pair).
_CLINICAL_VARIANTS = [
    # Non-ApoB path (cholesterol = HDL + non_HDL, where non_HDL = total_cholesterol − HDL)
    "wc_lpa_crp", "wc_lpa", "wc_crp", "wc",
    "lpa_crp",    "lpa",    "crp",    "base",
    # ApoB path (cholesterol = HDL + ApoB; non_HDL excluded)
    "wc_lpa_crp_apob", "wc_lpa_apob", "wc_crp_apob", "wc_apob",
    "lpa_crp_apob",    "lpa_apob",    "crp_apob",    "apob",
]

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

def select_variant(
    lpa: Optional[float],
    crp: Optional[float],
    waist_circumference: Optional[float],
    apob: Optional[float],
) -> str:
    """
    Route to one of 16 clinical model variants based on optional feature availability.

    ApoB path (apob is not None): model uses {HDL, ApoB}; non_HDL excluded.
    Non-ApoB path (apob is None): model uses {HDL, non_HDL} (derived as total_cholesterol − HDL).

    Within each path, waist_circumference, Lpa, and CRP determine the sub-variant.

    Args:
        lpa:                Patient's Lpa value, or None.
        crp:                Patient's CRP value, or None.
        waist_circumference: Patient's waist circumference, or None.
        apob:               Patient's ApoB value, or None.

    Returns:
        Variant name string (one of the 16 entries in _CLINICAL_VARIANTS).
    """
    has_wc   = waist_circumference is not None
    has_lpa  = lpa  is not None
    has_crp  = crp  is not None
    has_apob = apob is not None

    # Build suffix for non-ApoB sub-dimensions
    if has_wc and has_lpa and has_crp:
        base = "wc_lpa_crp"
    elif has_wc and has_lpa:
        base = "wc_lpa"
    elif has_wc and has_crp:
        base = "wc_crp"
    elif has_wc:
        base = "wc"
    elif has_lpa and has_crp:
        base = "lpa_crp"
    elif has_lpa:
        base = "lpa"
    elif has_crp:
        base = "crp"
    else:
        base = "base"

    if has_apob:
        # ApoB path: append "_apob" unless base is already bare "base"
        return f"{base}_apob" if base != "base" else "apob"
    return base


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
