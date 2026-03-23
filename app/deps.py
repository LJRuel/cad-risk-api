from typing import List, Optional
import os, joblib, numpy as np
from pathlib import Path

_default = Path(__file__).resolve().parent.parent / "models"
ARTIF_DIR = Path(os.environ.get("MODEL_DIR", _default))

# Preprocessors
MEN_PREP = joblib.load(ARTIF_DIR / "scaling_parameters_clinical_men.joblib")
WOMEN_PREP = joblib.load(ARTIF_DIR / "scaling_parameters_clinical_women.joblib")

# DSM models
MODEL_MEN = joblib.load(ARTIF_DIR / "PRED-CAD_DSM_clinical_model_men.joblib")
MODEL_WOMEN = joblib.load(ARTIF_DIR / "PRED-CAD_DSM_clinical_model_women.joblib")

# Combined (sex-agnostic) model — loaded only if artefacts exist.
# Train with: make train MODEL=clinical_combined
def _try_load(path: Path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

COMBINED_PREP  = _try_load(ARTIF_DIR / "scaling_parameters_clinical_combined_combined.joblib")
MODEL_COMBINED = _try_load(ARTIF_DIR / "PRED-CAD_DSM_clinical_combined_model_combined.joblib")

def times_from_age_to_80(age_current: float) -> List[int]:
    """
    Annual grid (days) from current age (>=40) to 80 years.
    If age_current < 40, we start at 40 (aligned with training data).
    """
    start_age = max(40.0, age_current)
    horizon_years = max(0, int(np.floor(80 - start_age)))
    return [y * 365 for y in range(0, horizon_years + 1)]