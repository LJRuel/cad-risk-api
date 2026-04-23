from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .schemas import InputPayload
from .deps import (
    CLINICAL_MODELS, CLINICAL_PREPS,
    COMBINED_MODELS, COMBINED_PREPS,
    select_variant, times_from_age_to_80,
)
from .prepare import apply_preprocessor_one, apply_preprocessor_combined

app = FastAPI(title="CAD Risk API", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def ui():
    return FileResponse("app/static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: InputPayload):
    raw = payload.model_dump()
    sex = raw.pop("sex")
    raw.pop("smoking_current", None)  # UI-only field, not a model feature

    # Determine which variant to use based on available optional features
    variant = select_variant(raw.get("Lpa"), raw.get("CRP"))

    # Temporal grid: entire life from max(40 years, current age) to 80 years (end of training data)
    times = times_from_age_to_80(raw["age_recruitment"])
    if len(times) == 0:
        raise HTTPException(400, "Age >= 80 years: no projection horizon available.")

    if sex is None:
        model = COMBINED_MODELS.get(variant)
        if model is None:
            raise HTTPException(
                503,
                f"Combined-sex model for variant '{variant}' not available. "
                f"Train it with: make train MODEL=clinical_sex_combined VARIANT={variant}"
            )
        X = apply_preprocessor_combined(variant, raw)
    else:
        model = CLINICAL_MODELS.get((variant, sex))
        if model is None:
            sex_label = "men" if sex == 1 else "women"
            raise HTTPException(
                503,
                f"Clinical model for variant '{variant}', sex='{sex_label}' not available. "
                f"Train it with: make train MODEL=clinical VARIANT={variant}"
            )
        X = apply_preprocessor_one(variant, sex, raw)

    # Baseline risk for 1 individual => index 0
    risk_baseline = model.predict_risk(X, times)[0].tolist()
    ages = [int(max(40, raw["age_recruitment"]) + t/365) for t in times]

    # ====== Interventions formulas ======
    # LDL: risk * (1 - (LDL * 0.5 (statin LDL reduction) * 0.2 (relative risk reduction per mmol/L LDL-C drop)))
    LDL_val = float(raw.get("LDL", 0.0) or 0.0)
    ldl_relative_risk = 1.0 - (LDL_val * 0.5 * 0.2)
    ldl_relative_risk = max(0.0, min(1.0, ldl_relative_risk))

    # SBP: if SBP >= 130 => risk * (1 - ((SBP - 129)/5) * 0.08 (relative risk reduction per 5 mmHg)), else unchanged
    SBP_val = float(raw.get("systolic_BP", 0.0) or 0.0)
    if SBP_val >= 130.0:
        bp_relative_risk = 1.0 - (((SBP_val - 129.0)/5.0) * 0.08)
    else:
        bp_relative_risk = 1.0
    bp_relative_risk = max(0.0, min(1.0, bp_relative_risk))

    # Multiplicative combinaison (independant effects)
    combined_rr = ldl_relative_risk * bp_relative_risk

    # PCSK9 inhibitor: 27% RR reduction on top of statin (VESALIUS-CV trial)
    pcsk9_rr = 0.73

    risk_ldl    = [max(0.0, min(1.0, p * ldl_relative_risk))              for p in risk_baseline]
    risk_bp     = [max(0.0, min(1.0, p * bp_relative_risk))               for p in risk_baseline]
    risk_combined = [max(0.0, min(1.0, p * combined_rr))                  for p in risk_baseline]
    risk_pcsk9  = [max(0.0, min(1.0, p * ldl_relative_risk * pcsk9_rr))   for p in risk_baseline]

    return {
        "ages": ages,
        "risk_baseline": risk_baseline,    # Baseline risk without intervention
        "risk_ldl": risk_ldl,              # Baseline risk × ldl_rr (statin)
        "risk_sbp": risk_bp,               # Baseline risk × bp_rr (antihypertensive, if SBP≥130)
        "risk_combined": risk_combined,    # Baseline risk × ldl_rr × bp_rr
        "risk_pcsk9": risk_pcsk9,          # Baseline risk × ldl_rr × 0.73 (statin + PCSK9 inhibitor)
        "relative_risks": {
            "ldl_rr": ldl_relative_risk,
            "bp_rr": bp_relative_risk,
            "combined_rr": combined_rr,
            "pcsk9_rr": pcsk9_rr
        },
        "assumptions": {
            "ldl_drop_fraction": 0.5,
            "rr_per_mmol_ldl": 0.2,
            "bp_target_threshold": 129.0,
            "rr_per_5mmhg": 0.08,
            "pcsk9_rr": pcsk9_rr
        }
    }
