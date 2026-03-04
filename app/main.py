from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .schemas import InputPayload
from .deps import MODEL_MEN, MODEL_WOMEN, times_from_age_to_80
from .prepare import apply_preprocessor_one

app = FastAPI(title="CAD Risk API", version="0.1.0")

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

    # Temporal grid: entire life from max(40 years, current age) to 80 years (end of training data)
    times = times_from_age_to_80(raw["age_recruitment"])
    if len(times) == 0:
        raise HTTPException(400, "Age ≥ 80 years: no projection horizon available.")

    X = apply_preprocessor_one(sex, raw)
    model = MODEL_MEN if sex==1 else MODEL_WOMEN

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

    risk_ldl = [max(0.0, min(1.0, p * ldl_relative_risk)) for p in risk_baseline]
    risk_bp  = [max(0.0, min(1.0, p * bp_relative_risk))  for p in risk_baseline]
    risk_combined = [max(0.0, min(1.0, p * combined_rr)) for p in risk_baseline]

    return {
        "ages": ages,                      # Input age
        "risk_baseline": risk_baseline,    # Baseline risk without intervention
        "risk_ldl": risk_ldl,              # Baseline risk × (1 - LDL*0.5*0.2)
        "risk_sbp": risk_bp,               # Baseline risk × (1 - ((SBP-129)/5)*0.08) if SBP≥130
        "risk_combined": risk_combined,    # Baseline risk × (ldl_rr * bp_rr) combined relative risk
        "relative_risks": {
            "ldl_rr": ldl_relative_risk,
            "bp_rr": bp_relative_risk,
            "combined_rr": combined_rr
        },
        "assumptions": {
            "ldl_drop_fraction": 0.5,
            "rr_per_mmol_ldl": 0.2,
            "bp_target_threshold": 129.0,
            "rr_per_5mmhg": 0.08
        }
    }