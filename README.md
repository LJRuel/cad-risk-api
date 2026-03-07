# CAD Risk API

A FastAPI service exposing sex-specific Deep Survival Machines (DSM) models to compute cumulative CAD risk curves and educational intervention overlays (statin and antihypertensive).

> **Important**
> Intervention overlays apply fixed, explicit coefficients derived from RCT meta-analyses on top of the model's absolute baseline risk. They are intended for educational and population-level illustration, not individual causal inference or clinical prescription.

---

## Project structure

```
cad-risk-api/
  app/
    main.py        # FastAPI: /predict (DSM baseline + overlays), /health
    schemas.py     # Pydantic input schema
    deps.py        # Loads preprocessors & DSM models; builds age→80 time grid
    prepare.py     # Applies the serialized preprocessors (MICE / Scaler / OHE)
  models/
    scaling_parameters_clinical_men.joblib
    scaling_parameters_clinical_women.joblib
    PRED-CAD_DSM_clinical_model_men.joblib
    PRED-CAD_DSM_clinical_model_women.joblib
  requirements.txt
  Dockerfile
  README.md
  .gitignore
  LICENSE
```

---

## Quick start (local development)

1. Create and activate a virtual environment:
```bash
python -m venv .venv && source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place model artifacts in `/models`:
   - `scaling_parameters_clinical_men.joblib`
   - `scaling_parameters_clinical_women.joblib`
   - `PRED-CAD_DSM_clinical_model_men.joblib`
   - `PRED-CAD_DSM_clinical_model_women.joblib`

4. Run the API:
```bash
uvicorn app.main:app --reload --port 8080
```

5. Verify:
```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

---

## API endpoints

### `GET /health`

Basic health probe.

```json
{"status": "ok"}
```

---

### `POST /predict`

Computes the baseline cumulative CAD risk from the current age (minimum 40) to age 80, and returns three educational intervention overlays.

**Request body**
```json
{
  "sex": 1,
  "age_recruitment": 55,
  "systolic_BP": 142,
  "HDL": 1.2,
  "LDL": 3.4,
  "diabetes": 0,
  "family_history": 1,
  "antiht": 0,
  "statin": 0,
  "smoking_current": 0,
  "pack_years": 5,
  "waist_circumference": 100,
  "Lpa": 60,
  "CRP": 2.0,
  "GestHtPreEcl": null,
  "menopause": null,
  "HRT": null,
  "interventions": {
    "delta_ldl_mmol": 0.0,
    "delta_sbp_mmHg": 0
  }
}
```

`sex`: 1 = male, 0 = female. Optional fields can be `null` if unknown. `pack_years` defaults to `0.0` if omitted.

**Response**
```json
{
  "ages": [55, 56, "...", 80],
  "risk_baseline": ["..."],
  "risk_ldl": ["..."],
  "risk_sbp": ["..."],
  "risk_combined": ["..."],
  "relative_risks": {
    "ldl_rr": 0.66,
    "bp_rr": 0.81,
    "combined_rr": 0.5346
  },
  "assumptions": {
    "ldl_drop_fraction": 0.5,
    "rr_per_mmol_ldl": 0.2,
    "bp_target_threshold": 129.0,
    "rr_per_5mmhg": 0.08
  }
}
```

---

## Intervention overlays

### Statin / LDL

Assumes a 50% LDL reduction under statin therapy and a 20% risk reduction per 1 mmol/L LDL lowered.

```
ldl_rr = 1 - (LDL × 0.5 × 0.2)    clipped to [0, 1]
risk_ldl(age) = risk_baseline(age) × ldl_rr
```

### Antihypertensive / SBP

Applies an 8% risk reduction per 5 mmHg lowered toward a threshold of 129 mmHg, only when SBP ≥ 130.

```
bp_rr = 1 - ((SBP - 129) / 5) × 0.08    if SBP ≥ 130
bp_rr = 1                                 if SBP < 130

bp_rr clipped to [0, 1]
risk_sbp(age) = risk_baseline(age) × bp_rr
```

### Combined

```
combined_rr = ldl_rr × bp_rr
risk_combined(age) = risk_baseline(age) × combined_rr
```

---

## Docker

```bash
docker build -t dsm-cad-api:latest .
docker run --rm -p 8080:8080 dsm-cad-api:latest
```

Exposes `GET /health` and `POST /predict` on port 8080.

---

## Configuration

- **Model artifacts**: loaded at startup from `/models`. Do not commit them to version control; distribute via secure methods (e.g., encrypted S3 with CI/CD download).
- **Overlay parameters**: `ldl_drop_fraction`, `rr_per_mmol_ldl`, `bp_target_threshold`, and `rr_per_5mmhg` are constants in `app/main.py`.
- **Statelessness**: inputs are processed in memory and not persisted.

---

## References

1. Mean LDL reduction of 50% under statin therapy — Canadian Journal of Cardiology, 2024, Figure 4. https://onlinecjc.ca/article/S0828-282X(24)00358-1/fulltext
2. 20% risk reduction per 1 mmol/L LDL reduction under statin — Lancet, 2012. https://www.sciencedirect.com/science/article/pii/S0140673612603675
3. Adverse effects of antihypertensive treatment below 120 mmHg SBP — NEJM, 2015. https://www.nejm.org/doi/full/10.1056/NEJMoa1511939
4. 8% CAD risk reduction per 5 mmHg SBP reduction — Lancet, 2021. https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)00590-0/fulltext

---

## License

MIT License. See `LICENSE` for details.
