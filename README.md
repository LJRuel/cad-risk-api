<img src="docs/banner.svg" alt="CAD Risk Estimator API" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Use-Research%20only-f97316" alt="Research only"/>
</p>

A stateless REST API serving sex-stratified **Deep Survival Machines (DSM)** models to estimate cumulative coronary artery disease (CAD) risk from the current age to 80, alongside educational overlays for two common interventions: statin therapy and antihypertensive treatment. A bilingual (EN/FR) single-page web application is served at `/`.

> **Important**
> Intervention overlays apply fixed coefficients derived from RCT meta-analyses on top of the model's predicted absolute baseline risk. They are intended for educational and population-level illustration — not individual causal inference or clinical prescription.

---

## Table of contents

- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Variant model selection](#variant-model-selection)
- [API endpoints](#api-endpoints)
  - [`GET /health`](#get-health)
  - [`POST /predict`](#post-predict)
- [Intervention overlays](#intervention-overlays)
- [Docker](#docker)
- [Configuration](#configuration)
- [References](#references)
- [License](#license)

---

## Project structure

```
cad-risk-api/
├── app/
│   ├── main.py        # FastAPI app: /predict endpoint, intervention formulas, variant routing
│   ├── schemas.py     # Pydantic InputPayload — field definitions and pack_years validation
│   ├── deps.py        # Model + preprocessor loading, select_variant(), times_from_age_to_80()
│   ├── prepare.py     # Per-variant preprocessing: MICE imputation → scaling → OHE
│   └── static/
│       ├── index.html  # Self-contained bilingual (EN/FR) single-page application
│       ├── waist_en.svg
│       └── waist_fr.svg
├── models/            # Serialized artifacts — not included, see note below
│   ├── PRED-CAD_DSM_clinical_full_model_men.joblib
│   ├── PRED-CAD_DSM_clinical_full_model_women.joblib
│   ├── PRED-CAD_DSM_clinical_lpa_model_men.joblib
│   ├── PRED-CAD_DSM_clinical_lpa_model_women.joblib
│   ├── PRED-CAD_DSM_clinical_crp_model_men.joblib
│   ├── PRED-CAD_DSM_clinical_crp_model_women.joblib
│   ├── PRED-CAD_DSM_clinical_base_model_men.joblib
│   ├── PRED-CAD_DSM_clinical_base_model_women.joblib
│   ├── scaling_parameters_clinical_full_men.joblib
│   ├── scaling_parameters_clinical_full_women.joblib
│   └── ... (same pattern for lpa / crp / base variants)
├── requirements.txt
├── Dockerfile
└── LICENSE
```

---

## Quick start

**Prerequisites:** Python 3.9 · `libgomp1` (`apt-get install libgomp1` on Debian/Ubuntu)

> **Model artifacts**
> The `.joblib` files in `models/` are not included in this repository — they are part of a private research project. The application will not start without them. Artifacts are produced by the [PRISME-Pred-CVD](https://github.com/ljruel/PRISME-Pred-CVD) training pipeline (`make train MODEL=clinical`).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model artifacts in models/

# 4. Start the server
uvicorn app.main:app --reload --port 8000

# 5. Verify
curl -s http://localhost:8000/health
# {"status":"ok"}
```

Open `http://localhost:8000` for the web interface.

---

## Variant model selection

The clinical model has two optional features — **Lp(a)** and **CRP** — that may be unavailable in a given clinical setting. Rather than imputing missing values, the API maintains four model variants corresponding to all combinations of these features' presence:

| Variant | Lp(a) provided | CRP provided |
|---------|----------------|--------------|
| `full`  | yes            | yes          |
| `lpa`   | yes            | no           |
| `crp`   | no             | yes          |
| `base`  | no             | no           |

The API selects the variant automatically: if `Lpa` is `null` in the request the model that was never trained with that feature is used — no imputation or zeroing occurs. This guarantees that risk estimates for Lp(a) are monotonically increasing without interaction artifacts from absent features.

---

## API endpoints

### `GET /health`

Basic liveness probe.

```json
{ "status": "ok" }
```

---

### `POST /predict`

Computes the baseline cumulative CAD risk curve from the current age (minimum 40) to age 80 and returns three educational intervention overlays.

#### Request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `sex` | `int \| null` | No | `1` = male · `0` = female · `null` = sex not reported (uses combined-sex model) |
| `age_recruitment` | `float` | Yes | Current age; projection starts at max(40, age) |
| `systolic_BP` | `float` | Yes | Systolic blood pressure (mmHg) |
| `HDL` | `float` | Yes | HDL cholesterol (mmol/L) |
| `LDL` | `float` | Yes | LDL cholesterol (mmol/L) |
| `waist_circumference` | `float` | Yes | Waist circumference (cm) |
| `diabetes` | `int` | Yes | `1` = yes · `0` = no |
| `family_history` | `int` | Yes | First-degree relative with premature CAD: `1` = yes · `0` = no |
| `antiht` | `int` | Yes | Currently on antihypertensive therapy: `1` = yes · `0` = no |
| `statin` | `int` | Yes | Currently on statin therapy: `1` = yes · `0` = no |
| `smoking_current` | `int` | No | `0` = never · `1` = current · `2` = former. UI-only — not a model feature. Controls pack-years validation. |
| `pack_years` | `float` | Conditional | Required and > 0 when `smoking_current` ∈ {1, 2}. Set to `0.0` automatically for never-smokers. |
| `Lpa` | `float` | No | Lipoprotein(a) (nmol/L). If `null`, routes to a variant without Lp(a). |
| `CRP` | `float` | No | C-reactive protein (mg/L). If `null`, routes to a variant without CRP. |
| `GestHtPreEcl` | `int` | No | History of gestational hypertension or pre-eclampsia (females only): `1` = yes · `0` = no |
| `menopause` | `int` | No | Menopausal status (females only): `1` = yes · `0` = no |
| `HRT` | `int` | No | Currently on hormone replacement therapy (females only): `1` = yes · `0` = no |

#### Example request — male, all features provided

```json
{
  "sex": 1,
  "age_recruitment": 55,
  "systolic_BP": 142,
  "HDL": 1.2,
  "LDL": 3.4,
  "waist_circumference": 95,
  "diabetes": 0,
  "family_history": 1,
  "antiht": 0,
  "statin": 0,
  "smoking_current": 2,
  "pack_years": 12,
  "Lpa": 60,
  "CRP": 2.0
}
```

#### Example request — female, no Lp(a) or CRP (routes to `base` variant)

```json
{
  "sex": 0,
  "age_recruitment": 62,
  "systolic_BP": 135,
  "HDL": 1.6,
  "LDL": 2.9,
  "waist_circumference": 84,
  "diabetes": 0,
  "family_history": 0,
  "antiht": 1,
  "statin": 0,
  "smoking_current": 0,
  "GestHtPreEcl": 0,
  "menopause": 1,
  "HRT": 0
}
```

#### Response

```json
{
  "ages":          [55, 56, "...", 80],
  "risk_baseline": [0.002, 0.004, "...", 0.18],
  "risk_ldl":      [0.001, 0.003, "...", 0.12],
  "risk_sbp":      [0.002, 0.003, "...", 0.15],
  "risk_combined": [0.001, 0.002, "...", 0.10],
  "relative_risks": {
    "ldl_rr":      0.66,
    "bp_rr":       0.81,
    "combined_rr": 0.53
  },
  "assumptions": {
    "ldl_drop_fraction":   0.5,
    "rr_per_mmol_ldl":     0.2,
    "bp_target_threshold": 129.0,
    "rr_per_5mmhg":        0.08
  }
}
```

All risk arrays contain one value per year from `max(40, age_recruitment)` to `80`, expressed as a probability in `[0, 1]`.

---

## Intervention overlays

Overlays are multiplicative reductions applied to the baseline risk curve. They are derived from published RCT meta-analyses and fixed at the values shown.

### Statin / LDL

Assumes a 50% LDL reduction under statin therapy and a 20% relative risk reduction per 1 mmol/L LDL lowered.

```
ldl_rr         = 1 − (LDL × 0.5 × 0.2)       clipped to [0, 1]
risk_ldl(age)  = risk_baseline(age) × ldl_rr
```

### Antihypertensive / SBP

Applies an 8% relative risk reduction per 5 mmHg reduction toward a target of 129 mmHg, only when SBP ≥ 130.

```
bp_rr          = 1 − ((SBP − 129) / 5) × 0.08    if SBP ≥ 130
bp_rr          = 1.0                               if SBP < 130

bp_rr clipped to [0, 1]
risk_sbp(age)  = risk_baseline(age) × bp_rr
```

### Combined (multiplicative)

```
combined_rr         = ldl_rr × bp_rr
risk_combined(age)  = risk_baseline(age) × combined_rr
```

---

## Docker

```bash
docker build -t cad-risk-api:latest .
docker run --rm -p 8000:8000 \
  -v /path/to/models:/app/models \
  cad-risk-api:latest
```

The image uses `python:3.9.7-slim` and installs `libgomp1`. Model artifacts must be mounted at `/app/models` (or set `MODEL_DIR` env var to override).

---

## Configuration

| Parameter | How to set | Description |
|---|---|---|
| `MODEL_DIR` | Environment variable | Path to the directory containing `.joblib` artifacts. Default: `./models` |
| Intervention coefficients | `app/main.py` constants | `ldl_drop_fraction`, `rr_per_mmol_ldl`, `bp_target_threshold`, `rr_per_5mmhg` |

Inputs are processed in memory and not persisted.

---

## References

1. Mean LDL reduction of ~50% under statin therapy — *Canadian Journal of Cardiology*, 2024, Figure 4.
   https://onlinecjc.ca/article/S0828-282X(24)00358-1/fulltext

2. 20% relative risk reduction per 1 mmol/L LDL lowered — *The Lancet*, 2012.
   https://www.sciencedirect.com/science/article/pii/S0140673612603675

3. Adverse effects of antihypertensive therapy below 120 mmHg SBP — *NEJM*, 2015.
   https://www.nejm.org/doi/full/10.1056/NEJMoa1511939

4. 8% CAD risk reduction per 5 mmHg SBP reduction — *The Lancet*, 2021.
   https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)00590-0/fulltext

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
