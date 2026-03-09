# CAD Risk Estimator App

<p>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Use-Research%20only-f97316" alt="Research only"/>
</p>

A stateless REST API serving sex-specific **Deep Survival Machines (DSM)** models to estimate cumulative coronary artery disease (CAD) risk from the current age to 80, alongside educational overlays for two common interventions: statin therapy and antihypertensive treatment.

> **Important**
> Intervention overlays apply fixed coefficients derived from RCT meta-analyses on top of the model's predicted absolute baseline risk. They are intended for educational and population-level illustration — not individual causal inference or clinical prescription.

---

## Table of contents

- [CAD Risk Estimator App](#cad-risk-estimator-app)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Quick start (local development)](#quick-start-local-development)
  - [API endpoints](#api-endpoints)
    - [`GET /health`](#get-health)
    - [`POST /predict`](#post-predict)
      - [Request fields](#request-fields)
      - [Example request](#example-request)
      - [Response](#response)
  - [Intervention overlays](#intervention-overlays)
    - [Statin / LDL](#statin--ldl)
    - [Antihypertensive / SBP](#antihypertensive--sbp)
    - [Combined (multiplicative)](#combined-multiplicative)
  - [Docker](#docker)
  - [Configuration](#configuration)
  - [References](#references)
  - [License](#license)

---

## Project structure

```
cad-risk-api/
├── app/
│   ├── main.py        # FastAPI application: /predict and /health
│   ├── schemas.py     # Pydantic input/output schemas
│   ├── deps.py        # Model loading and age-to-80 time grid
│   └── prepare.py     # Preprocessing pipeline (imputation, scaling, OHE)
├── models/            # Serialized artifacts (not included — see note below)
│   ├── scaling_parameters_clinical_men.joblib
│   ├── scaling_parameters_clinical_women.joblib
│   ├── PRED-CAD_DSM_clinical_model_men.joblib
│   └── PRED-CAD_DSM_clinical_model_women.joblib
├── requirements.txt
├── Dockerfile
└── LICENSE
```

---

## Quick start (local development)

**Prerequisites:** Python 3.9 · `libgomp1` (`apt-get install libgomp1` on Debian/Ubuntu; available via Homebrew on macOS)

> **Model artifacts**
> The `.joblib` files in `/models` are not included in this repository — they are part of a private research project and the application will not start without them. Contact the maintainers to request access.

**1. Create and activate a virtual environment**
```bash
python -m venv .venv && source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place model artifacts in `/models`**
```
models/
├── scaling_parameters_clinical_men.joblib
├── scaling_parameters_clinical_women.joblib
├── PRED-CAD_DSM_clinical_model_men.joblib
└── PRED-CAD_DSM_clinical_model_women.joblib
```

**4. Start the server**
```bash
uvicorn app.main:app --reload --port 8080
```

**5. Verify**
```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

---

## API endpoints

### `GET /health`

Basic liveness probe.

```json
{ "status": "ok" }
```

---

### `POST /predict`

Computes the baseline cumulative CAD risk curve from the current age (minimum 40) to age 80, and returns three educational intervention overlays.

#### Request fields

| Field | Type | Required | Description |
|---|---|---|---|
| `sex` | `int` | Yes | `1` = male, `0` = female |
| `age_recruitment` | `float` | Yes | Current age; projection starts at max(40, age) |
| `systolic_BP` | `float` | Yes | Systolic blood pressure (mmHg) |
| `HDL` | `float` | Yes | HDL cholesterol (mmol/L) |
| `LDL` | `float` | Yes | LDL cholesterol (mmol/L) |
| `diabetes` | `int` | Yes | `1` = yes, `0` = no |
| `family_history` | `int` | Yes | `1` = yes, `0` = no |
| `antiht` | `int` | Yes | Currently on antihypertensive: `1` = yes, `0` = no |
| `statin` | `int` | Yes | Currently on statin: `1` = yes, `0` = no |
| `smoking_current` | `int` | No | `1` = yes, `0` = no |
| `pack_years` | `float` | No | Pack-years of smoking (default `0.0`) |
| `waist_circumference` | `float` | No | Waist circumference (cm) |
| `Lpa` | `float` | No | Lipoprotein(a) (nmol/L) |
| `CRP` | `float` | No | C-reactive protein (mg/L) |
| `GestHtPreEcl` | `int` | No | History of gestational hypertension or pre-eclampsia (females only) |
| `menopause` | `int` | No | Menopausal status (females only) |
| `HRT` | `int` | No | Currently on hormone replacement therapy (females only) |
| `interventions.delta_ldl_mmol` | `float` | No | Reserved (default `0.0`) |
| `interventions.delta_sbp_mmHg` | `int` | No | Reserved (default `0`) |

Optional fields default to `null` unless stated otherwise.

#### Example request

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
  "CRP": 2.0
}
```

#### Response

```json
{
  "ages":          [55, 56, "...", 80],
  "risk_baseline": ["..."],
  "risk_ldl":      ["..."],
  "risk_sbp":      ["..."],
  "risk_combined": ["..."],
  "relative_risks": {
    "ldl_rr":      0.66,
    "bp_rr":       0.81,
    "combined_rr": 0.5346
  },
  "assumptions": {
    "ldl_drop_fraction":    0.5,
    "rr_per_mmol_ldl":      0.2,
    "bp_target_threshold":  129.0,
    "rr_per_5mmhg":         0.08
  }
}
```

All risk arrays contain one value per year from `max(40, age_recruitment)` to `80`, expressed as a probability in `[0, 1]`.

---

## Intervention overlays

Overlays are multiplicative reductions applied to the baseline risk curve. They are derived from published RCT meta-analyses and are fixed at the values shown below.

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
bp_rr          = 1                                 if SBP < 130

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
docker build -t dsm-cad-api:latest .
docker run --rm -p 8080:8080 dsm-cad-api:latest
```

The image uses `python:3.9.7-slim`, installs `libgomp1`, and serves `GET /health` and `POST /predict` on port **8080**.

---

## Configuration

| Parameter | Location | Description |
|---|---|---|
| Model artifacts | `/models/*.joblib` | Loaded at startup; keep outside version control |
| `ldl_drop_fraction` | `app/main.py` | Assumed LDL reduction under statin (default `0.5`) |
| `rr_per_mmol_ldl` | `app/main.py` | Relative risk reduction per mmol/L LDL (default `0.2`) |
| `bp_target_threshold` | `app/main.py` | SBP threshold for antihypertensive overlay (default `129.0`) |
| `rr_per_5mmhg` | `app/main.py` | Relative risk reduction per 5 mmHg SBP (default `0.08`) |

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
