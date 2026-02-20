# CAD Risk Education API (DSM)

**FastAPI** service that exposes a **Deep Survival Machines (DSM)** model (sex-specific) to:

- compute a **cumulative CAD risk curve** from the current age (min **40**) to **80**;
- display the **educational impact** of two interventions (**statin** and **antihypertensive**) using **explicit formulas**;
- run **statelessly** (no data stored) and deploy easily on **AWS App Runner**.

> **Important**  
> The API applies **fixed, explicit overlays** for interventions (statin & SBP) on top of the **absolute baseline** risk predicted by the model. These overlays are intended for **education and population‑level intuition** (not individual causal effects).
>
> See context references below.

---

## Table of contents

- [CAD Risk Education API (DSM)](#cad-risk-education-api-dsm)
  - [Table of contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Quick start (local dev)](#quick-start-local-dev)
  - [API endpoints](#api-endpoints)
    - [`GET /health`](#get-health)
      - [Response](#response)
    - [`POST /predict`](#post-predict)
      - [Request body (JSON)](#request-body-json)
      - [Response (JSON)](#response-json)
  - [Intervention overlays (exact formulas)](#intervention-overlays-exact-formulas)
    - [1) Statin / LDL overlay](#1-statin--ldl-overlay)
    - [2) Antihypertensive / SBP overlay](#2-antihypertensive--sbp-overlay)
    - [3) Combined overlay (independent effects, multiplicative)](#3-combined-overlay-independent-effects-multiplicative)
  - [Docker](#docker)
  - [Deploy to AWS App Runner (ECR → App Runner)](#deploy-to-aws-app-runner-ecr--app-runner)
  - [Configuration \& notes](#configuration--notes)
  - [Security \& privacy](#security--privacy)
  - [FAQ](#faq)
  - [References](#references)
  - [License](#license)

---

## Project structure

```
cad-risk-education-api-dsm/
  app/
    main.py        # FastAPI: /predict (DSM baseline + overlays), /health
    schemas.py     # Pydantic input schema
    deps.py        # Loads preprocessors & DSM models; builds age→80 time grid
    prepare.py     # Applies the serialized preprocessors (MICE / Scaler / OHE)
  models/
    #  - scaling_parameters_clinical_men.joblib
    #  - scaling_parameters_clinical_women.joblib
    #  - PRED-CAD_DSM_clinical_model_men.joblib
    #  - PRED-CAD_DSM_clinical_model_women.joblib
  requirements.txt
  Dockerfile
  README.md
  .gitignore
  LICENSE
```

---

## Quick start (local dev)

1) **Create and activate a virtual environment**
```bash
python -m venv .venv && source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Add model artifacts to /models**  
- `scaling_parameters_clinical_men.joblib`  
- `scaling_parameters_clinical_women.joblib`  
- `PRED-CAD_DSM_clinical_model_men.joblib`  
- `PRED-CAD_DSM_clinical_model_women.joblib`

4) **Run the API**
```bash
uvicorn app.main:app --reload --port 8080
```

5) **Health check**
```bash
curl -s http://localhost:8080/health
# {"status":"ok"}
```

---

## API endpoints

### `GET /health`
Basic health probe.

#### Response
```json
{"status":"ok"}
```

---

### `POST /predict`
Computes the **baseline cumulative CAD risk** from **current age (≥40) to 80** using the baseline risk, then returns three educational overlays:

- `risk_ldl` (statin overlay),
- `risk_sbp` (antihypertensive overlay),
- `risk_combined` (multiplicative combination).

#### Request body (JSON)
```jsonc
{
  "sex": 1,                // 1 = male, 0 = female (aligns with training)
  "age_recruitment": 55,   // current age; projection will go up to 80
  "systolic_BP": 142,
  "HDL": 1.2,
  "LDL": 3.4,
  "diabetes": 0,
  "family_history": 1,
  "antiht": 0,
  "statin": 0,

  // optional predictors used in the preprocessing pipeline:
  "smoking_current": 0,
  "pack_years": 5,
  "waist_circumference": 100,
  "Lpa": 60,
  "CRP": 2.0,
  "GestHtPreEcl": null,
  "menopause": null,
  "HRT": null
}
```

> **Note**  
> The service applies the serialized preprocessors (imputation, scaling, OHE) **exactly as trained**, then calls the model’s `predict_risk(X, times)` on a yearly grid from `max(40, age_recruitment)` to `80`.

#### Response (JSON)
```jsonc
{
  "ages": [55, 56, ..., 80],               // x-axis in years
  "risk_baseline": [...],                  // cumulative risk per age (0..1)
  "risk_ldl": [...],                       // baseline × LDL overlay
  "risk_sbp": [...],                       // baseline × SBP overlay
  "risk_combined": [...],                  // baseline × (LDL × SBP overlays)

  "relative_risks": {
    "ldl_rr": 0.66,                        // (1 - LDL*0.5*0.2), clipped to [0,1]
    "bp_rr": 0.81,                         // (1 - ((SBP-129)/5)*0.08) if SBP≥130 else 1
    "combined_rr": 0.5346                  // product of the two
  },
  "assumptions": {
    "ldl_drop_fraction": 0.5,              // average 50% LDL reduction under statin
    "rr_per_mmol_ldl": 0.2,                // 20% risk reduction per 1 mmol/L LDL lowered
    "bp_target_threshold": 129.0,          // threshold for applying SBP overlay
    "rr_per_5mmhg": 0.08                   // 8% risk reduction per 5 mmHg lowered
  }
}
```

---

## Intervention overlays (exact formulas)

### 1) Statin / LDL overlay

Let `LDL_user` be the user’s LDL value (in mmol/L).  
We assume a **50% LDL reduction** under statin and **20% risk reduction per 1 mmol/L** LDL lowered. (See references)

**Relative risk (LDL)**  
```
ldl_rr = 1 - (LDL_user * 0.5 * 0.2)
        = 1 - 0.1 * LDL_user
```
`ldl_rr` is clipped to `[0, 1]`.

**Overlay**
```
risk_ldl(age) = risk_baseline(age) × ldl_rr
```

### 2) Antihypertensive / SBP overlay

If `SBP ≥ 130`, apply **8% risk reduction per 5 mmHg** lowered towards **129 mmHg**: (See references)
```
bp_rr = 1 - ((SBP - 129) / 5) * 0.08     if SBP ≥ 130
bp_rr = 1                                 if SBP < 130
```
`bp_rr` is clipped to `[0, 1]`.

**Overlay**
```
risk_sbp(age) = risk_baseline(age) × bp_rr
```

### 3) Combined overlay (independent effects, multiplicative)
```
combined_rr        = ldl_rr × bp_rr
risk_combined(age) = risk_baseline(age) × combined_rr
```

> **Educational note**  
> These overlays are **didactic** and **population-level** approximations, layered on the **model's predicted absolute risk**. They are **not** individualized causal effects nor clinical prescriptions.

---

## Docker

Build and run locally:
```bash
docker build -t dsm-cad-api:latest .
docker run --rm -p 8080:8080 dsm-cad-api:latest
```

The container listens on port **8080** and exposes:
- `GET /health`
- `POST /predict`

---

## Deploy to AWS App Runner (ECR → App Runner)

**Why App Runner?** A Docker image is pushed to **ECR** to create an App Runner **service from ECR**. App Runner manages **HTTPS**, **autoscaling**, and **updates**—ideal for **low traffic** and quick operations.  
(See AWS decision guides for modern app compute choices.)

1) **Create ECR repo** and push the image:
```bash
aws ecr get-login-password --region <region> \
 | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com

docker tag dsm-cad-api:latest <acct>.dkr.ecr.<region>.amazonaws.com/dsm-cad-api:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/dsm-cad-api:latest
```

2) **Create App Runner service**
- Source: **ECR** image `…/dsm-cad-api:latest`  
- Port: **8080**  
- Health check: `GET /health`  
- Autoscaling: default (tune later if needed)  
- Observability: enable basic logs

> After creation, App Runner returns a **public URL**.  
> Test `GET /health` and `POST /predict` against that URL.

---

## Configuration & notes

- **Models**: the service loads 4 `.joblib` artifacts at startup (two preprocessors, two DSM models). Keep these **private** in production and distribute via secure methods (e.g., encrypted S3 + CI/CD download) rather than committing to Git.
- **Performance**: DSM inference is fast; for low traffic App Runner’s smallest size is sufficient. Adjust `OMP_THREAD_LIMIT` if needed.
- **Internationalization**: API responses are language-agnostic; a front-end can render English/French labels and tooltips externally.
- **Lifelong projection**: the x-axis returns **ages** from `max(40, current age)` to **80**.

---

## Security & privacy

- **No data storage**: inputs are processed in memory and **not persisted**.  
- **No cookies / analytics** in the API layer.  
- This tool is **educational** and **does not** replace professional medical advice.
- Do **not** make clinical decisions based solely on the outputs.
- Risk overlays are **population-level** illustrations.

---

## FAQ

**Q: Are the overlays causal?**  
*A:* No. They are **educational**, based on fixed coefficients taken from randomized control trials meta-analyses, and applied on top of the DSM absolute risk curve. See the references.

**Q: Can I change the overlay parameters later?**  
*A:* Yes. Adjust constants in `app/main.py` (`ldl_drop_fraction=0.5`, `rr_per_mmol_ldl=0.2`, `bp_target_threshold=129.0`, `rr_per_5mmhg=0.08`), rebuild, redeploy.

---

## References

Mean reduction of LDL is 50% from statin therapies: https://onlinecjc.ca/article/S0828-282X(24)00358-1/fulltext, figure 4
Risk reduction of statin intake is 20% per mmol/L of LDL reduction. https://www.sciencedirect.com/science/article/pii/S0140673612603675?via%3Dihub
Hypertension is considered when systolic blood pressure (SBP) is >= 130 mmHg, but no medication should be given under 120 mmHg as the adverse effects may outweigh the benefits. Adverse effects: https://www.nejm.org/doi/full/10.1056/NEJMoa1511939
Risk reduction of CAD is 8% per 5 mmHg of systolic BP reduction. https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)00590-0/fulltext


---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
