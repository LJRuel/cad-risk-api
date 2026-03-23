import pandas as pd
from .deps import MEN_PREP, WOMEN_PREP, COMBINED_PREP

REQ_NUM = [
    'age_recruitment', 'systolic_BP', 'pack_years', 'waist_circumference',
    'HDL', 'LDL', 'Lpa', 'CRP'
]
REQ_CAT_MEN      = ['diabetes', 'family_history', 'statin', 'antiht']
REQ_CAT_WOM      = ['diabetes', 'family_history', 'statin', 'antiht', 'GestHtPreEcl', 'menopause', 'HRT']
REQ_CAT_COMBINED = ['diabetes', 'family_history', 'statin', 'antiht']  # no sex-specific features

# Fallback optional feature list used when the saved preprocessor predates the
# masking architecture (no "optional_features" key in the payload).
_OPTIONAL_FALLBACK = ['waist_circumference', 'Lpa', 'CRP']


def _apply_prep(prep: dict, cat: list, raw: dict) -> pd.DataFrame:
    """
    Shared preprocessing logic: apply fitted scaler/imputers to one observation.
    Handles optional feature masking (zero-fill + indicator columns).
    """
    df = pd.DataFrame([raw])
    num      = [n for n in REQ_NUM if n in df.columns]
    cat      = [c for c in cat if c in df.columns]
    optional = prep.get("optional_features", _OPTIONAL_FALLBACK)

    num_imp = prep.get("numeric_imputer")
    scaler  = prep.get("scaler")
    cat_imp = prep.get("categorical_imputer")
    ohe     = prep.get("onehot_encoder")
    one_hot = prep.get("one_hot_encode", False)

    absent_optional = []
    if scaler is not None:
        scaler_means = dict(zip(num, scaler.mean_))
        for feat in optional:
            if feat in num and raw.get(feat) is None:
                df[feat] = scaler_means.get(feat, 0.0)
                absent_optional.append(feat)

    if num:
        if num_imp is not None:
            Xn_imp = pd.DataFrame(num_imp.transform(df[num]), columns=num)
        else:
            Xn_imp = df[num].copy()
        Xn = pd.DataFrame(scaler.transform(Xn_imp), columns=num) if scaler is not None else Xn_imp
        for feat in absent_optional:
            if feat in Xn.columns:
                Xn[feat] = 0.0
    else:
        Xn = pd.DataFrame()

    if cat:
        if cat_imp is not None:
            Xc_imp = pd.DataFrame(cat_imp.transform(df[cat]), columns=cat)
        else:
            Xc_imp = df[cat].copy()
        if one_hot and ohe is not None:
            Xc = pd.DataFrame(ohe.transform(Xc_imp), columns=list(ohe.get_feature_names_out(cat)))
        else:
            Xc = Xc_imp
    else:
        Xc = pd.DataFrame()

    Xi_data = {}
    for feat in optional:
        if feat in num:
            Xi_data[f"{feat}_missing"] = [1.0 if feat in absent_optional else 0.0]
    Xi = pd.DataFrame(Xi_data)

    return pd.concat(
        [Xn.reset_index(drop=True), Xc.reset_index(drop=True), Xi.reset_index(drop=True)],
        axis=1,
    )


def apply_preprocessor_one(sex: int, raw: dict) -> pd.DataFrame:
    """
    Apply the serialized preprocessors to one observation and return a 1×N
    DataFrame ready for MODEL.predict_risk.

    Optional features (waist_circumference, Lpa, CRP) that are None in `raw`
    are handled without imputation:
      - Their value is set to 0.0 in scaled space (= training mean).
      - A binary indicator column ({feature}_missing = 1.0) is appended.
    Present optional features pass through the imputer and scaler normally,
    with their indicator set to 0.0.

    This mirrors the masking architecture used during model training.
    """
    prep = MEN_PREP if sex == 1 else WOMEN_PREP
    cat  = REQ_CAT_MEN if sex == 1 else REQ_CAT_WOM
    return _apply_prep(prep, cat, raw)


def apply_preprocessor_combined(raw: dict) -> pd.DataFrame:
    """
    Apply the combined-sex preprocessor to one observation.

    Used when the patient has not reported biological sex. Does not include
    sex-specific features (GestHtPreEcl, menopause, HRT).
    Optional feature masking follows the same convention as apply_preprocessor_one.
    """
    return _apply_prep(COMBINED_PREP, REQ_CAT_COMBINED, raw)
