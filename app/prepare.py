import pandas as pd
from .deps import CLINICAL_PREPS, COMBINED_PREPS

REQ_CAT_MEN      = ['diabetes', 'family_history', 'statin', 'antiht']
REQ_CAT_WOM      = ['diabetes', 'family_history', 'statin', 'antiht', 'GestHtPreEcl', 'menopause', 'HRT']
REQ_CAT_COMBINED = ['diabetes', 'family_history', 'statin', 'antiht']  # no sex-specific features


def _apply_prep(prep: dict, cat: list, raw: dict) -> pd.DataFrame:
    """
    Shared preprocessing logic: apply fitted scaler/imputers to one observation.

    The numeric feature list is read from prep["numeric_features"] so that the
    correct columns are used for each variant (absent optional features are simply
    not present in the preprocessor and not passed here).

    Args:
        prep: Preprocessor dict saved by data_preparation.py.
        cat:  List of categorical feature names relevant for this sex/context.
        raw:  Dict of raw feature values from the request payload.

    Returns:
        1-row DataFrame ready for model.predict_risk().
    """
    df = pd.DataFrame([raw])

    num     = prep.get("numeric_features", [])
    cat     = [c for c in cat if c in df.columns]

    num_imp = prep.get("numeric_imputer")
    scaler  = prep.get("scaler")
    cat_imp = prep.get("categorical_imputer")
    ohe     = prep.get("onehot_encoder")
    one_hot = prep.get("one_hot_encode", False)

    if num:
        df_num = df[[c for c in num if c in df.columns]].copy()
        if num_imp is not None:
            Xn_imp = pd.DataFrame(num_imp.transform(df_num), columns=df_num.columns)
        else:
            Xn_imp = df_num.copy()
        Xn = pd.DataFrame(scaler.transform(Xn_imp), columns=df_num.columns) if scaler is not None else Xn_imp
    else:
        Xn = pd.DataFrame()

    if cat:
        df_cat = df[[c for c in cat if c in df.columns]].copy()
        if cat_imp is not None:
            Xc_imp = pd.DataFrame(cat_imp.transform(df_cat), columns=df_cat.columns)
        else:
            Xc_imp = df_cat.copy()
        if one_hot and ohe is not None:
            Xc = pd.DataFrame(ohe.transform(Xc_imp), columns=list(ohe.get_feature_names_out(cat)))
        else:
            Xc = Xc_imp
    else:
        Xc = pd.DataFrame()

    return pd.concat(
        [Xn.reset_index(drop=True), Xc.reset_index(drop=True)],
        axis=1,
    )


def apply_preprocessor_one(variant: str, sex: int, raw: dict) -> pd.DataFrame:
    """
    Apply the serialized clinical preprocessor for one sex and variant to a
    single observation and return a 1×N DataFrame ready for model.predict_risk.

    The preprocessor's "numeric_features" key determines which numeric columns
    are used, so absent optional features are handled automatically — each
    variant's preprocessor was trained only on the features included in that
    variant.

    Args:
        variant: One of "full", "lpa", "crp", "base".
        sex:     1 for men, 0 for women.
        raw:     Dict of raw feature values from the request payload
                 (smoking_current already removed).

    Returns:
        1-row DataFrame.
    """
    prep = CLINICAL_PREPS.get((variant, sex))
    cat  = REQ_CAT_MEN if sex == 1 else REQ_CAT_WOM
    return _apply_prep(prep, cat, raw)


def apply_preprocessor_combined(variant: str, raw: dict) -> pd.DataFrame:
    """
    Apply the combined-sex preprocessor for the given variant to one observation.

    Used when the patient has not reported biological sex. Does not include
    sex-specific features (GestHtPreEcl, menopause, HRT).

    Args:
        variant: One of "full", "lpa", "crp", "base".
        raw:     Dict of raw feature values from the request payload
                 (smoking_current already removed).

    Returns:
        1-row DataFrame.
    """
    prep = COMBINED_PREPS.get(variant)
    return _apply_prep(prep, REQ_CAT_COMBINED, raw)
