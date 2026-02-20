import pandas as pd
from .deps import MEN_PREP, WOMEN_PREP

REQ_NUM = [
    'age_recruitment','systolic_BP','pack_years','waist_circumference',
    'HDL','LDL','Lpa','CRP'
]
REQ_CAT_MEN = ['diabetes','family_history','statin','antiht']
REQ_CAT_WOM = ['diabetes','family_history','statin','antiht','GestHtPreEcl','menopause','HRT']

def apply_preprocessor_one(sex:int, raw:dict) -> pd.DataFrame:
    """
    Applies the serialized preprocessors (imputation + scaling + OHE) on 1 observation.
    Returns a 1xN DataFrame ready for MODEL.predict_risk.
    """
    df = pd.DataFrame([raw])
    prep = MEN_PREP if sex==1 else WOMEN_PREP
    cat = [c for c in (REQ_CAT_MEN if sex==1 else REQ_CAT_WOM) if c in df.columns]
    num = [n for n in REQ_NUM if n in df.columns]

    num_imp = prep.get("numeric_imputer")
    scaler  = prep.get("scaler")
    cat_imp = prep.get("categorical_imputer")
    ohe     = prep.get("onehot_encoder")
    one_hot = prep.get("one_hot_encode", False)

    # Numeric: impute -> scale
    if num:
        if num_imp is not None:
            Xn_imp = pd.DataFrame(num_imp.transform(df[num]), columns=num)
        else:
            Xn_imp = df[num].copy()
        Xn = pd.DataFrame(scaler.transform(Xn_imp), columns=num) if scaler is not None else Xn_imp
    else:
        Xn = pd.DataFrame()

    # Categorial: impute -> OHE if present
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

    X = pd.concat([Xn.reset_index(drop=True), Xc.reset_index(drop=True)], axis=1)
    return X