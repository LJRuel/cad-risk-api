from pydantic import BaseModel, Field

class Interventions(BaseModel):
    """
    Interventions parameters.
    """
    delta_ldl_mmol: float = Field(0.0, description="LDL-C reduction in mmol/L (step of 0.1)")
    delta_sbp_mmHg: int   = Field(0,   description="SBP reduction in mmHg (step of 1)")

class InputPayload(BaseModel):
    sex: int = Field(..., ge=0, le=1) # 1=men, 0=women
    age_recruitment: float
    systolic_BP: float
    HDL: float
    LDL: float
    diabetes: int
    family_history: int
    antiht: int
    statin: int
    smoking_current: int | None = None
    pack_years: float | None = 0.0
    waist_circumference: float | None = None
    Lpa: float | None = None
    CRP: float | None = None
    GestHtPreEcl: int | None = None
    menopause: int | None = None
    HRT: int | None = None

    interventions: Interventions = Interventions()