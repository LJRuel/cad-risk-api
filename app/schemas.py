from typing import Optional
from pydantic import BaseModel, Field, model_validator

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
    smoking_current: Optional[int] = None  # 0=never, 1=current, 2=former (UI only, not a model feature)
    pack_years: Optional[float] = None
    waist_circumference: Optional[float] = None
    Lpa: Optional[float] = None
    CRP: Optional[float] = None
    GestHtPreEcl: Optional[int] = None
    menopause: Optional[int] = None
    HRT: Optional[int] = None

    @model_validator(mode="after")
    def validate_pack_years(self) -> "InputPayload":
        if self.smoking_current in (1, 2):
            if not self.pack_years or self.pack_years <= 0:
                raise ValueError("pack_years must be provided and > 0 for current or former smokers.")
        else:
            self.pack_years = 0.0
        return self