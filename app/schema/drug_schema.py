from __future__ import annotations

from pydantic import BaseModel, Field


class DrugInteractionCheckRequest(BaseModel):
    drug_name_list: list[str] = Field(min_length=1)
    sync_to_archive: bool = False


class DrugMatchItem(BaseModel):
    drug_name: str
    match_status: str


class DrugInteractionItem(BaseModel):
    drug_a: str
    drug_b: str
    risk_level: str
    interaction_desc: str


class DrugInteractionCheckResponse(BaseModel):
    drug_list: list[DrugMatchItem]
    interaction_result: list[DrugInteractionItem]
    final_desc: str
