from __future__ import annotations

from pydantic import BaseModel, Field


class LabItemInput(BaseModel):
    item_name: str = Field(min_length=1, max_length=64)
    test_value: str = Field(min_length=1, max_length=32)
    unit: str | None = Field(default=None, max_length=32)


class LabReportInterpretRequest(BaseModel):
    lab_item_list: list[LabItemInput] = Field(min_length=1)
    sync_to_archive: bool = False


class LabItemOutput(BaseModel):
    item_name: str
    test_value: str
    reference_range: str | None = None
    abnormal_flag: str | None = None
    meaning: str


class LabReportInterpretResponse(BaseModel):
    item_list: list[LabItemOutput]
    final_desc: str
