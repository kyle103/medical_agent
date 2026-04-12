from __future__ import annotations

import json

from sqlalchemy import or_, select

from app.common.exceptions import NotFoundException
from app.db.database import get_sessionmaker
from app.db.models import DrugKnowledgeBase


class DrugKnowledgeService:
    async def match_drugs(self, drug_names: list[str]) -> list[dict]:
        if not drug_names:
            return []

        async_session = get_sessionmaker()
        results: list[dict] = []

        async with async_session() as session:
            for name in drug_names:
                q = select(DrugKnowledgeBase).where(
                    DrugKnowledgeBase.is_deleted == 0,
                    or_(
                        DrugKnowledgeBase.drug_name == name,
                        DrugKnowledgeBase.drug_alias.like(f"%{name}%"),
                    ),
                )
                res = await session.execute(q)
                row = res.scalars().first()
                if not row:
                    results.append({"query": name, "match": None})
                else:
                    results.append(
                        {
                            "query": name,
                            "match": {
                                "drug_name": row.drug_name,
                                "drug_alias": row.drug_alias,
                                "interaction_drugs": row.interaction_drugs,
                                "interaction_desc": row.interaction_desc,
                            },
                        }
                    )

        return results

    @staticmethod
    def parse_interactions(row_match: dict) -> tuple[list[str], dict]:
        drugs_raw = row_match.get("interaction_drugs") or "[]"
        desc_raw = row_match.get("interaction_desc") or "{}"
        try:
            drugs = json.loads(drugs_raw)
            if not isinstance(drugs, list):
                drugs = []
        except Exception:
            drugs = []

        try:
            desc = json.loads(desc_raw)
            if not isinstance(desc, dict):
                desc = {}
        except Exception:
            desc = {}

        return drugs, desc
