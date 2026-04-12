from __future__ import annotations

from sqlalchemy import or_, select

from app.db.database import get_sessionmaker
from app.db.models import LabItemReferenceBase


class LabReferenceService:
    async def match_items(self, item_names: list[str]) -> list[dict]:
        if not item_names:
            return []

        async_session = get_sessionmaker()
        out: list[dict] = []

        async with async_session() as session:
            for name in item_names:
                q = select(LabItemReferenceBase).where(
                    LabItemReferenceBase.is_deleted == 0,
                    or_(
                        LabItemReferenceBase.item_name == name,
                        LabItemReferenceBase.item_en_name == name,
                    ),
                )
                res = await session.execute(q)
                row = res.scalars().first()
                out.append(
                    {
                        "query": name,
                        "match": None
                        if not row
                        else {
                            "item_name": row.item_name,
                            "item_en_name": row.item_en_name,
                            "reference_range": row.reference_range,
                            "unit": row.unit,
                            "high_meaning": row.high_meaning,
                            "low_meaning": row.low_meaning,
                        },
                    }
                )
        return out
