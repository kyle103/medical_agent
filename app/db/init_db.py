from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.common.logger import get_logger
from app.config.settings import settings
from app.db.database import get_engine
from app.db.models import Base, DrugKnowledgeBase, LabItemReferenceBase

logger = get_logger(__name__)


async def init_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await _migrate_missing_columns(engine)


async def _migrate_missing_columns(engine: AsyncEngine) -> None:
    def _do_migrate(sync_conn) -> None:
        model_tables = Base.metadata.tables
        for table_name, table_obj in model_tables.items():
            try:
                existing_cols = {row["name"] for row in sync_conn.execute(text(f"PRAGMA table_info({table_name})")).mappings().all()}
            except Exception:
                continue
            for col in table_obj.columns:
                if col.name not in existing_cols:
                    col_type = str(col.type).upper()
                    nullable = "NULL" if col.nullable else "NOT NULL"
                    default = ""
                    if col.server_default is not None:
                        default = f"DEFAULT {col.server_default.arg}"
                    elif col.nullable:
                        default = "DEFAULT NULL"
                    sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type} {nullable} {default}".strip()
                    logger.info("Migrating missing column: %s", sql)
                    sync_conn.execute(text(sql))

    async with engine.begin() as conn:
        await conn.run_sync(_do_migrate)


async def import_min_kb(engine: AsyncEngine) -> None:
    kb_dir = Path("data/knowledge_base")
    drug_csv = kb_dir / "drug_knowledge.csv"
    lab_csv = kb_dir / "lab_item_reference.csv"

    from sqlalchemy.ext.asyncio import async_sessionmaker

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        existing_drug_names = set(
            (await session.execute(select(DrugKnowledgeBase.drug_name))).scalars().all()
        )
        existing_item_names = set(
            (await session.execute(select(LabItemReferenceBase.item_name))).scalars().all()
        )
        if drug_csv.exists():
            with drug_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    drug_name = (row.get("drug_name", "") or "").strip()
                    if not drug_name or drug_name in existing_drug_names:
                        continue
                    existing_drug_names.add(drug_name)
                    session.add(
                        DrugKnowledgeBase(
                            drug_name=drug_name,
                            drug_alias=row.get("drug_alias") or None,
                            indications=row.get("indications") or None,
                            contraindications=row.get("contraindications") or None,
                            side_effects=row.get("side_effects") or None,
                            interaction_drugs=row.get("interaction_drugs") or None,
                            interaction_desc=row.get("interaction_desc") or None,
                        )
                    )

        if lab_csv.exists():
            with lab_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item_name = (row.get("item_name", "") or "").strip()
                    if not item_name or item_name in existing_item_names:
                        continue
                    existing_item_names.add(item_name)
                    session.add(
                        LabItemReferenceBase(
                            item_name=item_name,
                            item_en_name=row.get("item_en_name", "").strip(),
                            reference_range=row.get("reference_range", "").strip(),
                            unit=row.get("unit") or None,
                            high_meaning=row.get("high_meaning") or None,
                            low_meaning=row.get("low_meaning") or None,
                        )
                    )

        await session.commit()


def ensure_min_csv() -> None:
    os.makedirs("data/knowledge_base", exist_ok=True)

    drug_csv = Path("data/knowledge_base/drug_knowledge.csv")
    if not drug_csv.exists():
        drug_csv.write_text(
            "drug_name,drug_alias,indications,contraindications,side_effects,interaction_drugs,interaction_desc\n"
            "对乙酰氨基酚,扑热息痛,解热镇痛,对本品过敏者禁用,恶心等,[],{}\n",
            encoding="utf-8",
        )

    lab_csv = Path("data/knowledge_base/lab_item_reference.csv")
    if not lab_csv.exists():
        lab_csv.write_text(
            "item_name,item_en_name,reference_range,unit,high_meaning,low_meaning\n"
            "血糖,GLU,3.9-6.1,mmol/L,可能与饮食/应激等因素相关，建议结合复查与医生意见进行评估。,可能与进食不足等因素相关，建议结合复查与医生意见进行评估。\n",
            encoding="utf-8",
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="local", choices=["local", "prod"])
    args = parser.parse_args()

    os.environ["APP_ENV"] = args.env

    ensure_min_csv()
    engine = get_engine()
    await init_schema(engine)
    await import_min_kb(engine)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
