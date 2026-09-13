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
    """把 CSV 种子数据导入知识库表。

    **语义是 upsert，不是 insert-if-absent。** 这两张表是纯种子数据（运行时没有任何
    写入路径：`app/db/crud/` 下只有 archive/user），CSV 就是唯一事实源。
    早先的"已存在就跳过"会让**改 CSV 后重跑导入静默无效** —— 看着像更新了、
    库里其实没变，属于最难排查的一类故障；改列（如新增 `item_alias`）时
    老行也永远拿不到新字段。upsert 后 CSV 的改动一定落地。
    """
    kb_dir = Path("data/knowledge_base")
    drug_csv = kb_dir / "drug_knowledge.csv"
    lab_csv = kb_dir / "lab_item_reference.csv"

    from sqlalchemy.ext.asyncio import async_sessionmaker

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        existing_drugs = {
            row.drug_name: row
            for row in (await session.execute(select(DrugKnowledgeBase))).scalars().all()
        }
        existing_items = {
            row.item_name: row
            for row in (await session.execute(select(LabItemReferenceBase))).scalars().all()
        }

        if drug_csv.exists():
            with drug_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    drug_name = (row.get("drug_name", "") or "").strip()
                    if not drug_name:
                        continue
                    fields = {
                        "drug_alias": row.get("drug_alias") or None,
                        "indications": row.get("indications") or None,
                        "contraindications": row.get("contraindications") or None,
                        "side_effects": row.get("side_effects") or None,
                        "interaction_drugs": row.get("interaction_drugs") or None,
                        "interaction_desc": row.get("interaction_desc") or None,
                    }
                    obj = existing_drugs.get(drug_name)
                    if obj is None:
                        obj = DrugKnowledgeBase(drug_name=drug_name, **fields)
                        session.add(obj)
                        existing_drugs[drug_name] = obj
                    else:
                        for k, v in fields.items():
                            setattr(obj, k, v)

        if lab_csv.exists():
            with lab_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    item_name = (row.get("item_name", "") or "").strip()
                    if not item_name:
                        continue
                    fields = {
                        "item_en_name": (row.get("item_en_name", "") or "").strip(),
                        "item_alias": row.get("item_alias") or None,
                        "reference_range": (row.get("reference_range", "") or "").strip(),
                        "unit": row.get("unit") or None,
                        "high_meaning": row.get("high_meaning") or None,
                        "low_meaning": row.get("low_meaning") or None,
                    }
                    obj = existing_items.get(item_name)
                    if obj is None:
                        obj = LabItemReferenceBase(item_name=item_name, **fields)
                        session.add(obj)
                        existing_items[item_name] = obj
                    else:
                        for k, v in fields.items():
                            setattr(obj, k, v)

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
            "item_name,item_en_name,item_alias,reference_range,unit,high_meaning,low_meaning\n"
            "血糖,GLU,空腹血糖/葡萄糖,3.9-6.1,mmol/L,"
            "通用科普信息：升高可能与饮食、应激等因素相关，建议结合复查与医生意见综合评估。,"
            "通用科普信息：降低可能与进食不足等因素相关，建议结合复查与医生意见综合评估。\n",
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
