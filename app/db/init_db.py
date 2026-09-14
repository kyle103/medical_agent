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
from app.db.models import Base, DrugKnowledgeBase, LabItemAdvice, LabItemReferenceBase

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

    **语义是 upsert，不是 insert-if-absent。** 这三张表是纯种子数据（运行时没有任何
    写入路径：`app/db/crud/` 下只有 archive/user），CSV 就是唯一事实源。
    早先的"已存在就跳过"会让**改 CSV 后重跑导入静默无效** —— 看着像更新了、
    库里其实没变，属于最难排查的一类故障；改列（如新增 `item_alias`）时
    老行也永远拿不到新字段。upsert 后 CSV 的改动一定落地。
    """
    kb_dir = Path("data/knowledge_base")
    drug_csv = kb_dir / "drug_knowledge.csv"
    lab_csv = kb_dir / "lab_item_reference.csv"
    lab_advice_csv = kb_dir / "lab_item_advice.csv"

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
        #: 建议表的主键是 `(item_name, direction)` —— 一个项目两条独立文本，
        #: 所以这里用二元组建索引，不能用单个 item_name。
        existing_advices = {
            (row.item_name, (row.direction or "").strip().upper()): row
            for row in (await session.execute(select(LabItemAdvice))).scalars().all()
        }

        if drug_csv.exists():
            # 三处 CSV 读取统一用 `utf-8-sig`：BOM 会让 `DictReader` 的第一列名带上
            # `\ufeff`，于是所有行都被"主键为空"挡掉 —— 静默导入零行。
            # 详见下面 advice 分支的注释。
            with drug_csv.open("r", encoding="utf-8-sig") as f:
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
            with lab_csv.open("r", encoding="utf-8-sig") as f:
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

        if lab_advice_csv.exists():
            # 用 `utf-8-sig` 而不是 `utf-8`：这份 CSV 曾被编辑器加上 BOM，
            # `DictReader` 的第一列名于是变成 `\ufeffitem_name`，
            # 所有行都被"item_name 为空"挡掉 —— **零行导入、且不报任何错**，
            # 表现是"表建好了、永远是空的"。`utf-8-sig` 严格更宽
            # （没有 BOM 时行为与 utf-8 完全一致），所以用它没有代价。
            with lab_advice_csv.open("r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    item_name = (row.get("item_name", "") or "").strip()
                    direction = (row.get("direction", "") or "").strip().upper()
                    if not item_name or direction not in ("H", "L"):
                        # 方向不在 H/L 的行直接跳过：留着只会变成一条永远查不到的死数据
                        continue
                    fields = {
                        "causes": row.get("causes") or None,
                        "advice": row.get("advice") or None,
                        "when_to_see_doctor": row.get("when_to_see_doctor") or None,
                        "disclaimer": row.get("disclaimer") or None,
                    }
                    obj = existing_advices.get((item_name, direction))
                    if obj is None:
                        obj = LabItemAdvice(item_name=item_name, direction=direction, **fields)
                        session.add(obj)
                        existing_advices[(item_name, direction)] = obj
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

    #: 建议表的最小种子：只为保证"表非空、查询路径可跑通"。
    #: 真正的内容在仓库自带的 `data/knowledge_base/lab_item_advice.csv` 里。
    advice_csv = Path("data/knowledge_base/lab_item_advice.csv")
    if not advice_csv.exists():
        advice_csv.write_text(
            "item_name,direction,causes,advice,when_to_see_doctor,disclaimer\n"
            "血糖,H,单次升高常见于检测前进食或应激。,建议空腹复查并记录数值。,"
            "多次复查仍偏高请到内分泌科就诊。,以上为通用健康提示，不能替代医生诊断。\n",
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
