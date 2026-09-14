from __future__ import annotations

from sqlalchemy import select

from app.core.rag.entity_dictionary import _split_aliases
from app.db.database import get_sessionmaker
from app.db.models import LabItemAdvice, LabItemReferenceBase


def _norm_key(v: str | None) -> str:
    """查询键归一化：去首尾空白 + casefold（`WBC` 与 `wbc` 视为同一个键）。"""
    return (v or "").strip().casefold()


def _reference_lookup_keys(row: LabItemReferenceBase) -> list[str]:
    """一行参考数据能接受的全部查询键（已归一化、去重）。"""
    out: list[str] = []
    for raw in (row.item_name, row.item_en_name, *_split_aliases(row.item_alias)):
        key = _norm_key(raw)
        if key and key not in out:
            out.append(key)
    return out


class LabReferenceService:
    async def match_items(self, item_names: list[str]) -> list[dict]:
        """逐项在参考库里找对应条目；找不到时该项的 `match` 为 `None`。

        匹配口径：`item_name` / `item_en_name` / `item_alias` **任一命中即算命中**，
        且忽略首尾空白与英文大小写。

        别名这一路是必需的 —— 文本解析器（`lab_item_parser`）输出的是短名
        （白细胞 / 转氨酶），而化验单上印的是正式名（白细胞计数 / 丙氨酸氨基转移酶），
        只靠两个精确等值键两边永远对不上。

        实现说明：参考库是**种子数据**（几十行、运行时无写入路径），所以一次性载入
        建内存索引，而不是逐项发 SQL（原实现是 N 项 = N 次查询）。表变大到万级时
        需要换回 SQL 侧匹配，届时带索引的键表比现在的全表载入更合适。
        """
        if not item_names:
            return []

        async_session = get_sessionmaker()
        async with async_session() as session:
            rows = (
                (
                    await session.execute(
                        select(LabItemReferenceBase).where(LabItemReferenceBase.is_deleted == 0)
                    )
                )
                .scalars()
                .all()
            )

        # 正式名 / 英文名优先于别名：先建一轮，别名再补空缺。
        # 这样"同一个键既是 A 的别名、又是 B 的正式名"时由正式名赢，结果与行序无关。
        index: dict[str, LabItemReferenceBase] = {}
        for row in rows:
            for key in (_norm_key(row.item_name), _norm_key(row.item_en_name)):
                if key:
                    index.setdefault(key, row)
        for row in rows:
            for alias in _split_aliases(row.item_alias):
                key = _norm_key(alias)
                if key:
                    index.setdefault(key, row)

        out: list[dict] = []
        for name in item_names:
            row = index.get(_norm_key(name))
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

    async def get_advices(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        """批量取「偏高 / 偏低之后怎么办」的建议文本。

        入参 `pairs` 是 `(项目正式名, 方向)`，方向取 `H` / `L`；返回以同一二元组
        为键的字典。**没有对应行的项不会出现在结果里** —— 调用方据此判断
        "这一项没有建议可说"，而不是拿到一段空文本再去猜它是有意留白还是漏配。

        与 `match_items` 同一取舍：建议表是种子数据（几十行、运行时无写入路径），
        一次载入内存过滤，不逐项发 SQL。
        """
        if not pairs:
            return {}

        wanted = {(_norm_key(name), (direction or "").strip().upper()) for name, direction in pairs}

        async_session = get_sessionmaker()
        async with async_session() as session:
            rows = (
                (
                    await session.execute(
                        select(LabItemAdvice).where(LabItemAdvice.is_deleted == 0)
                    )
                )
                .scalars()
                .all()
            )

        out: dict[tuple[str, str], dict] = {}
        for row in rows:
            key = (_norm_key(row.item_name), (row.direction or "").strip().upper())
            if key in wanted:
                out[key] = {
                    "causes": row.causes,
                    "advice": row.advice,
                    "when_to_see_doctor": row.when_to_see_doctor,
                    "disclaimer": row.disclaimer,
                }
        return out
