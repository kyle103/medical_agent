from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()


class UserInfo(Base):
    __tablename__ = "user_info"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 新增：用于长期身份识别（登录名）
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # 新增：密码哈希（不保存明文）
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)

    user_nickname: Mapped[str] = mapped_column(String(32), default="匿名用户")
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


class UserChatRecord(Base):
    __tablename__ = "user_chat_records"

    chat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    session_id: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("idx_user_session", "user_id", "session_id"),
        Index("idx_create_time", "user_id", "create_time"),
    )


class UserChatSummary(Base):
    """会话摘要（MemoryService 懒更新：按 covered_chat_id 增量维护）。"""

    __tablename__ = "user_chat_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    session_id: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(Text)
    covered_chat_id: Mapped[int] = mapped_column(Integer, default=0)  # 已总结到的最大 chat_id
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_chat_summary", "user_id", "session_id"),)


class UserLongMemoryCursor(Base):
    """长期记忆批量写入游标（记录某会话已提取到哪条 chat_id，避免重复提取）。"""

    __tablename__ = "user_long_memory_cursor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    session_id: Mapped[str] = mapped_column(String(64))
    chat_id: Mapped[int] = mapped_column(Integer, default=0)
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("idx_lm_cursor", "user_id", "session_id"),)


class AgentSessionState(Base):
    __tablename__ = "agent_session_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    session_id: Mapped[str] = mapped_column(String(64))
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("idx_agent_session", "user_id", "session_id"),)


class DrugKnowledgeBase(Base):
    __tablename__ = "drug_knowledge_base"

    drug_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    drug_alias: Mapped[str | None] = mapped_column(String(512), nullable=True)
    indications: Mapped[str | None] = mapped_column(Text, nullable=True)
    contraindications: Mapped[str | None] = mapped_column(Text, nullable=True)
    side_effects: Mapped[str | None] = mapped_column(Text, nullable=True)
    interaction_drugs: Mapped[str | None] = mapped_column(Text, nullable=True)
    interaction_desc: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


class LabItemReferenceBase(Base):
    __tablename__ = "lab_item_reference_base"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    item_en_name: Mapped[str] = mapped_column(String(64), index=True)
    #: 别名，用 `,` `，` `、` `;` `；` `/` `|` 分隔（与 `drug_alias` 同一套切分口径，
    #: 复用 `entity_dictionary._split_aliases`）。
    #: 为什么必需：`lab_item_parser` 的白名单输出的是**短名**（白细胞 / 红细胞 /
    #: 血小板 / 转氨酶 / 胆固醇 / 尿酸），而化验单上印的是**正式名**
    #: （白细胞计数 / 丙氨酸氨基转移酶 / 总胆固醇）。只靠 item_name + item_en_name
    #: 两个精确等值键，文本链路抽出来的名字一个都对不上 → 全部落到"暂未纳入参考库"。
    item_alias: Mapped[str | None] = mapped_column(String(256), nullable=True)
    reference_range: Mapped[str] = mapped_column(String(64))
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    high_meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
    low_meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


class LabItemAdvice(Base):
    """指标"偏高/偏低之后怎么办"的建议文本（`lab_item_advice.csv` 种子数据）。

    为什么单独一张表、而不是给 `LabItemReferenceBase` 继续加列
    ------------------------------------------------------------
    1. **方向是行维度，不是列维度。** 一个项目有偏高、偏低两条独立文本，
       若做成 `high_causes/high_advice/…/low_causes/low_advice/…` 就要 8 列，
       主表从 7 列变 15 列，且加"严重度分层""人群分层"时列数再翻倍。
       按 `(item_name, direction)` 一行一条，后续加维度只加列、不加组。
    2. **缺失即不写行**（而不是写空串）。有些项目只有一个方向有话说，
       空串行会让"有没有内容"变成"字符串是否为空"的判断，容易写错。

    为什么用 `item_name` 关联而不是 `item_id`
    -----------------------------------------
    与 CSV 种子数据的导入口径一致（`init_db` 按 `item_name` upsert），
    也与 `LabReferenceService.match_items` 返回的键一致 —— 用 `item_id` 就要在
    导入时先查主表拿 id，多一层脆弱耦合，而这张表只有几十行。

    内容边界（**合规**）：`advice` 只写生活方式、复查节奏与"何时必须就医"，
    **不写用药与剂量** —— 那是医药建议，超出本系统的资质范围。
    """

    __tablename__ = "lab_item_advice"

    advice_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: 与 `lab_item_reference.item_name` 对应的正式项目名
    item_name: Mapped[str] = mapped_column(String(64), index=True)
    #: `H` = 偏高，`L` = 偏低
    direction: Mapped[str] = mapped_column(String(2))
    causes: Mapped[str | None] = mapped_column(Text, nullable=True)
    advice: Mapped[str | None] = mapped_column(Text, nullable=True)
    when_to_see_doctor: Mapped[str | None] = mapped_column(Text, nullable=True)
    disclaimer: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


# 业务档案表（MVP：保留字段但不做诊断推断；仅存储用户录入内容）
class UserDrugRecord(Base):
    __tablename__ = "user_drug_records"

    drug_record_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    drug_name: Mapped[str] = mapped_column(String(128), index=True)
    drug_alias: Mapped[str | None] = mapped_column(String(256), nullable=True)
    dosage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    frequency: Mapped[str | None] = mapped_column(String(64), nullable=True)
    start_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)          # 服药日期（年月日）
    start_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)      # 服药时间（年月日+时刻，如 2026-08-08 20:00）
    intake_time_text: Mapped[str | None] = mapped_column(String(64), nullable=True)   # 原始时间描述（"昨天晚上八点"）
    end_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    prescribe_hospital: Mapped[str | None] = mapped_column(String(128), nullable=True)
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotent_key: Mapped[str | None] = mapped_column(String(32), unique=True, index=True, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("idx_user_drug_dedup", "user_id", "drug_name", "dosage", "is_deleted"),
        Index("idx_idempotent_key", "idempotent_key"),
    )


class UserLabReportRecord(Base):
    __tablename__ = "user_lab_report_records"

    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    report_name: Mapped[str] = mapped_column(String(128))
    test_time: Mapped[datetime] = mapped_column(Date)
    test_organization: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


class UserLabReportItem(Base):
    __tablename__ = "user_lab_report_items"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_lab_report_records.report_id")
    )
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    item_name: Mapped[str] = mapped_column(String(64))
    item_en_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    test_value: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reference_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    abnormal_flag: Mapped[str | None] = mapped_column(String(8), nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


class UserMedicalRecord(Base):
    __tablename__ = "user_medical_records"

    record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("user_info.user_id"))
    visit_time: Mapped[datetime] = mapped_column(Date)
    hospital_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    department_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnosis_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    doctor_advice: Mapped[str | None] = mapped_column(Text, nullable=True)
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("idx_visit_time", "user_id", "visit_time"),)
