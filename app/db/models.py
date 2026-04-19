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
    reference_range: Mapped[str] = mapped_column(String(64))
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    high_meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
    low_meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    start_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    prescribe_hospital: Mapped[str | None] = mapped_column(String(128), nullable=True)
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)


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
