import asyncio

import pytest

from app.db.init_db import init_schema
from app.db.database import get_engine
from app.db.crud.user_crud import UserCRUD
import uuid

@pytest.mark.asyncio
async def test_multi_user_isolation_user_table():
    engine = get_engine()
    await init_schema(engine)

    crud = UserCRUD()
    u1_id = f"u1_{uuid.uuid4().hex}"
    u2_id = f"u2_{uuid.uuid4().hex}"

    await crud.create_user(user_id=u1_id, user_nickname="n1")
    await crud.create_user(user_id=u2_id, user_nickname="n2")

    u1 = await crud.get_user(user_id=u1_id)
    u2 = await crud.get_user(user_id=u2_id)

    assert u1 is not None and u1.user_id == u1_id
    assert u2 is not None and u2.user_id == u2_id
