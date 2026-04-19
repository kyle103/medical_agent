import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.asyncio
async def test_response_plan_injects_memory_without_keywords():
    """不依赖“你还记得/总结”等关键词：当用户追问很短且含指代时应自动注入会话记忆并走 LLM 生成链路。

    这里不要求真实 external LLM 可用：如果未配置 LLM，会回退到 tool_result 文本。
    重点验证：接口不 500、且状态机能写入记忆，第二轮可以利用记忆注入（从输出可见性角度做弱断言）。
    """

    client = TestClient(app)

    r = client.post(
        "/api/v1/user/register",
        json={
            "user_nickname": f"t_{uuid.uuid4().hex[:8]}",
            "phone": f"u_{uuid.uuid4().hex[:12]}",
            "password": "test123456",
        },
    )
    assert r.status_code == 200
    data = r.json()["data"]
    token = data.get("access_token") or data.get("token") or data.get("jwt")
    assert token

    headers = {"Authorization": f"Bearer {token}"}
    session_id = f"s_{uuid.uuid4().hex}"

    r1 = client.post(
        "/api/v1/chat/completion",
        headers=headers,
        json={"session_id": session_id, "user_input": "我叫张三", "stream": False, "enable_archive_link": True},
    )
    assert r1.status_code == 200

    # 第二轮不带关键词，但用短追问+指代触发 inject_memory
    r2 = client.post(
        "/api/v1/chat/completion",
        headers=headers,
        json={"session_id": session_id, "user_input": "那呢？", "stream": False, "enable_archive_link": True},
    )
    assert r2.status_code == 200

    out = r2.json()["data"]["assistant_output"]
    assert isinstance(out, str) and len(out) > 0
