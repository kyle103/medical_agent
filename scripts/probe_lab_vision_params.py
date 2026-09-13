"""视觉模型接入前的参数探针：schema 档位 + 思考开关。

接入前必须确认两件事，否则要么 schema 静默不生效、要么参数不被接受直接 400：

1. `structured_output.resolve_mode(client, VISION_MODEL)` 能不能判定视觉模型的档位
   （决定 `response_format` 是否下发；注意模式缓存按 `(base_url, model)` 分键，
   视觉模型是**独立的一次探测**，不会蹭主模型的结论）。
2. 视觉模型是否接受 `extra_body={"enable_thinking": False}`。
   本项目文本侧的 `LLM_DISABLE_THINKING` 写在文档里，但那是 `deepseek-v4-flash` 的实测；
   **换模型/换提供方要重测**，这就是这次要测的。

用法：python scripts/probe_lab_vision_params.py [vision_model]
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from app.config.settings import settings  # noqa: E402
from app.core.llm.llm_service import get_shared_client  # noqa: E402
from app.core.llm.structured_output import resolve_mode  # noqa: E402
from probe_lab_vision import _PROMPT, _SCHEMA, _score, build_image  # noqa: E402


async def _call(*, client, model: str, b64: str, mime: str, thinking: bool | None) -> dict:
    """thinking=None 表示不传该参数。"""
    kwargs: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        "max_tokens": 1500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "lab_report", "strict": True, "schema": _SCHEMA},
        },
    }
    if thinking is not None:
        kwargs["extra_body"] = {"enable_thinking": thinking}

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(**kwargs)
    dt = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    out_tok = getattr(usage, "completion_tokens", None) if usage else None
    try:
        items = (json.loads(raw) or {}).get("items") or []
    except Exception:  # noqa: BLE001
        items = []
    return {"dt": dt, "raw_len": len(raw), "out_tok": out_tok, "score": _score(items)}


async def main() -> int:
    vision_model = sys.argv[1] if len(sys.argv) > 1 else settings.LLM_VISION_MODEL_NAME

    print("=" * 92)
    print(f"视觉模型参数探针 | model={vision_model} | base_url={settings.LLM_API_BASE}")
    print("=" * 92)

    client = await get_shared_client()

    # ---- 1. schema 档位 ----
    print("\n[1] structured_output 档位判定")
    try:
        mode = await resolve_mode(client, vision_model)
        print(f"  resolve_mode({vision_model}) = {mode}")
        print("  → 接入时可复用 build_response_format()，按同一套档位逻辑下发 response_format")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 档位判定失败：{type(e).__name__} {str(e)[:160]}")
        print("  → 接入层必须自带客户端校验（本项目本来就是这么设计的）")

    # ---- 2. 思考开关 ----
    b64, mime = build_image(degraded=False)
    print("\n[2] enable_thinking 参数是否被接受")
    for tag, thinking in (("不传该参数", None), ("enable_thinking=False", False), ("enable_thinking=True", True)):
        try:
            r = await _call(client=client, model=vision_model, b64=b64, mime=mime, thinking=thinking)
            s = r["score"]
            print(
                f"  {tag:<22} {r['dt']:5.1f}s  out_tok={r['out_tok']}  "
                f"命中 {s['hit']}/{s['total']}  错值={s['wrong'] or '无'}  漏项={s['missed'] or '无'}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:<22} ❌ 被拒绝：{type(e).__name__} {str(e)[:150]}")

    print("\n判读：若「不传」与「False」表现接近，就**不要**下发该参数（少一个提供方耦合点）；")
    print("      若某档被 400 拒绝，则该参数在此模型上不可用，不能写进默认配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
