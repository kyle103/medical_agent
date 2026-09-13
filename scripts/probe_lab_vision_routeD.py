"""化验单识别：路线 D ——「OCR 文本 + 文本大模型」对照组。

为什么要单独跑这一段：
  第一轮实验里路线 B（`qwen-vl-ocr` → `parse_lab_items`）拿到 0/5，但那是"OCR + 纯正则解析器"。
  真实讨论「本地 OCR」时，方案形态必然是 **本地 OCR + 现有文本大模型**：
  OCR 负责识字，模型负责把二维表格还原成「项目 → 数值」的对应关系。
  所以必须补测 D，否则 B 的 0/5 会被认为是"没用 LLM 才失败"，而不是"OCR 路线的结构缺陷"。

路线 D 的定位是**本地 OCR 的乐观上界**：
  OCR 文本质量与 `qwen-vl-ocr` 相同（本地 RapidOCR/PaddleOCR 通常更差、阅读顺序更乱），
  后面还额外给了一次文本大模型的修复机会。若 D 仍不如路线 A，则"本地 OCR"的结论无可辩驳。

顺带加测**重退化**条件（降采样 + 模糊 + 低质量 JPEG），看视觉模型的鲁棒边界在哪。

用法：
    python scripts/probe_lab_vision_routeD.py
"""

from __future__ import annotations

import asyncio
import base64
import io
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
from probe_lab_vision import (  # noqa: E402  # 复用第一轮脚本的造图/评分/真值
    _PROMPT,
    _SCHEMA,
    _call_vision,
    _dump_image,
    _score,
    build_image,
)


def build_hard_image() -> tuple[str, str]:
    """重退化：降采样到 620px + 高斯模糊 + JPEG q40（模拟远距离翻拍 / 手抖）。"""
    from PIL import Image, ImageFilter

    b64, mime = build_image(degraded=False)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = img.size
    img = img.resize((620, int(h * 620 / w)), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


async def _call_text_llm(*, prompt: str, schema: dict) -> tuple[str, float]:
    """用项目当前文本模型 + 严格 schema 做抽取（等价于路线 D 的第二步）。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE)
    kwargs: dict = {
        "model": settings.LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "lab_report", "strict": True, "schema": schema},
        },
    }
    if settings.LLM_DISABLE_THINKING:
        kwargs["extra_body"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(**kwargs)
    dt = time.perf_counter() - t0
    return (resp.choices[0].message.content or ""), dt


def _parse_items(raw: str) -> tuple[list[dict], str]:
    """返回 (items, 备注)。"""
    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return [], f"JSON 解析失败：{type(e).__name__}"
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], "schema 未被遵守：缺少 items 数组"
    return items, ""


async def main() -> int:
    vision_model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl-flash"
    ocr_model = "qwen-vl-ocr"

    print("=" * 92)
    print(f"D 组对照：OCR 文本 + 文本大模型（{settings.LLM_MODEL_NAME}）  |  视觉模型={vision_model}")
    print("=" * 92)

    conditions: list[tuple[str, tuple[str, str]]] = [
        ("clear", build_image(degraded=False)),
        ("photo", build_image(degraded=True)),
        ("hard", build_hard_image()),
    ]
    for tag, (b64, mime) in conditions:
        print(f"\n[{tag}]")
        await _dump_image(b64, mime, tag)

    summary: list[str] = []

    for tag, (b64, mime) in conditions:
        print("\n" + "-" * 92)
        print(f"图像条件：{tag}")
        print("-" * 92)

        # ---- A. 视觉模型 + json_schema（同第一轮，作为基准）----
        try:
            raw, dt_a = await _call_vision(model=vision_model, b64=b64, mime=mime, prompt=_PROMPT, schema=_SCHEMA)
            items, note = _parse_items(raw)
            s = _score(items)
            print(f"  A. 视觉模型 + schema            {dt_a:.1f}s")
            print(f"     命中 {s['hit']}/{s['total']}  错值={s['wrong'] or '无'}  漏项={s['missed'] or '无'}")
            print(f"     抽出：{s['got']}")
            if note:
                print(f"     ⚠ {note}")
            summary.append(f"{tag} | A 视觉+schema      | {dt_a:5.1f}s | {s['hit']}/{s['total']} | 错={len(s['wrong'])} 漏={len(s['missed'])}")
        except Exception as e:  # noqa: BLE001
            print(f"  A. 视觉模型 + schema            调用失败：{type(e).__name__} {str(e)[:140]}")
            summary.append(f"{tag} | A 视觉+schema      | 调用失败")

        # ---- D. OCR → 文本大模型 → 同 schema ----
        try:
            ocr_text, dt_ocr = await _call_vision(
                model=ocr_model, b64=b64, mime=mime, prompt="请输出图中全部文字内容。", schema=None
            )
            prompt_d = (
                "下面是从一张化验单图片 OCR 出来的原始文本。它按行输出，"
                "**行与行之间可能属于同一行的不同列**（例如「项目 / 结果 / 单位 / 参考范围」四列被拆成四行）。\n"
                "请还原表格结构，提取全部检验项目，输出结构化结果。\n"
                "要求：① `item_name` 用中文项目名；② `test_value` 只填数值，不带单位和箭头；\n"
                "③ `unit` 与 `reference_range` 严格照抄原文，不要换算或推断；④ 无法确定的字段留空字符串，不要猜。\n\n"
                f"--- OCR 文本开始 ---\n{ocr_text}\n--- OCR 文本结束 ---"
            )
            raw, dt_llm = await _call_text_llm(prompt=prompt_d, schema=_SCHEMA)
            items, note = _parse_items(raw)
            s = _score(items)
            dt_total = dt_ocr + dt_llm
            print(f"  D. OCR({dt_ocr:.1f}s) + 文本模型({dt_llm:.1f}s)  合计 {dt_total:.1f}s")
            print(f"     命中 {s['hit']}/{s['total']}  错值={s['wrong'] or '无'}  漏项={s['missed'] or '无'}")
            print(f"     抽出：{s['got']}")
            if note:
                print(f"     ⚠ {note}")
            summary.append(f"{tag} | D OCR+文本模型    | {dt_total:5.1f}s | {s['hit']}/{s['total']} | 错={len(s['wrong'])} 漏={len(s['missed'])}")
        except Exception as e:  # noqa: BLE001
            print(f"  D. OCR + 文本模型               调用失败：{type(e).__name__} {str(e)[:140]}")
            summary.append(f"{tag} | D OCR+文本模型    | 调用失败")

    print("\n" + "=" * 92)
    print("汇总：A（视觉一步到位） vs D（OCR + 文本模型）")
    print("=" * 92)
    for line in summary:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
