"""化验单图像识别路线对比实验：视觉大模型直出结构化 vs OCR 文本流 + 既有解析器。

目的：为「加多模态能力」这一决策提供**实测证据**，而不是靠偏好选型。

三条路线，同图同题：
  A. `qwen3-vl-flash` + `json_schema`  → 直出 `[{item_name, test_value, unit, reference_range}]`
  B. `qwen-vl-ocr`      → 纯文本 → 复用既有 `lab_item_parser.parse_lab_items`
  C. `qwen3-vl-flash`   纯自然语言提问（不给 schema）→ 观察输出形态是否可程序化消费

图像两种条件：
  ① 清晰 PNG（数字打印体，模拟电子版/截图）
  ② JPEG q55 + 旋转 1.5°（模拟手机翻拍：压缩伪影 + 轻微倾斜）

评分维度：**逐项数值是否与真值一致**（数值错一位在医疗场景就是明确危害），
另记：漏项 / 多项 / 单位是否正确 / 参考范围是否抄对 / 墙钟耗时。

用法：
    python scripts/probe_lab_vision.py
    python scripts/probe_lab_vision.py qwen3-vl-plus        # 换视觉模型
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import settings  # noqa: E402
from app.core.tools.lab_item_parser import parse_lab_items  # noqa: E402

#: 真值：一张血常规（含 H/N/L 三种形态，含小数与百分比）
GROUND_TRUTH: dict[str, str] = {
    "白细胞计数": "11.2",
    "红细胞计数": "4.51",
    "血红蛋白": "128",
    "血小板计数": "210",
    "中性粒细胞百分比": "78.5",
}

_ROWS = [
    ("白细胞计数", "WBC", "11.2", "10^9/L", "3.5-9.5"),
    ("红细胞计数", "RBC", "4.51", "10^12/L", "4.3-5.8"),
    ("血红蛋白", "HGB", "128", "g/L", "130-175"),
    ("血小板计数", "PLT", "210", "10^9/L", "125-350"),
    ("中性粒细胞百分比", "NEUT%", "78.5", "%", "40-75"),
]

_HEADER = ("示例市第一人民医院", "血常规检验报告", "姓名：张三    性别：男    年龄：35 岁    采样日期：2026-09-10")

#: 结构化输出 schema：字段名与既有 `lab_items` 完全一致，便于直接接下游
_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string"},
                    "test_value": {"type": "string"},
                    "unit": {"type": "string"},
                    "reference_range": {"type": "string"},
                },
                "required": ["item_name", "test_value", "unit", "reference_range"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_PROMPT = (
    "这是一张化验单图片。请逐行提取全部检验项目，输出结构化结果。\n"
    "要求：\n"
    "1) `item_name` 用中文项目名（若图中是缩写如 WBC，可作为补充写在中文名后）；\n"
    "2) `test_value` 只填数值本身，不要带单位、不要带箭头标记；\n"
    "3) `unit` 填单位，`reference_range` 填参考范围，**严格照抄图中内容**，不要自行换算或推断；\n"
    "4) 看不清的字段留空字符串，**不要猜测**。"
)


def _font(size: int):
    from PIL import ImageFont

    for name in ("msyh.ttc", "simhei.ttf", "Deng.ttf"):
        p = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def build_image(*, degraded: bool):
    """生成一张化验单图。`degraded=True` 追加 JPEG 压缩 + 轻微旋转，模拟翻拍。"""
    from PIL import Image, ImageDraw

    w, h = 1000, 620
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((40, 28), _HEADER[0], font=_font(26), fill="black")
    d.text((40, 68), _HEADER[1], font=_font(22), fill="black")
    d.text((40, 104), _HEADER[2], font=_font(15), fill="#333333")
    d.line([(40, 132), (w - 40, 132)], fill="#888888", width=2)

    cols = [40, 340, 470, 620, 800]
    heads = ["项目", "结果", "单位", "参考范围"]
    for x, t in zip(cols, heads):
        d.text((x, 150), t, font=_font(17), fill="black")
    d.line([(40, 178), (w - 40, 178)], fill="#bbbbbb", width=1)

    y = 192
    for name, en, val, unit, ref in _ROWS:
        d.text((cols[0], y), name, font=_font(17), fill="black")
        d.text((cols[0] + 150, y), en, font=_font(13), fill="#666666")
        d.text((cols[1], y), val, font=_font(17), fill="black")
        d.text((cols[2], y), unit, font=_font(17), fill="black")
        d.text((cols[3], y), ref, font=_font(17), fill="black")
        y += 42

    d.text((40, y + 16), "本报告仅示例用途，不作为临床诊断依据。", font=_font(13), fill="#777777")

    buf = io.BytesIO()
    if degraded:
        img = img.rotate(1.5, expand=True, fillcolor="white")
        img.save(buf, format="JPEG", quality=55)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


async def _call_vision(*, model: str, b64: str, mime: str, prompt: str, schema: dict | None) -> tuple[str, float]:
    """直连 provider 的多模态调用（不改项目代码，仅做实验）。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE)
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1500}
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "lab_report", "strict": True, "schema": schema},
        }
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(**kwargs)
    dt = time.perf_counter() - t0
    return (resp.choices[0].message.content or ""), dt


def _score(items: list[dict]) -> dict:
    """与真值逐项比对。"""
    got = {}
    for it in items:
        name = str(it.get("item_name", "") or "")
        val = str(it.get("test_value", "") or "").strip()
        for key in GROUND_TRUTH:
            if key in name:
                got[key] = val
                break
    hit = sum(1 for k, v in GROUND_TRUTH.items() if got.get(k, "").strip() == v)
    wrong = [k for k, v in GROUND_TRUTH.items() if k in got and got[k].strip() != v]
    missed = [k for k in GROUND_TRUTH if k not in got]
    extra = [k for k in got if k not in GROUND_TRUTH]
    return {
        "hit": hit, "total": len(GROUND_TRUTH),
        "wrong": wrong, "missed": missed, "extra": extra,
        "got": got,
    }


async def _dump_image(b64: str, mime: str, tag: str) -> None:
    out = _ROOT / "scripts" / "_out"
    out.mkdir(exist_ok=True)
    ext = "jpg" if "jpeg" in mime else "png"
    (out / f"lab_report_{tag}.{ext}").write_bytes(base64.b64decode(b64))
    print(f"  [图] scripts/_out/lab_report_{tag}.{ext}")


async def main() -> int:
    vision_model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl-flash"
    ocr_model = "qwen-vl-ocr"

    print("=" * 92)
    print(f"化验单图像识别对比 | 视觉模型={vision_model} | OCR 模型={ocr_model}")
    print("=" * 92)

    conditions = [("clear", False), ("photo", True)]
    images = {}
    for tag, degraded in conditions:
        b64, mime = build_image(degraded=degraded)
        images[tag] = (b64, mime)
        print(f"\n[{tag}] {'清晰 PNG' if not degraded else 'JPEG q55 + 旋转 1.5°（模拟翻拍）'}")
        await _dump_image(b64, mime, tag)

    summary: list[str] = []

    for tag, _ in conditions:
        b64, mime = images[tag]
        print("\n" + "-" * 92)
        print(f"图像条件：{tag}")
        print("-" * 92)

        # ---- A. 视觉模型 + json_schema ----
        try:
            raw, dt = await _call_vision(model=vision_model, b64=b64, mime=mime, prompt=_PROMPT, schema=_SCHEMA)
            parsed, note = None, ""
            try:
                parsed = json.loads(raw)
            except Exception as e:  # noqa: BLE001
                note = f"JSON 解析失败：{type(e).__name__}"
            if parsed is not None:
                items = parsed.get("items") if isinstance(parsed, dict) else None
                if not isinstance(items, list):
                    note = "schema 未被遵守：缺少 items 数组"
                    items = []
                keys_ok = all(
                    set(it.keys()) == {"item_name", "test_value", "unit", "reference_range"}
                    for it in items
                ) if items else False
                s = _score(items)
                print(f"  A. 视觉模型 + json_schema   {dt:.1f}s  字段集合严格={keys_ok}")
                print(f"     命中 {s['hit']}/{s['total']}  错值={s['wrong'] or '无'}  漏项={s['missed'] or '无'}  多项={s['extra'] or '无'}")
                print(f"     抽出：{s['got']}")
                if note:
                    print(f"     ⚠ {note}")
                summary.append(f"{tag} | A 视觉+schema | {dt:.1f}s | {s['hit']}/{s['total']} | 错={len(s['wrong'])} 漏={len(s['missed'])}")
            else:
                print(f"  A. 视觉模型 + json_schema   {dt:.1f}s  ⚠ {note}")
                print(f"     原始返回前 200 字：{raw[:200]!r}")
                summary.append(f"{tag} | A 视觉+schema | {dt:.1f}s | 解析失败")
        except Exception as e:  # noqa: BLE001
            print(f"  A. 视觉模型 + json_schema   调用失败：{type(e).__name__} {str(e)[:160]}")
            summary.append(f"{tag} | A 视觉+schema | 调用失败")

        # ---- B. OCR 模型 → 文本 → 既有解析器 ----
        try:
            raw, dt = await _call_vision(model=ocr_model, b64=b64, mime=mime, prompt="请输出图中全部文字内容。", schema=None)
            items = parse_lab_items(raw)
            s = _score(items)
            print(f"  B. OCR 模型 → 既有解析器     {dt:.1f}s")
            print(f"     命中 {s['hit']}/{s['total']}  错值={s['wrong'] or '无'}  漏项={s['missed'] or '无'}  多项={s['extra'] or '无'}")
            print(f"     抽出：{s['got']}")
            print(f"     OCR 文本前 180 字：{raw[:180]!r}")
            summary.append(f"{tag} | B OCR+解析器  | {dt:.1f}s | {s['hit']}/{s['total']} | 错={len(s['wrong'])} 漏={len(s['missed'])}")
        except Exception as e:  # noqa: BLE001
            print(f"  B. OCR 模型 → 既有解析器     调用失败：{type(e).__name__} {str(e)[:160]}")
            summary.append(f"{tag} | B OCR+解析器  | 调用失败")

        # ---- C. 视觉模型纯自然语言（不给 schema）----
        try:
            raw, dt = await _call_vision(model=vision_model, b64=b64, mime=mime, prompt=_PROMPT, schema=None)
            print(f"  C. 视觉模型 无 schema        {dt:.1f}s")
            print(f"     返回形态：{raw[:150]!r}")
            summary.append(f"{tag} | C 视觉无schema | {dt:.1f}s | 形态见上")
        except Exception as e:  # noqa: BLE001
            print(f"  C. 视觉模型 无 schema        调用失败：{type(e).__name__} {str(e)[:160]}")

    print("\n" + "=" * 92)
    print("汇总")
    print("=" * 92)
    for line in summary:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
