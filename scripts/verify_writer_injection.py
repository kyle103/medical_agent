"""验证改造后的 llm_generate 是否真的被注入了 writer。

这是 Step 3 唯一的"静默失败"风险点：langgraph 按「参数名 == writer 且注解
== StreamWriter」注入，名字/注解不对**不会报错**，只会静默退化。
另外还要确认 writer 的默认兜底（不经图直接调用时不崩）。
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.utils.runnable import KWARGS_CONFIG_KEYS  # noqa: E402

from app.core.agent import nodes  # noqa: E402
from app.core.agent.workflow import MedicalAgent  # noqa: E402


def main() -> int:
    ok = True

    # 1) 签名自检：langgraph 认不认这个参数
    sig = inspect.signature(nodes.llm_generate)
    p = sig.parameters.get("writer")
    print(f"[sig] writer param: {p!r}")
    print(f"[sig] annotation    : {p.annotation!r}  (from __future__ annotations -> 字符串)")

    kw, typ, ck, defv = KWARGS_CONFIG_KEYS[0]
    accepts = p is not None and p.annotation in typ and p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
    print(f"[sig] langgraph 判定可注入 writer: {accepts}")
    ok &= accepts

    # 2) 图构建后，节点是否被包成 RunnableCallable 且 func_accepts['writer'] 为真
    agent = MedicalAgent()
    node = agent.graph.nodes.get("llm")
    inner = getattr(node, "bound", None) or getattr(node, "func", None)
    accepts_flag = getattr(inner, "func_accepts", {})
    print(f"[graph] llm node inner: {type(inner).__name__}  func_accepts={accepts_flag}")
    ok &= bool(accepts_flag.get("writer"))

    # 3) 静态检查：llm_generate 里不残留旧的非流式调用
    src = inspect.getsource(nodes.llm_generate)
    no_blocking = "chat_completion(" not in src
    uses_stream = "chat_completion_stream" in src
    print(f"[src] 已无 chat_completion( 阻塞调用: {no_blocking}; 使用 stream: {uses_stream}")
    ok &= no_blocking and uses_stream

    # 4) 直接调用（无 writer）必须走兜底 no-op 而不是 ValueError
    print("[direct] llm_generate(state) 直接调用走兜底 -> ", end="")
    try:
        # 构造一个必然走 final_response 短路分支的最小 state，避免真调 LLM
        st = {"final_response": "你好。", "intent": "general", "user_input": "你好"}
        out = asyncio.run(nodes.llm_generate(st))
        print(f"OK, llm_output={out.get('llm_output')!r}")
        ok &= out.get("llm_output") == "你好。"
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {type(e).__name__}: {e}")
        ok = False

    print("\n=== 结论 ===")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
