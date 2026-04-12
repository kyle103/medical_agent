# //测试LLM连通性，可删除。
import asyncio

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)


async def main() -> None:
    # 1) 输出关键配置（不要输出密钥）
    logger.info("LLM_API_BASE=%s", settings.LLM_API_BASE)
    logger.info("LLM_MODEL_NAME=%s", settings.LLM_MODEL_NAME)
    logger.info("LLM_TEMPERATURE=%s LLM_MAX_TOKENS=%s", settings.LLM_TEMPERATURE, settings.LLM_MAX_TOKENS)

    # 2) 做一次最小调用
    llm = LLMService()
    resp = await llm.chat_completion(
        prompt="请只回复：ok",
        system_prompt="你是一个测试助手，只能输出 ok。",
        stream=False,
    )

    # 兼容 openai>=1.x 的返回结构
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)

    print("LLM_HEALTHCHECK_RESPONSE:", content)


if __name__ == "__main__":
    asyncio.run(main())