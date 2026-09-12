# test_langfuse.py
import os
from dotenv import load_dotenv

# 🔥 关键：加载根目录的 .env 文件（修复密钥读取失败）
load_dotenv(dotenv_path="./.env.local", override=True)

# 导入 Langfuse
try:
    from langfuse import Langfuse
except ImportError:
    print("❌ 请先安装：pip install langfuse python-dotenv")
    exit()

# 初始化客户端
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# 🔥 关键：新版SDK正确验证方式（修复方法不存在）
if langfuse.enabled:
    print("✅ Langfuse 配置成功！客户端已启用，可正常上报日志")
    print(f"Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:10]}...")
else:
    print("❌ Langfuse 客户端禁用，请检查 .env 配置")