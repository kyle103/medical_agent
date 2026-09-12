import os
from dotenv import load_dotenv
# 正确导入
from langsmith import Client

# 加载项目根目录 .env 文件（你的路径）
load_dotenv(dotenv_path="./.env.local", override=True)

# 🔥 修复：endpoint → api_url
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),  # 这里改对！
)

# 极简验证
try:
    # 测试连通性
    client.list_projects(limit=1)
    print("✅ LangSmith 配置成功！API 密钥有效")
except Exception as e:
    print(f"❌ LangSmith 连接失败：{str(e)}")