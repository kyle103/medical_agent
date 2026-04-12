# 医疗问答 Agent（合规版 MVP）

本项目严格遵循《医疗问答Agent项目完整开发文档 V1.0》：

- **不提供诊断/处方/用药调整/治疗方案**等执业医师专属内容。
- 药物相互作用与化验单异常判断的**核心结论仅来自结构化知识库/参考库**，LLM 仅用于**意图识别、实体提取与文本润色**。
- 所有非档案管理类输出将强制追加统一免责声明。

## 1. 本地启动（Windows / PowerShell）

1) 创建虚拟环境并安装依赖

```powershell
Set-Location .\medical_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

<!-- 
conda deactivate
.\.venv\Scripts\Activate.ps1
 -->

2) 配置环境变量

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

> 注意：`.env.local` 内所有 `{{...}}` 均为**待用户手动填充**占位符，代码不会写入任何可用默认密钥/地址。

3) 初始化数据库与导入最小知识库

```powershell
python -m app.db.init_db --env local
```

4) 启动服务

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问：

- Swagger： http://127.0.0.1:8000/docs

eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJlNTkyZmIwNDE0NGE0MjNmOWRkOTIzM2Q5NWVmNWFhMSIsImV4cCI6MTc3NTkxODMyM30.myldW2vQVBOUcjPQ2qXN-Ns4Z72kLw36XEstZ8qsA4s

## 2. 测试

# 进入项目目录并激活 venv
Set-Location .\medical_agent
.\.venv\Scripts\Activate.ps1

# 关键：仅当前会话净化 PATH（移除 Anaconda）
$env:Path = ($env:Path -split ';' | Where-Object { $_ -and ($_ -notmatch '(?i)\\anaconda') }) -join ';'
$env:Path = (Resolve-Path .\.venv\Scripts).Path + ';' + $env:Path

# 验证 urllib 不再来自 C:\anaconda
.\.venv\Scripts\python.exe -c "import urllib,urllib.request as r; print(urllib.__file__); print(r.__file__)"

# 用 venv 解释器运行 pytest（不要直接 pytest）
.\.venv\Scripts\python.exe -m pytest -q

```powershell
pytest -q
```

## 3. 免责声明（统一文本）

免责声明：本内容仅为通用健康科普参考，不构成任何医疗诊断、用药建议，身体不适请及时前往正规医疗机构就诊，遵医嘱治疗。
