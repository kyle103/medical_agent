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

2) 配置环境变量

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

3) 初始化数据库与导入最小知识库

```powershell
python -m app.db.init_db --env local
```

4) 启动服务

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
前端：
python frontend_server.py

访问：

- Swagger： http://127.0.0.1:8000/docs


## 2. 测试

# 进入项目目录并激活 venv
Set-Location .\medical_agent
.\.venv\Scripts\Activate.ps1


# 用 venv 解释器运行 pytest
python -m pytest -q


## 3. 免责声明

免责声明：本内容仅为通用健康科普参考，不构成任何医疗诊断、用药建议，身体不适请及时前往正规医疗机构就诊，遵医嘱治疗。
