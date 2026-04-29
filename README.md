<div align="center">

# 🏥 MedAgent — 合规医疗问答智能体

**基于 LangGraph 多 Agent 工作流 · RAG 混合检索 · 全链路合规管控**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-2.4-teal.svg)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

</div>

---

## ⚕️ 合规声明

> **本系统不提供诊断、处方、用药调整、治疗方案等执业医师专属内容。**
> 药物相互作用与化验单异常判断的核心结论 **100% 来自结构化知识库**，LLM 仅负责意图识别、实体提取与文本润色。所有非档案管理类输出强制追加免责声明。

---

## 🧭 项目概览

MedAgent 是一个面向医疗健康领域的智能问答系统，核心设计理念是 **"AI 辅助 + 知识库决策 + 合规兜底"**——让大模型做它擅长的事（理解、提取、润色），把医疗结论的生成交给可溯源的结构化知识。

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────────┐
│              LangGraph 多 Agent 工作流                 │
│                                                      │
│  input_check → memory_load → intent_recognition      │
│       → entity_extraction → knowledge_retrieve       │
│       → plan → execute → reconcile → response_plan   │
│       → llm_generate → output_check → commit         │
│       → memory_update                                │
└──────────────────────────────────────────────────────┘
  │              │              │              │
  ▼              ▼              ▼              ▼
MainQAAgent  DrugConflict   LabReport     DrugRecord
(通用问答)    Agent(药物冲突) Agent(化验解读) Agent(用药记录)
```

### ✨ 核心特性

| 特性 | 说明 |
|:---|:---|
| 🧠 **多 Agent 协作** | LangGraph 状态图驱动，4 个专业 Agent 按意图自动路由 |
| 🔍 **混合检索** | 向量语义检索 + 模拟 BM25 关键词检索 + RRF 融合 + Rerank 精排 |
| 🛡️ **全链路合规** | 输入拦截（敏感信息/Prompt 注入/违规意图）+ 输出过滤 + 强制免责 |
| 💊 **药物冲突检测** | 基于结构化药品知识库的配伍禁忌查询，结论可溯源 |
| 🧪 **化验单解读** | 检验指标 vs 参考范围自动比对，异常标记 + 通俗解释 |
| 📋 **用药记录管理** | 状态机驱动的用药记录增删改查，支持多轮对话确认 |
| 🧩 **长期记忆** | 用户级向量记忆，跨会话保持上下文 |
| 📊 **RAG 评估框架** | 内置 RAGAS 风格评估工具，支持多策略 A/B 测试 |

---

### 意图识别策略

采用 **规则优先 + LLM 增强** 的双层架构：

| 层级 | 策略 | 触发条件 |
|:---|:---|:---|
| 第一层 | 关键词规则匹配 | 默认，零延迟 |
| 第二层 | LLM 结构化输出 | 规则置信度 < 0.75 且 LLM 已配置 |

支持的意图类型：`archive`（档案）、`drug`（药物）、`lab`（化验）、`general`（通用问答）、`multi`（多意图拆分）

---

## 🔍 RAG 检索架构

```
用户 Query
    │
    ▼
┌─────────────────────────────────────────┐
│           检索策略选择                    │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  向量检索  │    │  模拟 BM25 检索   │   │
│  │(Embedding)│    │ (jieba/LLM 分词   │   │
│  │          │    │  + Milvus LIKE)   │   │
│  └────┬─────┘    └────────┬─────────┘   │
│       │                   │             │
│       └───────┬───────────┘             │
│               ▼                         │
│        RRF 融合排序                      │
│   score = w_v·1/(k+rank_v) + w_k·1/(k+rank_k)  │
│               │                         │
│               ▼                         │
│        [可选] Rerank 精排               │
│   · replace: 完全用 rerank 分数          │
│   · merge:   加权合并原始分数            │
│               │                         │
│               ▼                         │
│        窗口扩展（上下文拼接）             │
│               │                         │
│               ▼                         │
│        去重 → Top-K 截断                │
└─────────────────────────────────────────┘
```

---

## 📊 RAG 评估框架

内置 RAGAS 风格的评估工具，支持多策略 A/B 对比测试：

```bash
# 生成评估数据集（LLM 语义匹配构建 GT）
python -m rag_comprehensive_assessment.run generate --backfill

# 纯向量检索基线
python -m rag_comprehensive_assessment.run evaluate

# 混合检索（向量 + BM25）
python -m rag_comprehensive_assessment.run evaluate --retrieval-mode hybrid --vector-weight 0.7 --keyword-weight 0.3

# 混合检索 + Rerank 精排
python -m rag_comprehensive_assessment.run evaluate --retrieval-mode hybrid --rerank --rerank-mode merge --rerank-weight 0.6

# Query Rewrite / HyDE 变换
python -m rag_comprehensive_assessment.run evaluate --query-transform rewrite
python -m rag_comprehensive_assessment.run evaluate --query-transform hyde
```

评估指标：`Recall@K` · `Precision@K` · `MRR@K` · `Coverage@K`

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Milvus 2.4+（向量数据库）
- LLM API（OpenAI 兼容接口）
- Embedding API 或本地模型

### 1. 安装依赖

```powershell
cd medical_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

必填项：

| 配置项 | 说明 |
|:---|:---|
| `LLM_API_BASE` | LLM API 地址（OpenAI 兼容） |
| `LLM_API_KEY` | API 密钥 |
| `LLM_MODEL_NAME` | 模型名称 |
| `EMBEDDING_TYPE` | `local` 或 `api` |
| `MILVUS_URI` | Milvus 连接地址 |
| `DB_TYPE` | `sqlite` 或 `mysql` |

### 3. 初始化

```powershell
# 初始化数据库
python -m app.db.init_db --env local

# 导入知识库到 Milvus
python -m scripts.import_jsonl_to_milvus --source-dir data/Source_data --collection kb_general
```

### 4. 启动服务

```powershell
# 后端 API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 前端 Demo（另开终端）
python frontend_server.py
```

访问：
- 🌐 前端界面：http://127.0.0.1:3000
- 📖 API 文档：http://127.0.0.1:8000/docs

### Docker 部署

```bash
docker-compose up -d
```

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|:---|:---|:---|
| `/api/v1/user/register` | POST | 用户注册 |
| `/api/v1/user/login` | POST | 用户登录 |
| `/api/v1/chat/` | POST | 智能对话（流式/非流式） |
| `/api/v1/drug/interaction` | POST | 药物冲突查询 |
| `/api/v1/lab/interpret` | POST | 化验单解读 |
| `/api/v1/archive/` | GET/POST | 档案管理 |

所有接口需 JWT 鉴权（注册/登录除外）。

---

## 🛡️ 合规体系

```
输入侧                              输出侧
──────                              ──────
┌─────────────────┐                ┌─────────────────┐
│ 敏感信息检测     │                │ 违规词汇过滤     │
│ (身份证/医保号)  │                │ (确诊/处方/诊断) │
├─────────────────┤                ├─────────────────┤
│ Prompt 注入检测  │                │ 免责声明追加     │
│ (越权指令拦截)   │                │ (强制附加)       │
├─────────────────┤                └─────────────────┘
│ 违规意图拦截     │
│ (开药/诊断请求)  │
└─────────────────┘
```

**核心原则**：LLM 不生成任何医疗结论。药物冲突结论来自 `drug_knowledge.csv`，化验异常判断来自 `lab_item_reference.csv`，均可溯源至结构化知识库。

---

## 🧪 测试

```powershell
python -m pytest -q
```

---

## 📦 技术栈

| 类别 | 技术 |
|:---|:---|
| Web 框架 | FastAPI + Uvicorn |
| 工作流引擎 | LangGraph |
| LLM | OpenAI 兼容 API（通义千问 / DeepSeek 等） |
| Embedding | sentence-transformers / API |
| 向量数据库 | Milvus 2.4 |
| 关系数据库 | SQLite / MySQL |
| Rerank | 阿里云 DashScope qwen3-rerank |
| 分词 | jieba |
| 可观测性 | Langfuse / LangSmith |
| 部署 | Docker + docker-compose |

---

## 📁 知识库数据

| 文件 | 说明 | 规模 |
|:---|:---|:---|
| `train_datasets.jsonl` | 通用医学问答对 | 36 万+ 条 |
| `test_datasets.jsonl` | 测试问答对 | - |
| `drug_knowledge.csv` | 药品配伍禁忌库 | 结构化 |
| `lab_item_reference.csv` | 检验指标参考范围 | 结构化 |

---

## ⚖️ 免责声明

免责声明：本内容仅为通用健康科普参考，不构成任何医疗诊断、用药建议，身体不适请及时前往正规医疗机构就诊，遵医嘱治疗。
