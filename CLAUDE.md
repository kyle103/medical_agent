# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Run backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run frontend demo (separate terminal)
python frontend_server.py

# Start both services (Windows PowerShell)
.\run_project.ps1

# Run all tests
python -m pytest -q

# Run a single test file
python -m pytest -q tests/test_api.py

# Initialize database and import knowledge base
python -m app.db.init_db --env local
python -m scripts.import_jsonl_to_milvus --source-dir data/Source_data --collection kb_general

# RAG evaluation (separate package)
python -m rag_comprehensive_assessment.run generate --backfill
python -m rag_comprehensive_assessment.run evaluate
python -m rag_comprehensive_assessment.run evaluate --retrieval-mode hybrid --rerank
```

## Architecture

### LangGraph Workflow (the core)

The entire system is a **LangGraph StateGraph** pipeline defined in [app/core/agent/workflow.py](app/core/agent/workflow.py). `MedicalAgent` compiles a graph with these nodes in order:

```
input_check → memory_load → intent_recognition → entity_extraction
→ knowledge_retrieve → plan → execute → reconcile → response_plan
→ llm_generate → output_check → commit → memory_update
```

- `plan` generates an execution plan (which agent/tool to call). If the result is insufficient, `execute` can loop back to `plan` (max 2 replans).
- The graph has two error-check conditional edges that can short-circuit to `error_finalize`.
- State is a flat `TypedDict` (`AgentState`) with ~40 fields — see [app/core/agent/state.py](app/core/agent/state.py).
- Node implementations are in [app/core/agent/nodes.py](app/core/agent/nodes.py) (~57KB, the largest single file).

### Intent Recognition

Two-layer strategy in [app/core/agent/intent_classifier.py](app/core/agent/intent_classifier.py):

1. **LLM first** — delegates to `LLMDecisionService` for structured intent classification with agent/tool descriptions exposed to the model.
2. **Rule fallback** — keyword-based regex if LLM is unavailable or fails.

Supported intents: `archive`, `drug`, `lab`, `general`, `multi` (multi-intent decomposition via `QueryOrchestrator`).

### Agent/Tool Routing

`PlannerAgent` in [app/core/agent/planner_agent.py](app/core/agent/planner_agent.py) determines the execution target:

| Target | Type | Purpose |
|:---|:---|:---|
| `MainQAAgent` | Agent | General QA + archive queries |
| `DrugRecordAgent` | Agent | Medication record CRUD (state-machine driven) |
| `drug_interaction` | Tool | Drug-drug conflict lookup from structured KB |
| `lab_report` | Tool | Lab result interpretation against reference ranges |

### RAG Pipeline

Dual-retrieval with RRF fusion in [app/core/rag/public_kb_service.py](app/core/rag/public_kb_service.py) and [rag_comprehensive_assessment/](rag_comprehensive_assessment/):

1. **Vector search** — embedding-based via Milvus
2. **Simulated BM25** — jieba keyword extraction + Milvus `LIKE` patterns
3. **RRF fusion** — weighted reciprocal rank fusion
4. **Optional rerank** — DashScope qwen3-rerank with `replace` or `merge` modes
5. **Window expansion** — context window stitching around hits
6. **Dedup + Top-K truncation**

Configuration keys: `PUBLIC_KB_TOP_K`, `PUBLIC_KB_BM25_TOP_K`, `PUBLIC_KB_RRF_K`, `PUBLIC_KB_EXPAND_WINDOW`.

### Compliance System

In [app/core/compliance/compliance_service.py](app/core/compliance/compliance_service.py) with rules in [app/config/compliance_rules.py](app/config/compliance_rules.py):

- **Input side**: sensitive PII detection (ID card, insurance number), prompt injection detection, banned intent keyword blocking (e.g., "帮我开药").
- **Output side**: regex scanning for forbidden patterns (确诊, 开处方, 制定治疗方案, etc.).
- **Mandatory disclaimer** appended to all non-archive responses when `FORCE_DISCLAIMER=true`.

Toggle via `ENABLE_INPUT_CHECK`, `ENABLE_OUTPUT_CHECK`, `FORCE_DISCLAIMER` env vars.

### Long-Term Memory

`LongMemoryService` in [app/core/memory/long_memory_service.py](app/core/memory/long_memory_service.py) stores user facts in a dedicated Milvus collection (`user_long_memory`). Uses jieba-based chunking with overlap. LLM extracts structured memories from conversation history. Supports batch write on session end via `/api/v1/chat/session/end`.

### Drug Record State Machine

`DrugRecordStateMachine` in [app/core/skills/drug_record_state_machine.py](app/core/skills/drug_record_state_machine.py) manages multi-turn medication record operations with phases: `IDLE → COLLECTING → CONFIRMING → COMMITTING`. `MedicationConfirmationSkill` builds confirmation messages when entities are ambiguous.

### Key Dependencies

- **LLM**: OpenAI-compatible API (configured via `LLM_API_BASE`/`LLM_API_KEY`/`LLM_MODEL_NAME`)
- **Embedding**: sentence-transformers (local) or API (configured via `EMBEDDING_TYPE`)
- **Vector DB**: Milvus 2.4 (primary, via pymilvus)
- **Relational DB**: SQLite (default) or MySQL (SQLAlchemy async)
- **Observability**: Langfuse (production) or LangSmith (dev)
- **Workflow**: LangGraph 0.2

### Configuration

Settings in [app/config/settings.py](app/config/settings.py) load from `.env.local` (or `.env.prod` when `APP_ENV=prod`). Uses `pydantic-settings`. Template values like `{{LLM_API地址}}` indicate unconfigured placeholders — the app gates LLM-dependent features by checking whether config values still contain `{{...}}` patterns.

### Frontend

Single-page HTML app at [frontend/index.html](frontend/index.html) served by a minimal Python HTTP server (`frontend_server.py` on port 3000). All API calls go to the FastAPI backend.
