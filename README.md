# linebase

USPTO 商标 **图形线稿 ↔ 实拍图** 自动匹配与裁切流水线。本地 web app：上传 xlsx → 抽样跑 → 浏览器审查 → 下载结果 xlsx + 裁好的图片。

## Status

dev-loop 调优阶段已基本收敛。当前默认模型：**Doubao Seed 2.0 Pro** (via Volcengine Ark)，10-fixture bench 上选择准确率 71% / 0 失败 (`research/lite-benchmark-4way.md`)。`gpt-5.5` (via 1m1ng 中转) 仍保留为 <28 px USPTO 缩略图的回落模型。prompt 在 `prompts/v_*.md` 下持续迭代。

## Project layout

```
src/linebase/      Python 后端（FastAPI + SQLite + 多 provider LLM client + SSE）
frontend/          Vite + React + TS 前端 SPA（Liquid Glass UI，6 页）
prompts/           prompt 版本（按 v_<n>.md 排序）
eval/              每轮评测产物（HTML 报告 + metrics.json + raw_log.json）
fixtures/          docx 10 个样本（ground truth，红框人工标注）
scripts/           probe_llm / baseline_eval / eval_runner / benchmark_models / e2e_real_xlsx
research/          每轮研究笔记（lite-benchmark-4way、user-model-picker 等）
.trellis/spec/     架构 / gotcha / multi-provider 等稳定知识
.data/             运行时（上传/缓存/数据库），gitignored
```

## Setup

```powershell
# Python backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Frontend
cd frontend
pnpm install
```

`.env` 已存在（不入 git，见 `.gitignore`）。三段 provider 配置：

```bash
# 必需 — OpenAI 兼容主 provider（gpt-* / claude-* prefix 全走这里）
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.1m1ng.net/v1
OPENAI_MODEL=gpt-5.5
OPENAI_REVIEW_MODEL=gpt-5.5

# 当前默认（覆盖 OPENAI_MODEL；pipeline 主路调用它）
LINEBASE_DEFAULT_MODEL=doubao-seed-2-0-pro-260215
LINEBASE_REVIEW_MODEL=doubao-seed-2-0-pro-260215

# 可选 — Volcengine Ark（doubao-* 走这里）
ARK_API_KEY=ark-...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 可选 — SiliconFlow（Qwen/* zai-org/* moonshotai/* THUDM/* deepseek-ai/* 走这里）
SILICONFLOW_API_KEY=sk-...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

只有 OpenAI 块是启动硬要求；其余两个 provider 缺 key 不报错，只是路由到对应模型时会抛清晰的 RuntimeError。详见 `.trellis/spec/backend/multi-provider.md`。

## Run

### Dev mode（推荐）

终端 1 — 后端：
```powershell
.venv\Scripts\python.exe -m linebase.cli serve --reload
```

终端 2 — 前端：
```powershell
cd frontend
pnpm dev
```

浏览器开 http://localhost:5173 （Vite dev 自动代理 `/api/*` → :8000）。

### 生产模式（单进程）

```powershell
cd frontend && pnpm build  # 产物输出到 src/linebase/static/
cd ..
.venv\Scripts\python.exe -m linebase.cli serve
```

打开 http://localhost:8000 。

## Models & providers

后端通过 `Settings.resolve_provider(model_id)` 按 prefix 路由：`doubao-*` → Ark，`Qwen/* zai-org/* moonshotai/*` 等 → SiliconFlow，`gpt-* claude-*` → OpenAI 兼容主块。完整 prefix 表与新增第 4 个 provider 的步骤见 `.trellis/spec/backend/multi-provider.md`。

UI 上每个 job 都能单选一个模型（Configure 页 → 模型 dropdown），whitelist 来自 `src/linebase/models_catalog.py`：

| Model id | Provider | 备注 |
|---|---|---|
| `doubao-seed-2-0-pro-260215` | ark | **默认 · 最准**（thinking model；71% sel-acc bench winner） |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | siliconflow | 快 · 小图自动回落到 gpt-5.5 |
| `Qwen/Qwen3-VL-32B-Instruct` | siliconflow | dense |
| `doubao-seed-2-0-mini-260428` | ark | 国产·快；偶发 150 s 超时 |
| `zai-org/GLM-4.5V` | siliconflow | 输出包 `<\|begin_of_box\|>...<\|end_of_box\|>` 已处理 |
| `gpt-5.5` | openai (1m1ng 中转) | 回落 / 验证用；bbox 精度一般 |

切换默认模型的两种方式：

- **UI 一次性**：Configure 页 → 模型 dropdown 选一个；只影响这一个 job。
- **系统级**：编辑 `.env` 的 `LINEBASE_DEFAULT_MODEL` 重启后端。

自定义 model id 也支持（dropdown 选「自定义…」后手输入），只要它的 prefix 落在 `_PROVIDER_PREFIXES` 里就能路由。

## Benchmark snapshot

10-fixture · prompt v_2 · verify-loop OFF · selection accuracy 分母 7（10 个样本里只有 7 个有人工红框）。完整表格、per-sample 对比与 28-px 缩略图问题分析见 `research/lite-benchmark-4way.md`。

| model | provider | sel_acc | failed/total | mean_latency_s | cost (real, adjusted) |
|---|---|---:|---:|---:|---:|
| `doubao-seed-2-0-pro-260215` | ark | **71%** (5/7) | **0/63** | 23.32 | $0.024 |
| `doubao-seed-2-0-mini-260428` | ark | 57% (4/7) | 1/63 | 9.65 | $0.024 |
| `gpt-5.4` | openai (1m1ng) | 43% (3/7) | 0/63 | n/a | $0.348 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | siliconflow | 29% (2/7) | 6/63 | 5.73 | $0.007 |

`cost_usd` 在 UI / SQLite 上已校准过 provider factor（openai = 1.0, ark/siliconflow = 0.02），所以一个 10-row job 的真实美元开销大致就是页面显示的数字，不再被高估 30-100×。

## Dev loop（评测 + prompt 迭代）

```powershell
# 单次 probe（最小调用，确认 LLM 可达）
.venv\Scripts\python.exe scripts\probe_llm.py

# 跑当前最新 prompt 版本（prompts/v_*.md 中最新的一个）对 10 个 docx 样本
.venv\Scripts\python.exe scripts\eval_runner.py

# 跑指定版本
$env:LINEBASE_PROMPT_VERSION = "1"
.venv\Scripts\python.exe scripts\eval_runner.py

# 多模型横评（用法见 scripts/benchmark_models.py 顶部）
.venv\Scripts\python.exe scripts\benchmark_models.py
```

每次评测产物落在 `eval/run_NNN_v<ver>/`：`report.html`（人眼对比图）、`metrics.json`（量化指标，含 selection_accuracy / bbox_iou_mean / mean_latency_s 等）、`raw_log.json`（每次 LLM 调用细节）。SQLite `eval_run` 表存 metrics 摘要 ，`/dev` 页面读它展示历史。

## Web 工作流

1. **上传**：拖入 xlsx → 后端解析 sheet 列表 + 前 5 行预览。
2. **配置**：选 sheet + 列（自动嗅 URL 列）+ 样本规模 + 置信阈值 + 模型。
3. **运行**：点 "开始处理"，SSE 流式回传每行进度。
4. **审查**：每行 OK / BAD / NEEDS_REVIEW；批量重跑非 OK 行。回落到 gpt-5.5 的行会显示 "回落 gpt-5.5" 小药丸。
5. **下载**：结果 xlsx（含嵌入的最佳裁切图）+ 图片 ZIP（按申请号命名 `<appno>_<idx>.<ext>`）。

## Known gotchas

每个坑都已在 `.trellis/spec/backend/llm-gotchas.md` 写清楚 (root cause + workaround)，常见有：

- Qwen3-VL 拒绝 <28 px 的图（USPTO 部分缩略图触发）→ 自动回落 gpt-5.5。
- GLM-4.5V 输出包 `<|begin_of_box|>...<|end_of_box|>` 已 strip。
- Kimi-K2.x 和 Doubao mini 偶发 50-150 s 延迟 → bench timeout 默认 120 s。
- USPTO TSDR 对 `linebase/0.1` UA 返回 403 → fetch 已 pin Chrome UA。
- SSE 在 uvicorn 上用 `\r\n\r\n` 分隔，不是 `\n\n`。
- FastAPI 的 sync-def 路由不能调 `asyncio.get_event_loop()`（anyio 工作线程没有 loop）。

multi-provider 路由细节看 `.trellis/spec/backend/multi-provider.md`。

## 注意

- 单用户本地工具，**没有认证**——不要暴露到公网。
- LLM 调用走多个 OpenAI-compatible 中转（1m1ng / Volcengine Ark / SiliconFlow），需要确认在本地网络可达。
- 上传的 xlsx 落在 `.data/uploads/`，**只读不修改**；所有产物在 `.data/runs/<job_id>/`。
- 数据库是 SQLite 单文件 `.data/linebase.db`，删掉即可重置。
