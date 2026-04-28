# WorldQuant Alpha 因子生成系统

基于本地 Ollama 大语言模型的智能 Alpha 因子生成系统，自动生成、测试并提交 Alpha 因子到 WorldQuant Brain 平台。

## 核心特性

- **本地 LLM 集成** — 使用 Ollama 运行 qwen3.5、deepseek-coder 等模型，Mixture of Experts 多主题策略生成
- **统一 API 会话层** — `SessionManager` 集中管理认证、代理、全局 429 冷却、指数退避重试
- **Alpha 生命周期状态机** — generated → simulating → simulated → checked → submitted，带状态转换校验
- **提交配额感知** — 每日提交计数，持久化追踪，避免超限
- **异步并发轮询** — `AsyncSimulationPoller` 基于 asyncio 的模拟进度并发轮询
- **全局槽位管理** — SQLite 实现跨进程并发数 + 每日上限控制
- **智能模型舰队管理** — 自动 VRAM 监控和模型降级
- **遗传进化引擎** — 表达式交叉、变异、锦标赛选择，多臂老虎机算子/字段探索
- **Web 监控界面** — Flask 实时状态与手动控制
- **自动化工作流** — 连续 Alpha 生成、挖掘和提交

## 系统要求

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器
- Ollama 本地服务
- WorldQuant Brain 账户
- GPU（可选，NVIDIA 加速推理）

## 快速开始

### 1. 安装依赖

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖
uv sync

# 如需 GPU 支持（PyTorch）
uv sync --extra gpu
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```env
WQ_USERNAME=your.email@worldquant.com
WQ_PASSWORD=your_password
OLLAMA_URL=http://localhost:11434
# 可选：覆盖默认模型
# WQ_DEFAULT_MODEL=qwen3.5:35b
```

### 3. 启动 Ollama 并拉取模型

```bash
ollama serve
ollama pull qwen3.5:35b
ollama pull qwen3-coder:latest
ollama pull deepseek-coder-v2:16b
```

### 4. 启动系统

```bash
# 连续挖掘模式（推荐）— 同时运行生成器 + 挖掘器 + 提交器
uv run alpha-orchestrator --mode continuous

# 或者使用长格式
uv run python -m core.alpha_orchestrator --mode continuous
```

## 项目结构

```
WorldQuant/
├── core/                                    # 核心业务逻辑
│   ├── config.py                            # 环境变量、凭据、模型配置
│   ├── api_session.py                       # 统一 API 会话（认证/代理/限流/重试）
│   ├── alpha_db.py                          # SQLite 数据库（WAL 模式，线程安全）
│   ├── alpha_lifecycle.py                   # Alpha 生命周期状态机
│   ├── submission_quota.py                  # 每日提交配额追踪
│   ├── async_poller.py                      # asyncio 并发轮询器
│   ├── simulation_slot_manager.py           # 全局模拟槽位管理
│   ├── region_config.py                     # 区域/股票池配置
│   ├── alpha_orchestrator.py                # 主编排器（调度、模型舰队管理）
│   ├── alpha_generator_ollama.py            # Alpha 生成器（Ollama LLM）
│   ├── improved_alpha_submitter.py          # Alpha 提交器
│   ├── machine_lib.py                       # WorldQuant Brain API 封装
│   ├── ast_validator.py                     # 表达式语法/语义校验
│   ├── expression_compiler.py               # 模板表达式生成与变异
│   ├── self_optimizer.py                    # 算子/窗口/分组成功率追踪
│   ├── quality_monitor.py                   # 质量监控与退化检测
│   ├── hypothesis_manager.py                # 假设提取与管理
│   ├── log_manager.py                       # 日志（按模块、按日期分文件）
│   ├── data_fetcher/                        # 数据获取子包
│   │   ├── operator_fetcher.py              #   算子获取 + 磁盘缓存
│   │   ├── data_field_fetcher.py            #   数据字段获取 + 磁盘缓存
│   │   └── smart_search.py                  #   TF-IDF 智能字段搜索
├── miners/                                  # 挖掘策略
│   ├── alpha_expression_miner.py            # 表达式参数挖掘
│   ├── alpha_expression_miner_continuous.py # 持续表达式挖掘
│   ├── template_grid_miner.py               # 模板网格挖掘
│   └── machine_miner.py                     # 暴力字段×算子组合挖掘
├── evolution/                               # 遗传算法与优化
│   ├── genetic_engine.py                    # 遗传引擎（交叉/变异/锦标赛选择）
│   ├── bandits.py                           # 多臂老虎机（UCB1 算子/字段探索）
│   └── similarity.py                        # 模板相似度去重
├── infrastructure/                          # 基础设施与监控
│   ├── model_fleet_manager.py               # Docker 模型舰队管理
│   ├── vram_monitor.py                      # GPU 显存监控（nvidia-smi）
│   └── health_check.py                      # 健康检查（Ollama / WQ / Web）
├── web/                                     # Web 界面
│   ├── dashboard.py                         # Flask 仪表盘
│   └── templates/dashboard.html             # 仪表盘模板
├── experiments/                             # 研究实验脚本
│   ├── pipeline.py                          # 闭环管道（AI + 遗传进化 + 提交）
│   ├── manual_breakthrough.py               # 手工突破尝试
│   ├── mutate_alphas.py                     # LLM 驱动表达式变异
│   ├── polish_alphas.py                     # 网格搜索参数优化
│   └── audit_cloud.py                       # 云端 Alpha 审计
├── legacy/                                  # 历史兼容
│   ├── alpha50_miner.py                     # Alpha50 论文公式挖掘
│   └── alpha101_miner.py                    # Alpha101 公式挖掘
├── tests/                                   # 测试
├── insights/                                # 假设/洞察文本文件
├── data/                                    # 运行时数据（DB、状态文件）
├── cache/                                   # API 响应缓存
├── log/                                     # 日志输出
├── .env.example
├── pyproject.toml
└── README.md
```

## 快捷命令

`pyproject.toml` 注册了以下脚本，可直接用 `uv run <script>` 运行：

| 命令 | 说明 |
|------|------|
| `uv run alpha-orchestrator` | 主编排器 |
| `uv run alpha-generator` | Alpha 生成器 |
| `uv run alpha-submitter` | Alpha 提交器 |
| `uv run alpha-miner --expression "..."` | 表达式参数挖掘 |
| `uv run alpha-miner-continuous` | 持续表达式挖掘 |
| `uv run template-miner` | 模板网格挖掘 |
| `uv run machine-miner` | 机器挖掘 |
| `uv run fleet-manager --status` | 模型舰队管理 |
| `uv run vram-monitor` | GPU 显存监控 |
| `uv run health-check` | 健康检查 |
| `uv run dashboard` | Web 仪表盘 (http://localhost:5000) |

## 编排器运行模式

| 模式 | 说明 |
|------|------|
| `--mode continuous` | 同时运行生成器和挖掘器（默认） |
| `--mode daily` | 完整每日工作流（生成 → 挖掘 → 提交） |
| `--mode generator` | 仅运行 Alpha 生成器 |
| `--mode miner` | 仅运行表达式挖掘器 |
| `--mode submitter` | 仅运行 Alpha 提交器 |
| `--mode fleet-status` | 查看模型舰队状态 |
| `--mode fleet-reset` | 重置到最大模型 |
| `--mode fleet-downgrade` | 强制降级到下一模型 |

## 命令行参数

### 编排器 (`alpha-orchestrator`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `continuous` | 运行模式 |
| `--ollama-url` | `.env` 中的值 | Ollama API URL |
| `--ollama-model` | `qwen3.5:35b` | 使用的 Ollama 模型 |
| `--max-concurrent` | `3` | 最大并发模拟数 |
| `--batch-size` | `3` | 操作批次大小 |
| `--mining-interval` | `6` | 挖掘间隔（小时） |
| `--restart-interval` | `30` | 进程重启间隔（分钟） |

### 生成器 (`alpha-generator`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ollama-url` | `http://localhost:11434` | Ollama API URL |
| `--ollama-model` | `qwen3.5:35b` | 使用的模型 |
| `--batch-size` | `3` | 每批生成数量 |
| `--sleep-time` | `10` | 批次间休眠（秒） |
| `--max-concurrent` | `2` | 最大并发模拟数 |
| `--sim-timeout` | `1800` | 单次模拟超时（秒） |
| `--log-level` | `INFO` | 日志级别 |

### 提交器 (`alpha-submitter`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | `3` | 批次大小 |
| `--interval-hours` | `24` | 提交间隔（小时） |
| `--auto-mode` | 否 | 单次运行（不循环） |
| `--timeout-minutes` | `20` | 提交超时（分钟） |
| `--min-hopeful-count` | `50` | 最小候选数 |

### 管道 (`experiments.pipeline`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--region` | `USA` | 区域 |
| `--model` | `deepseek-coder-v2:16b` | 模型 |
| `--target` | `2` | 目标 Alpha 数 |
| `--max-batches` | `50` | 最大批次数 |
| `--concurrent` | `2` | 并发数 |
| `--mode` | `auto` | auto / grid |

## 架构说明

### API 会话层 (`api_session.py`)

所有模块通过 `get_session_manager()` 获取统一的会话单例，不再各自创建 `requests.Session`。`SessionManager` 提供：

- **主动重认证** — 每 25 分钟自动刷新，`login_lock` 防止并发重复登录
- **全局 429 冷却** — 任一线程收到限流后，所有线程统一等待
- **`request_with_retry()`** — 自动处理 401/403 重认证、429 等待、SSL 重试、超时退避
- **代理自适应** — 自动检测 HTTP/HTTPS/SOCKS 代理并配置

### Alpha 生命周期

```
generated → simulating → simulated → checked → submitted
                ↓            ↓           ↓
              failed ←── retrying ←──────┘
```

每次状态转换都经过 `validate_transition()` 校验，非法转换抛出 `ValueError`。数据库 `alphas` 表的 `lifecycle_state` 列在 `ON CONFLICT` 更新时自动保留 `submitted`/`checked` 不被降级。

### 提交配额

`SubmissionQuota` 单例追踪每日提交数，数据持久化到 `data/submission_log.json`。`submit_hopeful_alphas` 和 `batch_submit` 在每次成功提交后检查配额，达到上限自动停止。

### 日志系统

`log_manager.setup_logger(__name__, "module_tag")` 按模块和日期输出日志文件：
`log/{module}/{YYYY-MM-DD}.log`

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `WQ_USERNAME` | 是 | WorldQuant Brain 用户名 |
| `WQ_PASSWORD` | 是 | WorldQuant Brain 密码 |
| `OLLAMA_URL` | 否 | Ollama API 地址（默认 `http://localhost:11434`） |
| `WQ_DEFAULT_MODEL` | 否 | 覆盖默认模型（优先级高于 `data/model_config.json`） |

模型选择优先级：`WQ_DEFAULT_MODEL` 环境变量 → `data/model_config.json` → 回退 `qwen3.5:35b`。运行时可通过 `set_default_model()` 持久化到配置文件。

## 模型舰队

系统维护从大到小的模型舰队，自动处理 VRAM 问题：

1. **qwen3.5:35b** (23.8 GB) — 首选默认模型
2. **qwen3-coder:latest** (18.6 GB) — 代码能力备选
3. **deepseek-coder-v2:16b** (8.9 GB) — 轻量级备选

连续 3 次 VRAM 错误后自动降级。状态持久化到 `data/model_fleet_state.json`。

## Docker 部署

```bash
# 启动所有服务（Ollama + 挖掘器 + 仪表盘）
docker-compose up -d

# 查看日志
docker-compose logs -f alpha-miner
```

## 故障排除

### 认证失败
- 确认 `.env` 文件存在且凭据正确
- 系统会自动重试 3 次（指数退避），无需手动干预

### Ollama 连接失败
- 确认 Ollama 已启动：`ollama serve`
- 确认模型已拉取：`ollama list`

### API 速率限制 (429)
- 全局冷却机制会自动等待 `Retry-After` 时间
- 可减少 `--batch-size` 和 `--max-concurrent`

### VRAM 不足
- 减少 `--batch-size` 和 `--max-concurrent`
- 系统会检测 VRAM 错误并自动降级模型

### 模拟超时
- 默认 30 分钟超时，可通过 `--sim-timeout` 调整
- 超时后模拟标记为失败，表达式进入重试队列