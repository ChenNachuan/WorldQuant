# WorldQuant Alpha 因子生成系统

基于本地 Ollama 大语言模型的智能 Alpha 因子生成系统，自动生成、测试并提交 Alpha 因子到 WorldQuant Brain 平台。

## 核心特性

- **本地 LLM 集成** — 使用 Ollama 运行 qwen3.5、deepseek-coder 等模型
- **智能模型舰队管理** — 自动 VRAM 监控和模型降级
- **并发执行架构** — Alpha 生成器和挖掘器同时运行
- **Web 监控界面** — Flask 实时监控与手动控制
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

复制示例文件并填入你的凭据：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
WQ_USERNAME=your.email@worldquant.com
WQ_PASSWORD=your_password
OLLAMA_URL=http://localhost:11434
```

> **注意**：`.env` 文件已被 `.gitignore` 忽略，不会被提交到版本控制。

### 3. 启动 Ollama 服务

```bash
ollama serve
ollama pull qwen3.5:35b
ollama pull qwen3-coder:latest
ollama pull deepseek-coder-v2:16b
```

### 4. 启动系统

```bash
# 连续挖掘模式（推荐）
uv run python -m core.alpha_orchestrator --mode continuous

# 仅运行生成器
uv run python -m core.alpha_generator_ollama

# 仅运行提交器
uv run python -m core.improved_alpha_submitter --auto-mode

# 运行健康检查
uv run python -m infrastructure.health_check
```

### 5. 访问 Web 界面

```bash
uv run python -m web.dashboard
```

浏览器打开 http://localhost:5000

## 项目结构

```
WorldQuant/
├── core/                                    # 核心业务逻辑
│   ├── config.py                            # 环境变量与凭据加载
│   ├── alpha_orchestrator.py                # 编排和调度
│   ├── alpha_generator_ollama.py            # Alpha 生成器（Ollama）
│   ├── improved_alpha_submitter.py          # Alpha 提交器
│   └── machine_lib.py                       # WorldQuant Brain API 库
├── miners/                                  # 挖掘策略
│   ├── alpha_expression_miner.py            # 表达式参数挖掘
│   ├── alpha_expression_miner_continuous.py # 持续表达式挖掘
│   ├── template_grid_miner.py               # 模板网格挖掘
│   └── machine_miner.py                     # 机器挖掘器
├── infrastructure/                          # 基础设施与监控
│   ├── model_fleet_manager.py               # 模型舰队管理
│   ├── vram_monitor.py                      # VRAM 监控
│   └── health_check.py                      # 健康检查
├── web/                                     # Web 界面
│   ├── dashboard.py                         # Flask 仪表盘
│   └── templates/
│       └── dashboard.html
├── tests/                                   # 测试
│   ├── test_orchestrator.py
│   └── test_alpha_removal.py
├── .env.example                             # 环境变量模板
├── .gitignore
├── pyproject.toml                           # 项目配置与依赖
├── uv.lock                                  # 锁定的依赖版本
└── README.md
```

## 运行方式

所有模块均使用 `python -m` 方式从项目根目录运行：

| 命令 | 说明 |
|------|------|
| `uv run python -m core.alpha_orchestrator` | 主编排器 |
| `uv run python -m core.alpha_generator_ollama` | Alpha 生成器 |
| `uv run python -m core.improved_alpha_submitter` | Alpha 提交器 |
| `uv run python -m miners.alpha_expression_miner --expression "..."` | 表达式挖掘 |
| `uv run python -m miners.alpha_expression_miner_continuous` | 持续挖掘 |
| `uv run python -m miners.template_grid_miner` | 模板网格挖掘 |
| `uv run python -m miners.machine_miner` | 机器挖掘 |
| `uv run python -m infrastructure.vram_monitor` | VRAM 监控 |
| `uv run python -m infrastructure.model_fleet_manager` | 模型舰队管理 |
| `uv run python -m infrastructure.health_check` | 健康检查 |
| `uv run python -m web.dashboard` | Web 仪表盘 |

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

### 编排器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `continuous` | 运行模式 |
| `--ollama-url` | `.env` 中的值 | Ollama API URL |
| `--ollama-model` | `qwen3.5:35b` | 使用的 Ollama 模型 |
| `--max-concurrent` | `3` | 最大并发模拟数 |
| `--batch-size` | `3` | 操作批次大小 |
| `--mining-interval` | `6` | 连续模式挖掘间隔（小时） |
| `--restart-interval` | `30` | 进程重启间隔（分钟） |

### 生成器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ollama-url` | `.env` 中的值 | Ollama API URL |
| `--ollama-model` | `qwen3.5:35b` | 使用的 Ollama 模型 |
| `--batch-size` | `3` | 每批生成 Alpha 数 |
| `--sleep-time` | `10` | 批次间休眠（秒） |
| `--max-concurrent` | `2` | 最大并发模拟数 |

## 模型舰队管理

系统维护从大到小的模型舰队，自动处理 VRAM 问题：

1. **qwen3.5:35b** (23.8 GB) — 首选默认模型
2. **qwen3-coder:latest** (18.6 GB) — 代码能力备选
3. **deepseek-coder-v2:16b** (8.9 GB) — 轻量级备选

当连续 3 次 VRAM 错误后，系统自动降级到更小的模型。状态持久化到 `model_fleet_state.json`。

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `WQ_USERNAME` | 是 | WorldQuant Brain 用户名 |
| `WQ_PASSWORD` | 是 | WorldQuant Brain 密码 |
| `OLLAMA_URL` | 否 | Ollama API 地址（默认 `http://localhost:11434`） |

## 故障排除

### 认证失败
- 确认 `.env` 文件存在且 `WQ_USERNAME` / `WQ_PASSWORD` 正确
- 检查 WorldQuant Brain API 状态

### Ollama 连接失败
- 确认 Ollama 服务已启动：`ollama serve`
- 确认模型已拉取：`ollama list`

### VRAM 不足
- 减少 `--batch-size` 和 `--max-concurrent`
- 系统会自动降级到更小的模型

### API 速率限制 (429)
- 减少提交批次大小
- 避免手动和定时提交同时进行

## 日志文件

- `alpha_orchestrator.log` — 编排器日志
- `alpha_generator_ollama.log` — 生成器日志
- `alpha_miner.log` — 挖掘器日志
- `improved_alpha_submitter.log` — 提交器日志
