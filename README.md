# WorldQuant Alpha 因子生成系统

基于 LLM 的智能 Alpha 因子生成系统，自动生成、测试并提交 Alpha 因子到 WorldQuant Brain 平台。

## 核心特性

- **双 LLM 支持** — 同时支持 DeepSeek API 和本地 Ollama 模型
- **动态赛道权重** — 强化学习风格的模块选择，按成功率动态调整
- **随机字段采样** — 每次从 7,642 个字段中随机抽取 15 个，最大化探索范围
- **非线性基因重组** — 20% 概率从共享池提取精英因子，使用 ts_corr、ts_cov、rank 等非线性算子杂交，避免相关性检测失败
- **参数调优** — 检查失败时自动尝试 4 组代表性参数组合（中性化、截断、衰减）
- **智能挽救池** — 边界因子自动进入挽救池，针对性修复失败检查，最多尝试 3 次
- **反向因子检测** — Sharpe < -0.8 时自动加负号重测
- **共享因子池** — 支持团队协作，防冲突的分布式因子池
- **飞书通知** — 发现 Alpha、定期汇总、异常熔断时自动推送飞书消息
- **严格筛选** — 只保存符合条件的因子（Sharpe ≥ 1.25, Fitness ≥ 1.0, 所有 checks 通过）
- **手动提交** — 不自动提交，通过独立脚本手动控制提交

## 系统要求

- Python ≥ 3.11
- WorldQuant Brain 账户
- DeepSeek API Key（推荐）或本地 Ollama 服务

## 快速开始

### 1. 安装依赖

```bash
pip install requests pandas python-dotenv openai
# 或使用 uv
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# WorldQuant Brain 凭据
WQ_USERNAME=your_email@example.com
WQ_PASSWORD=your_password

# LLM 配置（二选一）
# 选项 1: DeepSeek API（推荐）
DEEPSEEK_API_KEY=your_deepseek_api_key

# 选项 2: Ollama 本地模型
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

### 3. 获取数据字段和运算符

```bash
# 从 WQ Brain API 获取所有可用数据字段和运算符
python fetch_fields.py
```

数据字段会保存到 `data/fields/` 目录（16 个 CSV 文件，共 7,642 个字段），运算符保存到 `data/operators/operators.csv`。

### 4. 启动挖掘

```bash
# 使用 DeepSeek API（默认）
python run_alpha_miner.py

# 指定 LLM 提供商
python run_alpha_miner.py --llm deepseek
python run_alpha_miner.py --llm ollama

# 指定成员 ID（团队协作时使用）
python run_alpha_miner.py --member-id gao

# 调整并发数
python run_alpha_miner.py --workers 3
```

### 5. 提交因子

挖掘出的因子会自动保存到数据库（状态为 "unsubmitted"），使用以下命令提交：

```bash
# 提交单个因子
python submit_alpha.py <alpha_id>

# 提交多个因子
python submit_alpha.py pwnbR9Gq akNmojM1
```

提交时会检查所有条件，失败会显示详细原因并从数据库删除。

### 6. 手动添加因子

从 WorldQuant Brain 添加已知因子到数据库：

```bash
python add_alpha.py <alpha_id>
python add_alpha.py pwnbR9Gq akNmojM1
```

## 项目结构

```
WorldQuant/
├── core/                          # 核心业务逻辑
│   ├── config.py                  # 环境变量和凭据加载
│   ├── api_session.py             # API 会话管理
│   ├── alpha_db.py                # SQLite Alpha 数据库
│   ├── llm_client.py              # 统一 LLM 客户端（Ollama + DeepSeek）
│   ├── submission_quota.py        # 每日提交配额追踪
│   ├── notifier.py                # 飞书通知模块
│   ├── data_fetcher.py            # 数据字段和运算符获取
│   └── log_manager.py             # 日志管理（按天轮转）
├── data/                          # 运行时数据
│   ├── fields/                    # 数据字段 CSV（API 获取，16 个文件）
│   │   ├── analyst4.csv           # 分析师预估数据
│   │   ├── fundamental6.csv       # 公司基本面数据
│   │   ├── model77.csv            # 因子模型
│   │   ├── pv13.csv               # 价格与成交量
│   │   └── ...                    # 共 16 个数据集
│   ├── operators/                 # 运算符数据
│   │   └── operators.csv
│   └── shared_pool/               # 共享因子池（团队协作）
├── log/                           # 日志文件（按天自动轮转）
├── run_alpha_miner.py             # 主挖掘程序
├── submit_alpha.py                # 因子提交脚本
├── add_alpha.py                   # 手动添加因子脚本
├── fetch_fields.py                # 字段和运算符获取脚本
├── .env.example                   # 环境变量示例
└── README.md                      # 本文件
```

## 核心流程

### 1. 字段加载

系统启动时从 `data/fields/` 目录加载全部 16 个数据集的字段（共 7,642 个），存储在内存中作为字段池。

数据集包括：
- `analyst4.csv` — 分析师预估数据（1,325 字段）
- `fundamental6.csv` — 公司基本面数据（887 字段）
- `model77.csv` — 因子模型（3,257 字段）
- `pv13.csv` — 价格与成交量（166 字段）
- `news12.csv` — 新闻数据（876 字段）
- 等共 16 个数据集

### 2. 动态模块选择与随机采样

系统维护 16 个模块的统计数据，每次生成时：
1. 根据成功率计算权重，随机选择 1-2 个模块
2. 从选中模块的字段池中**随机抽取 15 个字段**（每次不同）
3. 成功率高的模块被选中概率更大

这确保了每次生成都使用不同的字段组合，最大化探索范围。

### 3. Alpha 生成策略

**新开采 (60% 概率)**
- 根据动态权重选择模块
- 从选中模块的字段中生成因子

**非线性基因重组 (20% 概率)**
- 从共享池中随机选择 2 个精英因子
- 提取核心逻辑，使用非线性算子杂交：
  - `ts_corr(A, B, d)` — 时序相关性
  - `ts_cov(A, B, d)` — 时序协方差
  - `rank(A) / rank(B)` — 排名比值
  - `sign(A) * abs(B)` — 符号组合
- 重新套上外壳 `ts_decay_linear(zscore(...), 5)`，生成 3 个新变种
- 中性化由 settings 控制（默认 INDUSTRY）

**挽救池 (20% 概率)**
- 从挽救池中提取边界因子
- 针对失败检查生成修复变种：
  - Turnover 过高 → 加大窗口参数
  - Self-Correlation 失败 → 改变中性化方式
  - Drawdown 过大 → 增加衰减平滑
- 最多尝试 3 次，失败则删除

### 4. 结果处理

| 条件 | 动作 |
|------|------|
| Sharpe ≥ 1.25 且 Fitness ≥ 1.0 且所有 checks 通过 | 保存到数据库（unsubmitted） |
| Sharpe ≥ 1.25 且 Fitness ≥ 1.0 但 checks 失败 | 参数调优 → 根据检查类型决定是否 rescue |
| Sharpe > 1.0 且 Fitness > 0.8 | 加入共享池 |
| Sharpe < -0.8 | 加负号重测（反向因子） |
| abs(Sharpe) + abs(Fitness) > 1.7 | 进入挽救池 |
| 其他 | 记录失败，更新模块权重 |

**参数调优流程**：
1. 检查失败时，尝试 4 组代表性参数组合
2. 参数组合包括不同的中性化方式（INDUSTRY/SUBINDUSTRY/SECTOR/MARKET）
3. 任一组合通过 → 保存到数据库
4. 全部失败 → 根据检查类型决定是否 rescue：
   - TURNOVER/DRAWDOWN → 进入挽救池（可通过调整参数修复）
   - SELF_CORRELATION → 丢弃（核心逻辑问题，rescue 无效）

### 5. 数据库状态

因子保存到数据库时的状态：
- `unsubmitted` — 符合条件但未提交
- `submitted` — 已成功提交到平台

提交时会进行最终验证：
- 检查 SELF_CORRELATION（相关性）
- 检查其他所有 checks
- 任何 check 失败都会显示详细原因并从数据库删除

### 6. 共享因子池

团队协作功能：
- 每个成员有独立的 JSON 文件：`shared_pool_{member_id}.json`
- 读取时合并所有成员的文件
- 按 Sharpe 排序，保留前 500 个因子
- 基因重组时从共享池提取精英

## 命令行参数

### run_alpha_miner.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm` | `auto` | LLM 提供商：`auto`, `deepseek`, `ollama` |
| `--workers` | `2` | 并发模拟 worker 数量 |
| `--member-id` | `default` | 成员 ID（团队协作时使用） |

### submit_alpha.py

```bash
python submit_alpha.py <alpha_id> [alpha_id2 ...]
```

提交因子到 WorldQuant Brain，失败会显示详细原因并从数据库删除。

### add_alpha.py

```bash
python add_alpha.py <alpha_id> [alpha_id2 ...]
```

从 WorldQuant Brain 获取因子信息并添加到数据库（状态为 "submitted"）。

### fetch_fields.py

从 WQ Brain API 获取数据字段和运算符：
- 数据字段保存到 `data/fields/`（16 个 CSV 文件）
- 运算符保存到 `data/operators/operators.csv`

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `WQ_USERNAME` | 是 | WorldQuant Brain 用户名 |
| `WQ_PASSWORD` | 是 | WorldQuant Brain 密码 |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key（使用 DeepSeek 时必需） |
| `OLLAMA_URL` | 否 | Ollama API 地址（默认 `http://localhost:11434`） |
| `OLLAMA_MODEL` | 否 | Ollama 模型名称（默认 `qwen3:8b`） |
| `FEISHU_WEBHOOK` | 否 | 飞书机器人 Webhook URL（接收通知） |

## 使用 DeepSeek API

DeepSeek API 是推荐的选择，因为：
- 无需本地 GPU
- 响应速度快
- 支持 DeepSeek v4-pro 模型

获取 API Key：https://platform.deepseek.com/

## 飞书通知

配置 `FEISHU_WEBHOOK` 环境变量后，系统会在以下情况自动推送飞书消息：

- **发现 Alpha**：Sharpe ≥ 1.25 且 Fitness ≥ 1.0
- **定期汇总**：每测试 100 个因子后发送统计摘要
- **异常熔断**：连续 3 次认证失败或连续 5 次 LLM 错误时报警

配置方法：
1. 飞书群 → 设置 → 机器人 → 添加自定义机器人
2. 复制 Webhook URL 到 `.env` 的 `FEISHU_WEBHOOK`

## 使用 Ollama 本地模型

如果选择使用 Ollama：

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型
ollama pull qwen3:8b

# 启动服务
ollama serve
```

## 团队协作

多人协作时，每人使用不同的 `--member-id`：

```bash
# 成员 A
python run_alpha_miner.py --member-id alice

# 成员 B
python run_alpha_miner.py --member-id bob
```

共享因子池会自动合并所有成员的因子，基因重组时会从全队精英中选择。

## 故障排除

### 认证失败
- 确认 `.env` 文件存在且凭据正确
- 系统会自动重试，无需手动干预

### DeepSeek API 错误
- 检查 `DEEPSEEK_API_KEY` 是否正确
- 确认 API 账户有足够额度

### Ollama 连接失败
- 确认 Ollama 已启动：`ollama serve`
- 确认模型已拉取：`ollama list`

### API 速率限制
- 系统会自动处理 429 错误
- 可减少 `--workers` 数量

### 模拟超时
- 默认 10 分钟超时
- 超时后模拟标记为失败，继续下一个

### 提交失败
- 提交时会检查所有条件
- 失败会显示详细原因（如 SELF_CORRELATION 失败）
- 失败的因子会自动从数据库删除

## 日志

日志文件保存在 `log/` 目录，按天自动轮转：
- 当前日志：`log/YYYY-MM-DD.log`
- 自动在午夜切换到新文件
- 保留最近 30 天的日志

## 开发说明

### 更新数据字段

```bash
# 从 API 获取最新的数据字段和运算符
python fetch_fields.py
```

字段会自动保存到 `data/fields/` 目录，程序启动时自动加载。

### 自定义 Alpha 生成策略

编辑 `run_alpha_miner.py` 中的方法：
- `generate_alphas()` — 新因子生成
- `generate_crossover_alphas()` — 基因重组
- `_process_rescue_task()` — 挽救裂变

### 数据库结构

**alphas 表**：存储符合条件的因子
- 主键：(expression, region, universe, neutralization)
- 状态值：
  - `unsubmitted` — 符合条件但未提交
  - `submitted` — 已成功提交

**rescue_pool 表**：存储需要挽救的边界因子
- 主键：alpha_id（WorldQuant 分配的唯一 ID）
- attempt_count：挽救尝试次数，最多 3 次
- failed_checks：失败的检查项，用于针对性修复

## 许可证

MIT License
