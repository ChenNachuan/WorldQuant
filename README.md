# WorldQuant Alpha 因子生成系统

基于 LLM 的智能 Alpha 因子生成系统，自动生成、测试并提交 Alpha 因子到 WorldQuant Brain 平台。

## 核心特性

- **双 LLM 支持** — 同时支持 DeepSeek API 和本地 Ollama 模型
- **动态赛道权重** — 强化学习风格的模块选择，按成功率动态调整
- **基因重组** — 30% 概率从共享池提取精英因子进行杂交
- **反向因子检测** — Sharpe < -0.8 时自动加负号重测
- **共享因子池** — 支持团队协作，防冲突的分布式因子池
- **智能挽救机制** — 边界因子自动变种优化
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

### 3. 获取运算符

```bash
# 从 WQ Brain API 获取所有可用运算符
python fetch_fields.py
```

运算符会保存到 `data/operators/operators.csv`

### 4. 添加数据字段

数据字段需要手动添加到 `data/fields/` 目录，CSV 文件需要包含 `Field` 和 `Description` 列：

```bash
# 示例：创建 price.csv
echo "Field,Description
close,收盘价
open,开盘价
high,最高价
low,最低价
volume,成交量" > data/fields/price.csv
```

### 5. 启动挖掘

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

### 6. 提交因子

挖掘出的因子会自动保存到数据库（状态为 "unsubmitted"），使用以下命令提交：

```bash
# 提交单个因子
python submit_alpha.py <alpha_id>

# 提交多个因子
python submit_alpha.py pwnbR9Gq akNmojM1
```

提交时会检查所有条件，失败会显示详细原因并从数据库删除。

### 7. 手动添加因子

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
│   └── log_manager.py             # 日志管理
├── data/                          # 运行时数据
│   ├── fields/                    # 数据字段 CSV（按类别分组）
│   │   ├── price&volume.csv
│   │   ├── fundamental.csv
│   │   ├── analyst.csv
│   │   └── sentiment.csv
│   ├── operators/                 # 运算符数据
│   │   └── operators.csv
│   └── shared_pool/               # 共享因子池（团队协作）
├── run_alpha_miner.py             # 主挖掘程序
├── submit_alpha.py                # 因子提交脚本
├── add_alpha.py                   # 手动添加因子脚本
├── fetch_fields.py                # 字段获取脚本
├── .env.example                   # 环境变量示例
└── README.md                      # 本文件
```

## 核心流程

### 1. 字段加载

系统从 `data/fields/` 目录读取 CSV 文件，每个文件代表一个数据类别：
- `price&volume.csv` — 价格和成交量相关字段
- `fundamental.csv` — 基本面字段
- `analyst.csv` — 分析师数据
- `sentiment.csv` — 情绪指标

### 2. 动态模块选择

系统维护 4 个模块的统计数据：
```python
MODULE_STATS = {
    "PRICE&VOLUME": {"tried": 0, "success": 0},
    "FUNDAMENTAL": {"tried": 0, "success": 0},
    "ANALYST": {"tried": 0, "success": 0},
    "SENTIMENT": {"tried": 0, "success": 0}
}
```

每次生成时：
- 根据成功率计算权重
- 随机选择 1-2 个模块进行交叉组合
- 成功率高的模块被选中概率更大

### 3. Alpha 生成策略

**新开采 (70% 概率)**
- 根据动态权重选择模块
- 从选中模块的字段中生成因子

**基因重组 (30% 概率)**
- 从共享池中随机选择 2 个精英因子
- 提取核心逻辑进行杂交
- 生成 3 个新变种

**挽救裂变 (优先级最高)**
- 对失败但有潜力的因子（|Sharpe| + |Fitness| > 1.7）
- 生成 3 个变种进行优化

### 4. 结果处理

| 条件 | 动作 |
|------|------|
| Sharpe ≥ 1.25 且 Fitness ≥ 1.0 且所有 checks 通过 | 保存到数据库（unsubmitted） |
| Sharpe > 1.0 且 Fitness > 0.8 | 加入共享池 |
| Sharpe < -0.8 | 加负号重测（反向因子） |
| abs(Sharpe) + abs(Fitness) > 1.7 | 触发挽救机制 |
| 其他 | 记录失败，更新模块权重 |

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

从 WQ Brain API 获取运算符并保存到 `data/operators/operators.csv`

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `WQ_USERNAME` | 是 | WorldQuant Brain 用户名 |
| `WQ_PASSWORD` | 是 | WorldQuant Brain 密码 |
| `DEEPSEEK_API_KEY` | 否 | DeepSeek API Key（使用 DeepSeek 时必需） |
| `OLLAMA_URL` | 否 | Ollama API 地址（默认 `http://localhost:11434`） |
| `OLLAMA_MODEL` | 否 | Ollama 模型名称（默认 `qwen3:8b`） |

## 使用 DeepSeek API

DeepSeek API 是推荐的选择，因为：
- 无需本地 GPU
- 响应速度快
- 支持 DeepSeek Reasoner 模型

获取 API Key：https://platform.deepseek.com/

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

## 开发说明

### 添加新的数据字段

1. 将字段信息保存为 CSV 文件到 `data/fields/` 目录
2. CSV 需要包含 `Field` 和 `Description` 列
3. 系统会自动加载并使用

### 自定义 Alpha 生成策略

编辑 `run_alpha_miner.py` 中的方法：
- `generate_alphas()` — 新因子生成
- `generate_crossover_alphas()` — 基因重组
- `_process_rescue_task()` — 挽救裂变

### 数据库结构

数据库使用 (expression, region, universe, neutralization) 作为主键，没有自增 ID 列。

状态值：
- `unsubmitted` — 符合条件但未提交
- `submitted` — 已成功提交

## 许可证

MIT License
