# WorldQuant Alpha 因子生成系统

基于 LLM 的智能 Alpha 因子生成系统，自动生成、测试并提交 Alpha 因子到 WorldQuant Brain 平台。

## 核心特性

- **双 LLM 支持** — 同时支持 DeepSeek API 和本地 Ollama 模型
- **简化 API 会话** — 直接 requests.Session，自动重试和重认证
- **提交配额感知** — 每日提交计数，持久化追踪
- **异步生产者-消费者** — LLM 生成 + 并发回测流水线
- **智能挽救机制** — 边界因子自动变种优化

## 系统要求

- Python ≥ 3.11
- WorldQuant Brain 账户
- DeepSeek API Key（推荐）或本地 Ollama 服务

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
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

### 3. 启动系统

```bash
# 使用 DeepSeek API（默认）
python run_alpha_miner.py

# 指定 LLM 提供商
python run_alpha_miner.py --llm deepseek
python run_alpha_miner.py --llm ollama

# 调整并发数
python run_alpha_miner.py --workers 3
```

## 项目结构

```
WorldQuant/
├── core/                          # 核心业务逻辑
│   ├── config.py                  # 环境变量和凭据加载
│   ├── api_session.py             # 简化 API 会话管理
│   ├── alpha_db.py                # SQLite Alpha 数据库
│   ├── llm_client.py              # 统一 LLM 客户端（Ollama + DeepSeek）
│   ├── submission_quota.py        # 每日提交配额追踪
│   ├── log_manager.py             # 日志管理
│   └── alpha_lifecycle.py         # Alpha 生命周期状态机
├── run_alpha_miner.py             # 主程序入口
├── IQC_final.ipynb                # 参考实现（Jupyter Notebook）
├── .env.example                   # 环境变量示例
├── pyproject.toml                 # 项目配置
└── README.md                      # 本文件
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm` | `auto` | LLM 提供商：`auto`, `deepseek`, `ollama` |
| `--workers` | `2` | 并发模拟 worker 数量 |

## 核心流程

### Alpha 生成

1. LLM 根据提供的数据字段生成 Alpha 表达式
2. 表达式格式：`ts_decay_linear(group_neutralize(zscore(...), subindustry), 5)`
3. 每次生成 5 个候选因子

### 回测与提交

1. 提交因子到 WorldQuant Brain 进行回测
2. 轮询等待回测结果（最长 10 分钟）
3. 根据结果决定下一步：
   - **Sharpe ≥ 1.25 且 Fitness ≥ 1.0** → 自动提交
   - **|Sharpe| + |Fitness| > 1.7** → 触发挽救机制，生成变种
   - **Sharpe < -0.8** → 加负号重测（反向因子）
   - 其他 → 记录失败，继续下一个

### 挽救机制

对于边界因子（表现一般但有潜力），系统会：
1. 将因子信息发送给 LLM
2. LLM 生成 3 个变种因子
3. 变种因子进入回测队列

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `WQ_USERNAME` | 是 | WorldQuant Brain 用户名 |
| `WQ_PASSWORD` | 是 | WorldQuant Brain 密码 |
| `DEEPSEEK_API_KEY` | 否* | DeepSeek API Key（使用 DeepSeek 时必需） |
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

## 开发说明

### 添加新的 LLM 提供商

编辑 `core/llm_client.py`，在 `LLMClient` 类中添加新方法：

```python
def _setup_new_provider(self):
    # 初始化代码
    pass

def _generate_new_provider(self, system_prompt, user_prompt):
    # 生成代码
    pass
```

### 自定义 Alpha 生成策略

编辑 `run_alpha_miner.py` 中的 `generate_alphas()` 方法，修改 prompt 或添加新的生成逻辑。

## 许可证

MIT License
