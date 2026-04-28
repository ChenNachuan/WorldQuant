# experiments/

这里放研究、验证和一次性实验脚本。

## 内容
- `pipeline.py` — 研究型闭环管道
- `manual_breakthrough.py` — 人工突破尝试
- `mutate_alphas.py` — 变异实验
- `polish_alphas.py` — 网格优化实验
- `audit_cloud.py` — 云端审计脚本
- `check_field_bug.py`、`check_heavy_ollama.py`、`check_ollama_custom.py` — 手工检查脚本

## 运行示例
```bash
uv run python -m experiments.pipeline
uv run python -m experiments.mutate_alphas
```

这些脚本不属于默认主线入口，只适合实验、调试和回溯分析。

