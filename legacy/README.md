# legacy/

这里放历史兼容脚本和旧版因子挖掘器。

## 内容
- `alpha50_miner.py` — Alpha50 CSV 挖掘器
- `alpha101_miner.py` — 101 Formulaic Alphas 挖掘器

## 运行示例
```bash
uv run python -m legacy.alpha50_miner --csv alpha50.csv
uv run python -m legacy.alpha101_miner
```

这些脚本主要用于兼容历史流程，不建议作为日常默认入口。

