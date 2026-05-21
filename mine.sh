#!/bin/bash
# 一键启动 Alpha 挖掘器
# Usage: ./mine.sh

cd "$(dirname "$0")" || exit

echo "=========================================="
echo "  WorldQuant Alpha 挖掘器"
echo "=========================================="
echo ""
echo "正在启动..."
echo "LLM: DeepSeek"
echo "成员: nachuan"
echo "并发: 2 workers"
echo ""
echo "按 Ctrl+C 停止"
echo "=========================================="
echo ""

python run_alpha_miner.py --llm deepseek --member-id nachuan --workers 2
