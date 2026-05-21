#!/bin/bash
# 提交未提交的因子
# Usage:
#   ./submit.sh          # 提交所有未提交的因子
#   ./submit.sh 3        # 只提交前 3 个
#   ./submit.sh 1        # 只提交 1 个

cd "$(dirname "$0")" || exit

LIMIT=${1:-""}  # 第一个参数，如果没提供则为空

echo "=========================================="
echo "  WorldQuant Alpha 提交工具"
echo "=========================================="
echo ""

# 从数据库获取未提交的因子
if [ -z "$LIMIT" ]; then
    UNSUBMITTED=$(sqlite3 data/alphas.db "SELECT alpha_id FROM alphas WHERE status = 'unsubmitted';")
    MODE="全部"
else
    UNSUBMITTED=$(sqlite3 data/alphas.db "SELECT alpha_id FROM alphas WHERE status = 'unsubmitted' LIMIT $LIMIT;")
    MODE="前 $LIMIT 个"
fi

if [ -z "$UNSUBMITTED" ]; then
    echo "没有未提交的因子"
    exit 0
fi

COUNT=$(echo "$UNSUBMITTED" | wc -l | tr -d ' ')
echo "找到 $COUNT 个未提交的因子 ($MODE):"
echo "$UNSUBMITTED"
echo ""
echo "开始提交..."
echo "=========================================="
echo ""

# 转换为参数列表并提交
python submit_alpha.py "$UNSUBMITTED"

echo ""
echo "=========================================="
echo "提交完成！"
echo "=========================================="
