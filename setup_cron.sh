#!/bin/bash
# 设置定时相关性检查
# Usage:
#   ./setup_cron.sh          # 添加定时任务（每天北京时间 9:00）
#   ./setup_cron.sh remove   # 移除定时任务
#   ./setup_cron.sh status   # 查看定时任务状态

cd "$(dirname "$0")" || exit

SCRIPT_DIR="$(pwd)"
CRON_ENTRY="0 1 * * * cd $SCRIPT_DIR && python check_correlation.py --delete-fail >> log/correlation_check.log 2>&1"

add_cron() {
    echo "=========================================="
    echo "  设置定时相关性检查"
    echo "=========================================="
    echo ""
    echo "将添加以下 crontab 任务："
    echo "  每天北京时间 9:00（UTC 1:00）运行相关性检查"
    echo "  自动删除 SELF_CORRELATION FAIL 的 alpha"
    echo ""

    # Check if cron entry already exists
    if crontab -l 2>/dev/null | grep -q "check_correlation.py"; then
        echo "定时任务已存在："
        crontab -l | grep "check_correlation.py"
        echo ""
        read -p "是否更新？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "已取消"
            return
        fi
        # Remove old entry
        crontab -l 2>/dev/null | grep -v "check_correlation.py" | crontab -
    fi

    # Add new cron entry
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    echo "定时任务已添加！"
    echo ""
    echo "当前 crontab："
    crontab -l | grep "check_correlation.py"
    echo ""
    echo "日志文件: log/correlation_check.log"
    echo ""
    echo "手动运行检查："
    echo "  python check_correlation.py --dry-run    # 只检查"
    echo "  python check_correlation.py --delete-fail # 检查并删除"
    echo "=========================================="
}

remove_cron() {
    echo "移除定时相关性检查..."
    if crontab -l 2>/dev/null | grep -q "check_correlation.py"; then
        crontab -l 2>/dev/null | grep -v "check_correlation.py" | crontab -
        echo "定时任务已移除"
    else
        echo "未找到定时任务"
    fi
}

status() {
    echo "=========================================="
    echo "  定时相关性检查状态"
    echo "=========================================="
    echo ""
    if crontab -l 2>/dev/null | grep -q "check_correlation.py"; then
        echo "  状态: 已启用"
        echo "  任务: $(crontab -l | grep 'check_correlation.py')"
        echo ""
        echo "  日志文件: log/correlation_check.log"
        if [ -f "log/correlation_check.log" ]; then
            echo "  最近日志:"
            tail -5 "log/correlation_check.log" | sed 's/^/    /'
        fi
    else
        echo "  状态: 未启用"
        echo ""
        echo "  使用 './setup_cron.sh' 添加定时任务"
    fi
    echo ""
    echo "=========================================="
}

# Main
case "${1:-add}" in
    add)
        add_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {add|remove|status}"
        echo ""
        echo "  add     添加定时任务（每天北京时间 9:00）"
        echo "  remove  移除定时任务"
        echo "  status  查看定时任务状态"
        exit 1
        ;;
esac
