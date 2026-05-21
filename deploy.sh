#!/bin/bash
# WorldQuant Alpha Miner - Docker 部署脚本
# Usage:
#   ./deploy.sh          # 启动矿工
#   ./deploy.sh stop     # 停止矿工
#   ./deploy.sh logs     # 查看日志
#   ./deploy.sh submit   # 提交因子
#   ./deploy.sh status   # 查看状态

cd "$(dirname "$0")" || exit

case "${1:-start}" in
    start)
        echo "=========================================="
        echo "  启动 WorldQuant Alpha Miner"
        echo "=========================================="
        echo ""

        # Check .env file
        if [ ! -f .env ]; then
            echo "错误: .env 文件不存在"
            echo "请先创建 .env 文件并配置凭据"
            exit 1
        fi

        # Build and start
        docker compose up -d --build miner

        echo ""
        echo "矿工已启动！"
        echo ""
        echo "常用命令："
        echo "  查看日志: docker compose logs -f miner"
        echo "  停止矿工: ./deploy.sh stop"
        echo "  查看状态: ./deploy.sh status"
        echo "=========================================="
        ;;

    stop)
        echo "停止矿工..."
        docker compose down
        echo "已停止"
        ;;

    logs)
        docker compose logs -f miner
        ;;

    status)
        echo "=========================================="
        echo "  WorldQuant Alpha Miner 状态"
        echo "=========================================="
        echo ""

        # Container status
        docker compose ps

        echo ""
        echo "最近日志："
        docker compose logs --tail=20 miner
        ;;

    submit)
        shift  # Remove 'submit' from args
        if [ -z "$1" ]; then
            echo "提交所有未提交的因子..."
            docker compose run --rm submit python submit_alpha.py
        else
            echo "提交因子: $*"
            docker compose run --rm submit python submit_alpha.py "$@"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|logs|status|submit}"
        exit 1
        ;;
esac
