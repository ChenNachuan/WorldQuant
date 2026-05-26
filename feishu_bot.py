"""
飞书机器人指令控制系统

通过飞书群消息远程控制挖掘器的运行状态、查询因子库、
执行相关性检查和提交因子。

命令列表:
    /summary           - 查看因子库统计
    /start [workers]   - 启动挖掘器（默认 2 workers）
    /stop              - 停止挖掘器
    /check             - 运行相关性检查
    /submit <id> [...] - 提交指定因子

Usage:
    python feishu_bot.py              # 启动 bot（默认端口 9000）
    python feishu_bot.py --port 8080  # 指定端口
"""

import os
import sys
import signal
import subprocess
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from core.alpha_db import get_alpha_db
from core.feishu_client import get_feishu_client
from core.notifier import get_notifier
from core.log_manager import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)

# PID 文件路径
PID_FILE = ".miner.pid"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 进程管理 ─────────────────────────────────────────────────────────

def is_miner_running() -> tuple:
    """检查挖掘进程是否运行。返回 (is_running, pid)。"""
    if not os.path.exists(PID_FILE):
        return False, 0

    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())

        # 检查进程是否存在
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, ValueError, PermissionError):
        # 进程不存在，清理 PID 文件
        try:
            os.remove(PID_FILE)
        except:
            pass
        return False, 0


def start_miner(workers: int = 2) -> tuple:
    """启动挖掘进程。返回 (success, message)。"""
    running, pid = is_miner_running()
    if running:
        return False, f"挖掘已在运行中 (PID={pid})"

    try:
        log_dir = os.path.join(PROJECT_DIR, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                [sys.executable, "run_alpha_miner.py", "--workers", str(workers)],
                cwd=PROJECT_DIR,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))

        logger.info(f"挖掘进程已启动: PID={proc.pid}, workers={workers}")
        return True, f"挖掘已启动 (PID={proc.pid}, workers={workers})"

    except Exception as e:
        logger.error(f"启动挖掘失败: {e}")
        return False, f"启动失败: {e}"


def stop_miner() -> tuple:
    """停止挖掘进程。返回 (success, message)。"""
    running, pid = is_miner_running()
    if not running:
        return False, "挖掘未在运行"

    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"已发送 SIGTERM 到进程 {pid}")

        # 等待进程退出
        import time
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except ProcessLookupError:
                break
        else:
            # 超时，强制终止
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info(f"已发送 SIGKILL 到进程 {pid}")
            except ProcessLookupError:
                pass

        # 清理 PID 文件
        try:
            os.remove(PID_FILE)
        except:
            pass

        return True, "挖掘已停止"

    except ProcessLookupError:
        try:
            os.remove(PID_FILE)
        except:
            pass
        return False, "挖掘进程已退出"
    except Exception as e:
        logger.error(f"停止挖掘失败: {e}")
        return False, f"停止失败: {e}"


# ── 命令执行器 ───────────────────────────────────────────────────────

def cmd_summary(args: list) -> tuple:
    """执行 /summary 命令。返回 (title, content)。"""
    db = get_alpha_db()
    summary = db.get_alpha_summary()

    lines = [
        "## 因子库汇总",
        f"- 因子总数: **{summary['total']}**",
        f"- 已提交: **{summary['submitted']}**",
        f"- 待检查: **{summary['pending']}**",
        f"- 可提交: **{summary['unsubmitted']}**",
        "",
        "## 过去 24 小时",
        f"- 新增因子: **{summary['new_24h']}**",
        f"- 可提交: **{summary['submittable_24h']}**",
    ]

    # 挖掘状态
    running, pid = is_miner_running()
    lines.append("")
    lines.append("## 挖掘状态")
    if running:
        lines.append(f"- 状态: **运行中** (PID={pid})")
    else:
        lines.append("- 状态: **未运行**")

    return "因子库汇总", "\n".join(lines)


def cmd_start(args: list) -> tuple:
    """执行 /start 命令。返回 (title, content)。"""
    workers = 2
    if args:
        try:
            workers = int(args[0])
            if workers < 1 or workers > 10:
                return "启动失败", "workers 数量应在 1-10 之间"
        except ValueError:
            return "启动失败", f"无效的 workers 参数: {args[0]}"

    success, message = start_miner(workers)
    return "启动挖掘" if success else "启动失败", message


def cmd_stop(args: list) -> tuple:
    """执行 /stop 命令。返回 (title, content)。"""
    success, message = stop_miner()
    return "停止挖掘" if success else "停止失败", message


def cmd_check(args: list) -> tuple:
    """执行 /check 命令。返回 (title, content)。"""
    try:
        result = subprocess.run(
            [sys.executable, "check_correlation.py", "--no-notify"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # 解析输出
        output = result.stdout
        lines = ["## 相关性检查结果", ""]

        # 提取关键信息
        for line in output.split("\n"):
            if "找到" in line and "pending" in line:
                lines.append(f"- {line.strip()}")
            elif "PASS:" in line:
                lines.append(f"- {line.strip()}")
            elif "FAIL:" in line:
                lines.append(f"- {line.strip()}")
            elif "已删除:" in line:
                lines.append(f"- {line.strip()}")
            elif "错误:" in line:
                lines.append(f"- {line.strip()}")

        if result.returncode != 0:
            lines.append("")
            lines.append(f"**错误:** {result.stderr[:200]}")

        return "相关性检查", "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "检查超时", "相关性检查执行超时（>300s）"
    except Exception as e:
        return "检查失败", f"执行错误: {e}"


def cmd_submit(args: list) -> tuple:
    """执行 /submit 命令。返回 (title, content)。"""
    if not args:
        return "提交失败", "请指定 alpha ID，例如: /submit akornja9 QPnwOqKr"

    try:
        result = subprocess.run(
            [sys.executable, "submit_alpha.py"] + args,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout
        lines = ["## 提交结果", ""]

        # 解析每个因子的提交结果
        for line in output.split("\n"):
            if "Submitted successfully" in line:
                # 提取 alpha ID
                parts = line.split()
                alpha_id = parts[2] if len(parts) > 2 else "unknown"
                lines.append(f"- {alpha_id}: ✅ 成功")
            elif "Submission failed" in line:
                parts = line.split()
                alpha_id = parts[2] if len(parts) > 2 else "unknown"
                lines.append(f"- {alpha_id}: ❌ 失败")
            elif "Alpha deleted from database" in line:
                lines.append(f"  - 已从数据库删除")

        if not lines[2:]:  # 没有解析到结果
            lines.append(f"```\n{output[:500]}\n```")

        return "因子提交", "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "提交超时", "因子提交执行超时（>60s）"
    except Exception as e:
        return "提交失败", f"执行错误: {e}"


def cmd_list(args: list) -> tuple:
    """执行 /list 命令。返回 (title, content)。"""
    db = get_alpha_db()

    # 解析参数
    limit = 20  # 默认显示 20 条
    status_filter = None

    for arg in args:
        if arg.isdigit():
            limit = min(int(arg), 100)  # 最多 100 条
        elif arg in ["submitted", "unsubmitted", "pending", "tested"]:
            status_filter = arg

    # 查询数据
    with db._cursor() as cur:
        if status_filter:
            cur.execute(
                "SELECT alpha_id, status, grade, fitness, sharpe FROM alphas WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status_filter, limit)
            )
        else:
            cur.execute(
                "SELECT alpha_id, status, grade, fitness, sharpe FROM alphas ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        rows = cur.fetchall()

    if not rows:
        return "因子列表", "暂无数据"

    # 构建表格
    lines = [f"## 因子列表（共 {len(rows)} 条）", ""]
    lines.append("| Alpha ID | Status | Grade | Fitness | Sharpe |")
    lines.append("|----------|--------|-------|---------|--------|")

    for row in rows:
        alpha_id = row["alpha_id"] or "-"
        status = row["status"] or "-"
        grade = row["grade"] or "-"
        fitness = f"{row['fitness']:.4f}" if row["fitness"] else "-"
        sharpe = f"{row['sharpe']:.4f}" if row["sharpe"] else "-"
        lines.append(f"| {alpha_id} | {status} | {grade} | {fitness} | {sharpe} |")

    # 添加使用提示
    lines.append("")
    lines.append("*提示: /list [数量] [状态]，例如: /list 50 submitted*")

    return "因子列表", "\n".join(lines)


# ── 命令路由 ─────────────────────────────────────────────────────────

COMMANDS = {
    "/summary": cmd_summary,
    "/start": cmd_start,
    "/stop": cmd_stop,
    "/check": cmd_check,
    "/submit": cmd_submit,
    "/list": cmd_list,
}


def parse_command(text: str) -> tuple:
    """解析命令文本。返回 (command, args)。"""
    parts = text.strip().split()
    if not parts:
        return None, []

    cmd = parts[0].lower()
    args = parts[1:]
    return cmd, args


# ── Webhook 路由 ─────────────────────────────────────────────────────

@app.route("/feishu/webhook", methods=["POST"])
def webhook():
    """接收飞书事件回调。"""
    body = request.get_json(force=True)

    # 处理 challenge 验证
    if "challenge" in body:
        logger.info("收到 challenge 验证请求")
        client = get_feishu_client()
        return jsonify(client.verify_challenge(body))

    # 记录收到的事件类型
    event_type = body.get("type") or body.get("header", {}).get("event_type", "unknown")
    logger.info(f"收到飞书事件: type={event_type}")

    # 解析消息
    client = get_feishu_client()
    message_id, chat_id, text = client.parse_message(body)

    if not message_id or not text:
        logger.debug(f"消息解析结果为空 (message_id={message_id}, text={text})")
        return jsonify({"code": 0})

    logger.info(f"收到消息: {text}")

    # 解析命令
    cmd, args = parse_command(text)

    if cmd not in COMMANDS:
        # 未知命令
        client.reply_message(
            message_id,
            "未知命令",
            f"支持的命令：\n" + "\n".join(f"- `{k}`" for k in COMMANDS.keys())
        )
        return jsonify({"code": 0})

    # 执行命令
    try:
        title, content = COMMANDS[cmd](args)
        client.reply_message(message_id, title, content)
    except Exception as e:
        logger.error(f"执行命令失败: {e}")
        client.reply_message(message_id, "执行失败", f"错误: {e}")

    return jsonify({"code": 0})


@app.route("/health", methods=["GET"])
def health():
    """健康检查。"""
    running, pid = is_miner_running()
    return jsonify({
        "status": "ok",
        "miner_running": running,
        "miner_pid": pid if running else None,
    })


@app.route("/test", methods=["POST"])
def test_command():
    """本地测试接口，直接执行命令并返回结果。"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "请提供 text 参数"}), 400

    logger.info(f"[测试] 收到命令: {text}")

    # 解析命令
    cmd, args = parse_command(text)

    if cmd not in COMMANDS:
        return jsonify({
            "command": text,
            "error": f"未知命令，支持: {', '.join(COMMANDS.keys())}"
        })

    # 执行命令
    try:
        title, content = COMMANDS[cmd](args)
        return jsonify({
            "command": text,
            "title": title,
            "content": content,
        })
    except Exception as e:
        logger.error(f"[测试] 执行命令失败: {e}")
        return jsonify({
            "command": text,
            "error": str(e),
        }), 500


# ── 主入口 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="飞书机器人指令控制系统")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FEISHU_BOT_PORT", "9000")),
        help="HTTP 服务端口 (默认: 9000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0)",
    )
    args = parser.parse_args()

    client = get_feishu_client()
    if not client.enabled:
        logger.error("未配置 FEISHU_APP_ID 或 FEISHU_APP_SECRET")
        print("错误: 请在 .env 中配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        sys.exit(1)

    print("=" * 50)
    print("  飞书机器人指令控制系统")
    print("=" * 50)
    print(f"  监听地址: {args.host}:{args.port}")
    print(f"  Webhook: http://<服务器IP>:{args.port}/feishu/webhook")
    print(f"  健康检查: http://localhost:{args.port}/health")
    print("-" * 50)
    print("  支持命令: /summary, /start, /stop, /check, /submit")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
