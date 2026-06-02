"""
检查 pending alpha 的相关性状态

通过 GET /alphas/{alpha_id}/check 端点查询真实的 check 状态，
不需要提交 alpha。检查结果和汇总统计发送到飞书。

新入库的因子 status 为 "pending"，通过相关性检测后变为 "unsubmitted"。
默认删除未通过相关性检查的因子。

Usage:
    python check_correlation.py              # 检查所有 pending alpha（失败自动删除）
    python check_correlation.py --dry-run    # 只检查，不更新数据库
    python check_correlation.py --keep-fail  # 保留失败的 alpha 不删除
    python check_correlation.py --no-notify  # 不发送飞书通知
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import requests
from requests.auth import HTTPBasicAuth
from core.config import load_credentials
from core.alpha_db import get_alpha_db
from core.notifier import get_notifier
from core.log_manager import setup_logger

logger = setup_logger(__name__)

BASE_URL = "https://api.worldquantbrain.com"
ACCEPT_V2 = "application/json;version=2.0"


def check_alpha(session: requests.Session, alpha_id: str, max_wait: int = 120) -> dict:
    """
    通过 /check 端点获取 alpha 的真实检查状态（不提交）。
    返回: {"success": True, "checks": [...]} 或 {"success": False, "error": "..."}
    """
    url = f"{BASE_URL}/alphas/{alpha_id}/check"
    deadline = time.time() + max_wait

    while time.time() < deadline:
        try:
            resp = session.get(
                url,
                headers={"Accept": ACCEPT_V2},
                verify=False,
                timeout=30
            )

            # 检查 Retry-After 头
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait = float(retry_after)
                time.sleep(wait if wait > 0 else 3)
                continue

            if resp.status_code == 200 and resp.text:
                data = resp.json()
                checks = data.get("is", {}).get("checks", [])
                return {"success": True, "checks": checks}

            time.sleep(3)

        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "error": "Timeout"}


def get_self_correlation(checks: list) -> dict:
    """从 checks 列表中提取 SELF_CORRELATION 状态"""
    for check in checks:
        if check.get("name") == "SELF_CORRELATION":
            return {
                "status": check.get("result", "UNKNOWN"),
                "value": check.get("value"),
                "limit": check.get("limit"),
            }
    return {"status": "NOT_FOUND", "value": None, "limit": None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="检查 alpha 相关性状态")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不更新数据库")
    parser.add_argument("--keep-fail", action="store_true", help="保留 SELF_CORRELATION FAIL 的 alpha（默认删除）")
    parser.add_argument("--no-notify", action="store_true", help="不发送飞书通知")
    args = parser.parse_args()

    # Authenticate
    username, password = load_credentials()
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    resp = session.post(f"{BASE_URL}/authentication", verify=False, timeout=15)

    if resp.status_code != 201:
        print(f"Authentication failed: {resp.text}")
        sys.exit(1)

    print("Authentication successful\n")

    db = get_alpha_db()
    notifier = get_notifier()

    # Get all pending alphas (waiting for correlation check)
    alphas = db.get_all_alphas(limit=10000)
    unsubmitted = [a for a in alphas if a.get("status") == "pending"]

    if not unsubmitted:
        print("没有 pending 的 alpha")
        return

    print(f"找到 {len(unsubmitted)} 个 pending alpha\n")

    stats = {
        "total": len(unsubmitted),
        "pass": 0,
        "fail": 0,
        "pending": 0,
        "updated": 0,
        "deleted": 0,
        "error": 0,
    }

    failed_alphas = []

    for i, alpha in enumerate(unsubmitted, 1):
        alpha_id = alpha.get("alpha_id")
        if not alpha_id:
            continue

        expression = alpha.get("expression", "")[:50]
        print(f"[{i}/{len(unsubmitted)}] {alpha_id}", end=" ", flush=True)

        result = check_alpha(session, alpha_id)

        if not result["success"]:
            print(f"错误: {result['error']}")
            stats["error"] += 1
            continue

        checks = result["checks"]
        sc = get_self_correlation(checks)

        if sc["status"] == "PASS":
            print(f"✓ PASS (value={sc['value']:.4f}, limit={sc['limit']})")
            stats["pass"] += 1
            if not args.dry_run:
                db.update_alpha_checks(alpha_id, checks)
                db.update_alpha_status(alpha_id, "unsubmitted")
                stats["updated"] += 1

        elif sc["status"] == "FAIL":
            print(f"✗ FAIL (value={sc['value']:.4f}, limit={sc['limit']})")
            stats["fail"] += 1
            failed_alphas.append({
                "alpha_id": alpha_id,
                "value": sc["value"],
                "limit": sc["limit"],
            })
            if not args.dry_run and not args.keep_fail:
                db.delete_alpha_by_alpha_id(alpha_id)
                stats["deleted"] += 1
                print(f"  -> 已删除")

        elif sc["status"] == "PENDING":
            print("⏳ PENDING")
            stats["pending"] += 1

        else:
            print(f"? {sc['status']}")
            stats["error"] += 1

        # 避免 API 限流
        time.sleep(2)

    # Get summary statistics
    summary = db.get_alpha_summary()

    # Print summary
    print("\n" + "=" * 50)
    print("检查完成")
    print("=" * 50)
    print(f"总数: {stats['total']}")
    print(f"PASS: {stats['pass']}")
    print(f"FAIL: {stats['fail']}")
    print(f"PENDING: {stats['pending']}")
    if not args.dry_run:
        print(f"已更新: {stats['updated']}")
        if not args.keep_fail:
            print(f"已删除: {stats['deleted']}")
    print(f"错误: {stats['error']}")
    print()
    print("因子库汇总:")
    print(f"  总数: {summary['total']}")
    print(f"  已提交: {summary['submitted']}")
    print(f"  未提交: {summary['unsubmitted']}")
    print(f"  累计因子: {summary['new_all_time']}")
    print(f"  累计可提交: {summary['submittable_all_time']}")

    # Send Feishu notification
    if not args.no_notify and notifier.enabled:
        print("\n发送飞书通知...")
        notifier.notify_correlation_check(
            total=stats["total"],
            passed=stats["pass"],
            failed=stats["fail"],
            failed_alphas=failed_alphas,
            summary=summary,
        )
        print("飞书通知已发送")


if __name__ == "__main__":
    main()
