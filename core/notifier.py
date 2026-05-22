"""
通知模块 — 飞书 Webhook 通知，支持 Alpha 发现、定期汇总、异常熔断报警。
"""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    """飞书 Webhook 通知器。"""

    def __init__(self):
        self.webhook_url: Optional[str] = os.getenv("FEISHU_WEBHOOK")
        if self.webhook_url:
            logger.info("飞书通知已启用")
        else:
            logger.info("未配置 FEISHU_WEBHOOK，通知已禁用")

        # 熔断计数器
        self._consecutive_auth_failures = 0
        self._consecutive_llm_errors = 0

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, title: str, content: str) -> bool:
        """发送飞书消息（富文本格式）。返回是否成功。"""
        if not self.webhook_url:
            return False

        try:
            resp = requests.post(
                self.webhook_url,
                json={
                    "msg_type": "post",
                    "content": {
                        "post": {
                            "zh_cn": {
                                "title": title,
                                "content": [[{"tag": "text", "text": content}]],
                            }
                        }
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    logger.info(f"飞书通知发送成功: {title}")
                    return True
                else:
                    logger.warning(f"飞书通知失败: {data}")
            else:
                logger.warning(f"飞书通知 HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"飞书通知异常: {e}")
        return False

    def send_markdown(self, title: str, markdown: str) -> bool:
        """发送飞书 Markdown 消息。返回是否成功。"""
        if not self.webhook_url:
            return False

        # 飞书 Markdown 消息格式
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": markdown
                    }
                ]
            }
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    logger.info(f"飞书 Markdown 通知发送成功: {title}")
                    return True
                else:
                    logger.warning(f"飞书通知失败: {data}")
            else:
                logger.warning(f"飞书通知 HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"飞书通知异常: {e}")
        return False

    # ── Alpha 发现通知 ──────────────────────────────────────────────

    def notify_alpha(
        self,
        alpha_id: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        expression: str,
        member_id: str = "",
    ):
        """发现符合条件的 Alpha 时发送通知。"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        expr_short = expression[:80] + ("..." if len(expression) > 80 else "")
        lines = [
            f"时间: {timestamp}",
            f"Alpha ID: {alpha_id}",
            f"Expression: {expr_short}",
            f"Sharpe: {sharpe:.2f} | Fitness: {fitness:.2f} | Turnover: {turnover:.2f}",
        ]
        if member_id:
            lines.append(f"Member: {member_id}")
        self.send("发现新 Alpha!", "\n".join(lines))

    # ── 相关性检查通知 ──────────────────────────────────────────────

    def notify_correlation_check(
        self,
        total: int,
        passed: int,
        failed: int,
        failed_alphas: list,
        summary: dict = None,
    ):
        """相关性检查结果通知（Markdown 格式）。"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md_lines = [
            f"**检查时间**: {timestamp}",
            "",
            "## 相关性检查结果",
            "",
            f"| 指标 | 数量 |",
            f"|------|------|",
            f"| 检查总数 | {total} |",
            f"| ✅ PASS | {passed} |",
            f"| ❌ FAIL | {failed} |",
            "",
        ]

        if failed_alphas:
            md_lines.append("## 失败详情")
            md_lines.append("")
            md_lines.append("| Alpha ID | Correlation | Limit |")
            md_lines.append("|----------|-------------|-------|")
            for alpha in failed_alphas[:10]:  # 最多显示 10 个
                md_lines.append(f"| {alpha['alpha_id']} | {alpha['value']:.4f} | {alpha['limit']} |")
            if len(failed_alphas) > 10:
                md_lines.append(f"| ... | 共 {len(failed_alphas)} 个 | ... |")
            md_lines.append("")

        if summary:
            md_lines.append("## 因子库汇总")
            md_lines.append("")
            md_lines.append("| 指标 | 数量 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 因子总数 | {summary.get('total', 0)} |")
            md_lines.append(f"| 已提交 | {summary.get('submitted', 0)} |")
            md_lines.append(f"| 未提交 | {summary.get('unsubmitted', 0)} |")
            md_lines.append("")
            md_lines.append("## 过去 24 小时")
            md_lines.append("")
            md_lines.append("| 指标 | 数量 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 新增因子 | {summary.get('new_24h', 0)} |")
            md_lines.append(f"| 可提交 | {summary.get('submittable_24h', 0)} |")

        self.send_markdown("相关性检查报告", "\n".join(md_lines))

    # ── 定期汇总通知 ────────────────────────────────────────────────

    def notify_summary(
        self,
        tested: int,
        passed: int,
        failed: int,
        best_sharpe: float,
        best_fitness: float,
        rescue_pool: int,
        member_id: str = "",
    ):
        """定期汇总通知。"""
        lines = [
            "已测试: {} | 通过: {} | 失败: {}".format(tested, passed, failed),
            "最佳 Sharpe: {:.2f} | 最佳 Fitness: {:.2f}".format(
                best_sharpe, best_fitness
            ),
            "Rescue Pool: {}".format(rescue_pool),
        ]
        if member_id:
            lines.append("Member: {}".format(member_id))
        self.send("挖矿进度汇总", "\n".join(lines))

    # ── 异常熔断报警 ────────────────────────────────────────────────

    def record_auth_failure(self):
        """记录一次鉴权失败，连续 3 次触发熔断报警。"""
        self._consecutive_auth_failures += 1
        if self._consecutive_auth_failures >= 3:
            self.send(
                "矿机宕机警告",
                "连续 {} 次鉴权失败 (AUTH_FAILED)\n"
                "Token 可能已过期，矿机已停止运行。\n"
                "请检查账号密码或重新登录。".format(
                    self._consecutive_auth_failures
                ),
            )

    def record_auth_success(self):
        """鉴权成功，重置计数器。"""
        self._consecutive_auth_failures = 0

    def record_llm_error(self):
        """记录一次 LLM 调用失败，连续 5 次触发熔断报警。"""
        self._consecutive_llm_errors += 1
        if self._consecutive_llm_errors >= 5:
            self.send(
                "矿机宕机警告",
                "连续 {} 次 LLM 调用失败\n"
                "DeepSeek API 额度可能已耗尽。\n"
                "请检查 API Key 余额。".format(
                    self._consecutive_llm_errors
                ),
            )

    def record_llm_success(self):
        """LLM 调用成功，重置计数器。"""
        self._consecutive_llm_errors = 0

    def notify_fatal(self, reason: str, member_id: str = ""):
        """致命错误导致矿机停止时发送最高级别警告。"""
        lines = [reason]
        if member_id:
            lines.append("Member: {}".format(member_id))
        self.send("矿机宕机警告", "\n".join(lines))


_notifier: Optional[Notifier] = None


def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
