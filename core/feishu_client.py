"""
飞书 API 客户端 — 处理消息解析和回复。
"""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class FeishuClient:
    """飞书 API 客户端，用于接收和回复消息。"""

    def __init__(self, app_id: str = None, app_secret: str = None):
        self.app_id = app_id or os.getenv("FEISHU_APP_ID")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET")
        self._tenant_token: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_secret)

    def _get_tenant_token(self) -> str:
        """获取 tenant_access_token（带缓存）。"""
        if self._tenant_token:
            return self._tenant_token

        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 0:
            self._tenant_token = data["tenant_access_token"]
            return self._tenant_token
        else:
            logger.error(f"获取 tenant_access_token 失败: {data}")
            return None

    def verify_challenge(self, body: dict) -> dict:
        """处理飞书 webhook 验证请求。"""
        challenge = body.get("challenge")
        if challenge:
            return {"challenge": challenge}
        return {}

    def parse_message(self, body: dict) -> tuple:
        """
        解析消息事件。
        返回: (message_id, chat_id, text_content) 或 (None, None, None)
        """
        try:
            # 飞书事件格式
            event = body.get("event", {})
            message = event.get("message", {})

            logger.debug(f"收到事件: {body}")

            message_id = message.get("message_id")
            chat_id = message.get("chat_id")
            msg_type = message.get("message_type")

            # 只处理文本消息
            if msg_type != "text":
                return None, None, None

            # 解析文本内容
            content = message.get("content", "{}")
            import json
            text = json.loads(content).get("text", "").strip()

            return message_id, chat_id, text

        except Exception as e:
            logger.error(f"解析消息失败: {e}")
            return None, None, None

    def reply_message(self, message_id: str, title: str, content: str) -> bool:
        """回复消息（Markdown 卡片格式）。"""
        import json
        token = self._get_tenant_token()
        if not token:
            return False

        # 飞书卡片 V2 格式 - content 必须是字符串
        card = {
            "schema": "2.0",
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue"
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content
                    }
                ]
            }
        }

        payload = {
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False)
        }

        try:
            resp = requests.post(
                f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == 0:
                logger.info(f"回复消息成功: {message_id}")
                return True
            else:
                logger.warning(f"回复消息失败: {data}")
                return False
        except Exception as e:
            logger.warning(f"回复消息异常: {e}")
            return False


# ── Global singleton ─────────────────────────────────────────────────

_client_instance: Optional[FeishuClient] = None


def get_feishu_client() -> FeishuClient:
    """获取全局 FeishuClient 单例。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = FeishuClient()
    return _client_instance
