"""Slack MCP Server Implementation.

Provides Slack messaging and notifications through MCP protocol:
- Send messages
- Send alerts with levels
- Post to channels
- Thread replies
- File uploads
"""

import logging
from typing import Any, Optional

try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class SlackMCPServer:
    """Slack MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not SLACK_SDK_AVAILABLE:
            raise ImportError("slack-sdk is required for Slack MCP server. Install with: pip install slack-sdk")
        
        self.config = config
        self.token = config.get("token")
        self.webhook_url = config.get("webhook_url")
        self.default_channel = config.get("default_channel", "#general")
        self.bot_name = config.get("bot_name", "Axiom Bot")
        
        self._client: Optional[AsyncWebClient] = None
        
        if self.token:
            self._client = AsyncWebClient(token=self.token)

    async def send_message(
        self,
        channel: str,
        message: str,
        thread_ts: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Send message to Slack channel.

        Args:
            channel: Channel name or ID
            message: Message text
            thread_ts: Thread timestamp for replies
            attachments: Message attachments

        Returns:
            Send result
        """
        if not self._client:
            return {
                "success": False,
                "error": "Slack client not configured. Token required.",
            }

        try:
            # Normalize channel name
            if not channel.startswith("#") and not channel.startswith("C"):
                channel = f"#{channel}"

            # Send message
            response = await self._client.chat_postMessage(
                channel=channel,
                text=message,
                thread_ts=thread_ts,
                attachments=attachments,
            )

            return {
                "success": True,
                "channel": channel,
                "message": message,
                "timestamp": response["ts"],
                "thread_ts": thread_ts,
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return {
                "success": False,
                "error": f"Slack API error: {e.response['error']}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return {
                "success": False,
                "error": f"Failed to send message: {str(e)}",
                "channel": channel,
            }

    async def send_alert(
        self,
        channel: str,
        title: str,
        message: str,
        level: str = "info",
        fields: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Send alert notification.

        Args:
            channel: Channel name or ID
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)
            fields: Additional fields

        Returns:
            Send result
        """
        # Define colors for different levels
        colors = {
            "info": "#36a64f",      # green
            "warning": "#ff9900",    # orange
            "error": "#ff0000",      # red
            "critical": "#8b0000",   # dark red
        }

        color = colors.get(level, colors["info"])

        # Build attachment
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "footer": self.bot_name,
            "ts": None,  # Will be set by Slack
        }

        if fields:
            attachment["fields"] = fields

        return await self.send_message(
            channel=channel,
            message=f"*{level.upper()}*: {title}",
            attachments=[attachment],
        )

    async def send_formatted_message(
        self,
        channel: str,
        blocks: list[dict[str, Any]],
        text: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send formatted message using Block Kit.

        Args:
            channel: Channel name or ID
            blocks: Block Kit blocks
            text: Fallback text

        Returns:
            Send result
        """
        if not self._client:
            return {
                "success": False,
                "error": "Slack client not configured. Token required.",
            }

        try:
            # Normalize channel name
            if not channel.startswith("#") and not channel.startswith("C"):
                channel = f"#{channel}"

            # Send message
            response = await self._client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=text or "Message",
            )

            return {
                "success": True,
                "channel": channel,
                "timestamp": response["ts"],
                "blocks": len(blocks),
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return {
                "success": False,
                "error": f"Slack API error: {e.response['error']}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to send formatted message: {e}")
            return {
                "success": False,
                "error": f"Failed to send message: {str(e)}",
                "channel": channel,
            }

    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upload file to Slack channel.

        Args:
            channel: Channel name or ID
            file_path: Path to file
            title: File title
            comment: File comment

        Returns:
            Upload result
        """
        if not self._client:
            return {
                "success": False,
                "error": "Slack client not configured. Token required.",
            }

        try:
            # Normalize channel name
            if not channel.startswith("#") and not channel.startswith("C"):
                channel = f"#{channel}"

            # Upload file
            response = await self._client.files_upload(
                channels=channel,
                file=file_path,
                title=title,
                initial_comment=comment,
            )

            return {
                "success": True,
                "channel": channel,
                "file_path": file_path,
                "file_id": response["file"]["id"],
                "permalink": response["file"]["permalink"],
            }

        except SlackApiError as e:
            logger.error(f"Slack file upload error: {e.response['error']}")
            return {
                "success": False,
                "error": f"File upload error: {e.response['error']}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return {
                "success": False,
                "error": f"Failed to upload file: {str(e)}",
                "channel": channel,
            }

    async def get_channel_info(self, channel: str) -> dict[str, Any]:
        """Get channel information.

        Args:
            channel: Channel name or ID

        Returns:
            Channel information
        """
        if not self._client:
            return {
                "success": False,
                "error": "Slack client not configured. Token required.",
            }

        try:
            # Normalize channel name
            if not channel.startswith("#") and not channel.startswith("C"):
                channel = f"#{channel}"

            # Get channel info
            response = await self._client.conversations_info(channel=channel)

            channel_info = response["channel"]

            return {
                "success": True,
                "channel_id": channel_info["id"],
                "name": channel_info["name"],
                "is_private": channel_info.get("is_private", False),
                "topic": channel_info.get("topic", {}).get("value", ""),
                "purpose": channel_info.get("purpose", {}).get("value", ""),
                "member_count": channel_info.get("num_members", 0),
            }

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return {
                "success": False,
                "error": f"Slack API error: {e.response['error']}",
                "channel": channel,
            }
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return {
                "success": False,
                "error": f"Failed to get channel info: {str(e)}",
                "channel": channel,
            }


def get_server_definition() -> dict[str, Any]:
    """Get Slack MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "slack",
        "category": "communication",
        "description": "Slack messaging and notifications (messages, alerts, files)",
        "tools": [
            {
                "name": "send_message",
                "description": "Send message to Slack channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name or ID (e.g., '#general' or 'C1234567890')",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message text",
                        },
                        "thread_ts": {
                            "type": "string",
                            "description": "Thread timestamp for replies",
                        },
                        "attachments": {
                            "type": "array",
                            "description": "Message attachments",
                        },
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "name": "send_alert",
                "description": "Send alert notification with level and formatting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name or ID",
                        },
                        "title": {
                            "type": "string",
                            "description": "Alert title",
                        },
                        "message": {
                            "type": "string",
                            "description": "Alert message",
                        },
                        "level": {
                            "type": "string",
                            "enum": ["info", "warning", "error", "critical"],
                            "description": "Alert level",
                            "default": "info",
                        },
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "value": {"type": "string"},
                                    "short": {"type": "boolean"},
                                },
                            },
                            "description": "Additional fields",
                        },
                    },
                    "required": ["channel", "title", "message"],
                },
            },
            {
                "name": "send_formatted_message",
                "description": "Send formatted message using Slack Block Kit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name or ID",
                        },
                        "blocks": {
                            "type": "array",
                            "description": "Block Kit blocks",
                        },
                        "text": {
                            "type": "string",
                            "description": "Fallback text",
                        },
                    },
                    "required": ["channel", "blocks"],
                },
            },
            {
                "name": "upload_file",
                "description": "Upload file to Slack channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name or ID",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to file to upload",
                        },
                        "title": {
                            "type": "string",
                            "description": "File title",
                        },
                        "comment": {
                            "type": "string",
                            "description": "File comment",
                        },
                    },
                    "required": ["channel", "file_path"],
                },
            },
            {
                "name": "get_channel_info",
                "description": "Get information about a Slack channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel name or ID",
                        }
                    },
                    "required": ["channel"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "communication",
            "requires": ["slack-sdk"],
        },
    }