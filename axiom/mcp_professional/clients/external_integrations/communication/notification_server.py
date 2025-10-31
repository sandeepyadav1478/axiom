"""Notification MCP Server Implementation.

Provides unified notification services through MCP protocol:
- Email via SMTP
- Email via SendGrid/Mailgun
- SMS via Twilio
- Multi-channel notifications with auto-routing
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Any, Optional, Union
from datetime import datetime

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, Content, Attachment, FileContent, FileName, FileType
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationMCPServer:
    """Notification MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        
        # SMTP configuration
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.smtp_user = config.get("smtp_user")
        self.smtp_password = config.get("smtp_password")
        self.smtp_from_address = config.get("smtp_from_address", self.smtp_user)
        self.smtp_use_tls = config.get("smtp_use_tls", True)
        
        # SendGrid configuration
        self.sendgrid_api_key = config.get("sendgrid_api_key")
        self._sendgrid_client = None
        
        # Mailgun configuration
        self.mailgun_api_key = config.get("mailgun_api_key")
        self.mailgun_domain = config.get("mailgun_domain")
        
        # Twilio configuration
        self.twilio_account_sid = config.get("twilio_account_sid")
        self.twilio_auth_token = config.get("twilio_auth_token")
        self.twilio_from_number = config.get("twilio_from_number")
        self._twilio_client = None
        
        # Default channels
        self.default_channel = config.get("default_channel", "email")

    def _get_sendgrid_client(self):
        """Get or create SendGrid client."""
        if not SENDGRID_AVAILABLE:
            raise ImportError("sendgrid is required. Install with: pip install sendgrid")
        
        if self._sendgrid_client is None and self.sendgrid_api_key:
            self._sendgrid_client = sendgrid.SendGridAPIClient(api_key=self.sendgrid_api_key)
        return self._sendgrid_client

    def _get_twilio_client(self):
        """Get or create Twilio client."""
        if not TWILIO_AVAILABLE:
            raise ImportError("twilio is required. Install with: pip install twilio")
        
        if self._twilio_client is None and self.twilio_account_sid and self.twilio_auth_token:
            self._twilio_client = TwilioClient(self.twilio_account_sid, self.twilio_auth_token)
        return self._twilio_client

    # ===== EMAIL (SMTP) OPERATIONS =====

    async def send_email(
        self,
        to: Union[str, list[str]],
        subject: str,
        body: str,
        from_address: Optional[str] = None,
        cc: Optional[Union[str, list[str]]] = None,
        bcc: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, Any]:
        """Send plain text email via SMTP.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body (plain text)
            from_address: Sender address (default: configured address)
            cc: CC recipient(s)
            bcc: BCC recipient(s)

        Returns:
            Send result
        """
        try:
            if not self.smtp_user or not self.smtp_password:
                return {
                    "success": False,
                    "error": "SMTP credentials not configured",
                }
            
            from_addr = from_address or self.smtp_from_address
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = to if isinstance(to, str) else ", ".join(to)
            msg['Subject'] = subject
            
            if cc:
                msg['Cc'] = cc if isinstance(cc, str) else ", ".join(cc)
            if bcc:
                msg['Bcc'] = bcc if isinstance(bcc, str) else ", ".join(bcc)
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Prepare recipients list
            recipients = [to] if isinstance(to, str) else to
            if cc:
                recipients.extend([cc] if isinstance(cc, str) else cc)
            if bcc:
                recipients.extend([bcc] if isinstance(bcc, str) else bcc)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return {
                "success": True,
                "to": to,
                "subject": subject,
                "method": "smtp",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {
                "success": False,
                "error": f"Email send failed: {str(e)}",
                "to": to,
            }

    async def send_html_email(
        self,
        to: Union[str, list[str]],
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send HTML email via SMTP.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            html_body: Email body (HTML)
            plain_body: Plain text alternative
            from_address: Sender address

        Returns:
            Send result
        """
        try:
            if not self.smtp_user or not self.smtp_password:
                return {
                    "success": False,
                    "error": "SMTP credentials not configured",
                }
            
            from_addr = from_address or self.smtp_from_address
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = from_addr
            msg['To'] = to if isinstance(to, str) else ", ".join(to)
            msg['Subject'] = subject
            
            # Add plain text alternative if provided
            if plain_body:
                msg.attach(MIMEText(plain_body, 'plain'))
            
            # Add HTML body
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            recipients = [to] if isinstance(to, str) else to
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return {
                "success": True,
                "to": to,
                "subject": subject,
                "format": "html",
                "method": "smtp",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send HTML email: {e}")
            return {
                "success": False,
                "error": f"HTML email send failed: {str(e)}",
                "to": to,
            }

    async def send_with_attachment(
        self,
        to: Union[str, list[str]],
        subject: str,
        body: str,
        attachments: list[str],
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send email with file attachments via SMTP.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body
            attachments: List of file paths to attach
            from_address: Sender address

        Returns:
            Send result
        """
        try:
            if not self.smtp_user or not self.smtp_password:
                return {
                    "success": False,
                    "error": "SMTP credentials not configured",
                }
            
            from_addr = from_address or self.smtp_from_address
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = to if isinstance(to, str) else ", ".join(to)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            attached_files = []
            for file_path in attachments:
                try:
                    with open(file_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        
                        import os
                        filename = os.path.basename(file_path)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {filename}'
                        )
                        msg.attach(part)
                        attached_files.append(filename)
                except Exception as e:
                    logger.warning(f"Failed to attach file {file_path}: {e}")
            
            # Send email
            recipients = [to] if isinstance(to, str) else to
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return {
                "success": True,
                "to": to,
                "subject": subject,
                "attachments": attached_files,
                "attachment_count": len(attached_files),
                "method": "smtp",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send email with attachments: {e}")
            return {
                "success": False,
                "error": f"Email send failed: {str(e)}",
                "to": to,
            }

    async def send_daily_report(
        self,
        to: Union[str, list[str]],
        report_title: str,
        report_data: dict[str, Any],
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send daily summary report via email.

        Args:
            to: Recipient email address(es)
            report_title: Report title
            report_data: Report data dictionary
            from_address: Sender address

        Returns:
            Send result
        """
        try:
            # Generate report HTML
            html_body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    h1 {{ color: #333; }}
                    .metric {{ padding: 10px; margin: 10px 0; background-color: #f5f5f5; border-radius: 5px; }}
                    .metric-name {{ font-weight: bold; color: #666; }}
                    .metric-value {{ font-size: 24px; color: #333; }}
                </style>
            </head>
            <body>
                <h1>{report_title}</h1>
                <p>Report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            """
            
            for key, value in report_data.items():
                html_body += f"""
                <div class="metric">
                    <div class="metric-name">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html_body += """
            </body>
            </html>
            """
            
            # Generate plain text version
            plain_body = f"{report_title}\n{'=' * len(report_title)}\n\n"
            for key, value in report_data.items():
                plain_body += f"{key.replace('_', ' ').title()}: {value}\n"
            plain_body += f"\nReport generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            
            # Send email
            return await self.send_html_email(
                to=to,
                subject=f"Daily Report: {report_title}",
                html_body=html_body,
                plain_body=plain_body,
                from_address=from_address,
            )

        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
            return {
                "success": False,
                "error": f"Report send failed: {str(e)}",
                "to": to,
            }

    # ===== EMAIL (SENDGRID/MAILGUN) OPERATIONS =====

    async def send_transactional(
        self,
        to: str,
        template_id: str,
        dynamic_data: Optional[dict[str, Any]] = None,
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send transactional email via SendGrid.

        Args:
            to: Recipient email address
            template_id: SendGrid template ID
            dynamic_data: Dynamic template data
            from_address: Sender address

        Returns:
            Send result
        """
        try:
            client = self._get_sendgrid_client()
            if not client:
                return {
                    "success": False,
                    "error": "SendGrid not configured",
                }
            
            from_addr = from_address or self.smtp_from_address
            
            message = Mail(
                from_email=from_addr,
                to_emails=to,
            )
            message.template_id = template_id
            
            if dynamic_data:
                message.dynamic_template_data = dynamic_data
            
            response = client.send(message)
            
            return {
                "success": True,
                "to": to,
                "template_id": template_id,
                "method": "sendgrid",
                "status_code": response.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send transactional email: {e}")
            return {
                "success": False,
                "error": f"Transactional email failed: {str(e)}",
                "to": to,
            }

    async def send_bulk(
        self,
        recipients: list[str],
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send bulk email to multiple recipients.

        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            body: Email body
            from_address: Sender address

        Returns:
            Send result
        """
        try:
            results = []
            failed = []
            
            for recipient in recipients:
                result = await self.send_email(
                    to=recipient,
                    subject=subject,
                    body=body,
                    from_address=from_address,
                )
                
                if result["success"]:
                    results.append(recipient)
                else:
                    failed.append({"email": recipient, "error": result.get("error")})
            
            return {
                "success": True,
                "total_recipients": len(recipients),
                "successful": len(results),
                "failed": len(failed),
                "failed_details": failed,
                "method": "bulk",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send bulk email: {e}")
            return {
                "success": False,
                "error": f"Bulk email failed: {str(e)}",
                "total_recipients": len(recipients),
            }

    async def track_opens(
        self,
        to: str,
        subject: str,
        body: str,
        from_address: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send email with open tracking via SendGrid.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            from_address: Sender address

        Returns:
            Send result with tracking info
        """
        try:
            client = self._get_sendgrid_client()
            if not client:
                return {
                    "success": False,
                    "error": "SendGrid not configured",
                }
            
            from_addr = from_address or self.smtp_from_address
            
            message = Mail(
                from_email=from_addr,
                to_emails=to,
                subject=subject,
                html_content=body
            )
            
            # Enable tracking
            message.tracking_settings = {
                "open_tracking": {"enable": True},
                "click_tracking": {"enable": True},
            }
            
            response = client.send(message)
            
            return {
                "success": True,
                "to": to,
                "subject": subject,
                "tracking_enabled": True,
                "method": "sendgrid",
                "status_code": response.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send tracked email: {e}")
            return {
                "success": False,
                "error": f"Tracked email failed: {str(e)}",
                "to": to,
            }

    # ===== SMS (TWILIO) OPERATIONS =====

    async def send_sms(
        self,
        to: str,
        message: str,
        from_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send SMS via Twilio.

        Args:
            to: Recipient phone number (E.164 format)
            message: SMS message text
            from_number: Sender phone number

        Returns:
            Send result
        """
        try:
            client = self._get_twilio_client()
            if not client:
                return {
                    "success": False,
                    "error": "Twilio not configured",
                }
            
            from_num = from_number or self.twilio_from_number
            
            sms = client.messages.create(
                body=message,
                from_=from_num,
                to=to
            )
            
            return {
                "success": True,
                "to": to,
                "message_sid": sms.sid,
                "status": sms.status,
                "method": "twilio",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return {
                "success": False,
                "error": f"SMS send failed: {str(e)}",
                "to": to,
            }

    async def send_alert_sms(
        self,
        to: str,
        alert_level: str,
        alert_message: str,
        from_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send critical alert via SMS.

        Args:
            to: Recipient phone number
            alert_level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            alert_message: Alert message
            from_number: Sender phone number

        Returns:
            Send result
        """
        try:
            # Format message with alert level
            formatted_message = f"[{alert_level.upper()}] {alert_message}"
            
            return await self.send_sms(
                to=to,
                message=formatted_message,
                from_number=from_number,
            )

        except Exception as e:
            logger.error(f"Failed to send alert SMS: {e}")
            return {
                "success": False,
                "error": f"Alert SMS failed: {str(e)}",
                "to": to,
            }

    async def send_2fa(
        self,
        to: str,
        code: str,
        from_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send 2FA verification code via SMS.

        Args:
            to: Recipient phone number
            code: Verification code
            from_number: Sender phone number

        Returns:
            Send result
        """
        try:
            message = f"Your verification code is: {code}. This code will expire in 10 minutes."
            
            return await self.send_sms(
                to=to,
                message=message,
                from_number=from_number,
            )

        except Exception as e:
            logger.error(f"Failed to send 2FA SMS: {e}")
            return {
                "success": False,
                "error": f"2FA SMS failed: {str(e)}",
                "to": to,
            }

    # ===== MULTI-CHANNEL OPERATIONS =====

    async def send_notification(
        self,
        recipients: dict[str, Union[str, list[str]]],
        subject: str,
        message: str,
        channel: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send notification via auto-selected channel.

        Args:
            recipients: Dictionary with 'email' and/or 'phone' keys
            subject: Notification subject
            message: Notification message
            channel: Force specific channel (email/sms)

        Returns:
            Send result
        """
        try:
            results = {}
            
            # Determine channel
            selected_channel = channel or self.default_channel
            
            # Send via email
            if selected_channel == "email" and "email" in recipients:
                email_result = await self.send_email(
                    to=recipients["email"],
                    subject=subject,
                    body=message,
                )
                results["email"] = email_result
            
            # Send via SMS
            if selected_channel == "sms" and "phone" in recipients:
                phone = recipients["phone"]
                if isinstance(phone, list):
                    phone = phone[0]  # Use first number
                
                sms_result = await self.send_sms(
                    to=phone,
                    message=f"{subject}: {message}",
                )
                results["sms"] = sms_result
            
            # Send via both if no channel specified and both available
            if not channel:
                if "email" in recipients and "email" not in results:
                    email_result = await self.send_email(
                        to=recipients["email"],
                        subject=subject,
                        body=message,
                    )
                    results["email"] = email_result
            
            return {
                "success": True,
                "channels": list(results.keys()),
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {
                "success": False,
                "error": f"Notification failed: {str(e)}",
            }

    async def send_alert(
        self,
        recipients: dict[str, Union[str, list[str]]],
        severity: str,
        title: str,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send alert with severity-based routing.

        Args:
            recipients: Dictionary with 'email', 'phone', 'slack' keys
            severity: Alert severity (info, warning, error, critical)
            title: Alert title
            message: Alert message
            metadata: Additional metadata

        Returns:
            Send result
        """
        try:
            results = {}
            
            # Route based on severity
            # CRITICAL: All channels
            # ERROR: Email + SMS
            # WARNING: Email
            # INFO: Email (optional)
            
            if severity.lower() in ["critical", "error", "warning", "info"]:
                # Always send email for all severities
                if "email" in recipients:
                    email_body = f"""
Alert: {title}
Severity: {severity.upper()}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Message:
{message}
"""
                    if metadata:
                        email_body += "\n\nAdditional Information:\n"
                        for key, value in metadata.items():
                            email_body += f"{key}: {value}\n"
                    
                    email_result = await self.send_email(
                        to=recipients["email"],
                        subject=f"[{severity.upper()}] {title}",
                        body=email_body,
                    )
                    results["email"] = email_result
            
            # Send SMS for ERROR and CRITICAL
            if severity.lower() in ["critical", "error"]:
                if "phone" in recipients:
                    phone = recipients["phone"]
                    if isinstance(phone, list):
                        phone = phone[0]
                    
                    sms_result = await self.send_alert_sms(
                        to=phone,
                        alert_level=severity,
                        alert_message=f"{title}: {message}",
                    )
                    results["sms"] = sms_result
            
            return {
                "success": True,
                "severity": severity,
                "title": title,
                "channels": list(results.keys()),
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return {
                "success": False,
                "error": f"Alert send failed: {str(e)}",
                "severity": severity,
            }


def get_server_definition() -> dict[str, Any]:
    """Get Notification MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "notification",
        "category": "communication",
        "description": "Unified notification services (Email, SMS, Multi-channel)",
        "tools": [
            # Email (SMTP) Operations
            {
                "name": "send_email",
                "description": "Send plain text email via SMTP",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": ["string", "array"],
                            "description": "Recipient email address(es)",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body (plain text)",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address (optional)",
                        },
                        "cc": {
                            "type": ["string", "array"],
                            "description": "CC recipient(s)",
                        },
                        "bcc": {
                            "type": ["string", "array"],
                            "description": "BCC recipient(s)",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
            },
            {
                "name": "send_html_email",
                "description": "Send HTML email via SMTP",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": ["string", "array"],
                            "description": "Recipient email address(es)",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "html_body": {
                            "type": "string",
                            "description": "Email body (HTML)",
                        },
                        "plain_body": {
                            "type": "string",
                            "description": "Plain text alternative",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["to", "subject", "html_body"],
                },
            },
            {
                "name": "send_with_attachment",
                "description": "Send email with file attachments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": ["string", "array"],
                            "description": "Recipient email address(es)",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body",
                        },
                        "attachments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to attach",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["to", "subject", "body", "attachments"],
                },
            },
            {
                "name": "send_daily_report",
                "description": "Send daily summary report via email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": ["string", "array"],
                            "description": "Recipient email address(es)",
                        },
                        "report_title": {
                            "type": "string",
                            "description": "Report title",
                        },
                        "report_data": {
                            "type": "object",
                            "description": "Report data dictionary",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["to", "report_title", "report_data"],
                },
            },
            # Email (SendGrid/Mailgun) Operations
            {
                "name": "send_transactional",
                "description": "Send transactional email via SendGrid",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address",
                        },
                        "template_id": {
                            "type": "string",
                            "description": "SendGrid template ID",
                        },
                        "dynamic_data": {
                            "type": "object",
                            "description": "Dynamic template data",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["to", "template_id"],
                },
            },
            {
                "name": "send_bulk",
                "description": "Send bulk email to multiple recipients",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of recipient email addresses",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["recipients", "subject", "body"],
                },
            },
            {
                "name": "track_opens",
                "description": "Send email with open tracking",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body",
                        },
                        "from_address": {
                            "type": "string",
                            "description": "Sender address",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
            },
            # SMS (Twilio) Operations
            {
                "name": "send_sms",
                "description": "Send SMS via Twilio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient phone number (E.164 format)",
                        },
                        "message": {
                            "type": "string",
                            "description": "SMS message text",
                        },
                        "from_number": {
                            "type": "string",
                            "description": "Sender phone number",
                        },
                    },
                    "required": ["to", "message"],
                },
            },
            {
                "name": "send_alert_sms",
                "description": "Send critical alert via SMS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient phone number",
                        },
                        "alert_level": {
                            "type": "string",
                            "enum": ["INFO", "WARNING", "ERROR", "CRITICAL"],
                            "description": "Alert level",
                        },
                        "alert_message": {
                            "type": "string",
                            "description": "Alert message",
                        },
                        "from_number": {
                            "type": "string",
                            "description": "Sender phone number",
                        },
                    },
                    "required": ["to", "alert_level", "alert_message"],
                },
            },
            {
                "name": "send_2fa",
                "description": "Send 2FA verification code via SMS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient phone number",
                        },
                        "code": {
                            "type": "string",
                            "description": "Verification code",
                        },
                        "from_number": {
                            "type": "string",
                            "description": "Sender phone number",
                        },
                    },
                    "required": ["to", "code"],
                },
            },
            # Multi-Channel Operations
            {
                "name": "send_notification",
                "description": "Send notification via auto-selected channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipients": {
                            "type": "object",
                            "description": "Dictionary with 'email' and/or 'phone' keys",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Notification subject",
                        },
                        "message": {
                            "type": "string",
                            "description": "Notification message",
                        },
                        "channel": {
                            "type": "string",
                            "enum": ["email", "sms"],
                            "description": "Force specific channel",
                        },
                    },
                    "required": ["recipients", "subject", "message"],
                },
            },
            {
                "name": "send_alert",
                "description": "Send alert with severity-based routing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipients": {
                            "type": "object",
                            "description": "Dictionary with 'email', 'phone' keys",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "error", "critical"],
                            "description": "Alert severity",
                        },
                        "title": {
                            "type": "string",
                            "description": "Alert title",
                        },
                        "message": {
                            "type": "string",
                            "description": "Alert message",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata",
                        },
                    },
                    "required": ["recipients", "severity", "title", "message"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "communication",
            "requires": ["sendgrid", "twilio"],
        },
    }