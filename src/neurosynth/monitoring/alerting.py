# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Alerting — Slack & PagerDuty webhook integration for drift/model alerts."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    LOG = "log"


class AlertPriority(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    title: str
    message: str
    priority: AlertPriority
    source: str = "neurosynth-monitoring"
    metadata: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AlertDispatcher:
    """Dispatches alerts to Slack, PagerDuty, and structured logs.

    Configuration via environment variables:
      NEUROSYNTH_SLACK_WEBHOOK — Slack incoming webhook URL
      NEUROSYNTH_PAGERDUTY_KEY — PagerDuty Events API v2 routing key
      NEUROSYNTH_ALERT_CHANNELS — Comma-separated channels (default: "log")
    """

    def __init__(
        self,
        slack_webhook: str | None = None,
        pagerduty_key: str | None = None,
        channels: list[AlertChannel] | None = None,
    ) -> None:
        self.slack_webhook = slack_webhook or os.getenv("NEUROSYNTH_SLACK_WEBHOOK", "")
        self.pagerduty_key = pagerduty_key or os.getenv("NEUROSYNTH_PAGERDUTY_KEY", "")

        if channels:
            self.channels = channels
        else:
            raw = os.getenv("NEUROSYNTH_ALERT_CHANNELS", "log")
            self.channels = [AlertChannel(c.strip()) for c in raw.split(",") if c.strip()]

    def dispatch(self, alert: Alert) -> dict[str, bool]:
        """Send alert to all configured channels. Returns success per channel."""
        results: dict[str, bool] = {}
        for ch in self.channels:
            try:
                if ch == AlertChannel.SLACK:
                    results["slack"] = self._send_slack(alert)
                elif ch == AlertChannel.PAGERDUTY:
                    results["pagerduty"] = self._send_pagerduty(alert)
                elif ch == AlertChannel.LOG:
                    results["log"] = self._send_log(alert)
            except Exception as e:
                logger.error("alert_dispatch_failed channel=%s error=%s", ch.value, e)
                results[ch.value] = False
        return results

    def _send_slack(self, alert: Alert) -> bool:
        """Send Slack incoming webhook message."""
        if not self.slack_webhook:
            logger.warning("slack_webhook_not_configured")
            return False

        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}
        color = {"info": "#3b82f6", "warning": "#f59e0b", "critical": "#ef4444"}

        payload = {
            "attachments": [{
                "color": color.get(alert.priority.value, "#6b7280"),
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{emoji.get(alert.priority.value, '')} {alert.title}"},
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": alert.message},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": f"*Source:* {alert.source} | *Priority:* {alert.priority.value} | *Time:* {alert.timestamp}"},
                        ],
                    },
                ],
            }],
        }

        if alert.metadata:
            meta_text = "\n".join(f"• *{k}:* {v}" for k, v in alert.metadata.items())
            payload["attachments"][0]["blocks"].insert(2, {
                "type": "section",
                "text": {"type": "mrkdwn", "text": meta_text},
            })

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.slack_webhook, json=payload)
                resp.raise_for_status()
            logger.info("slack_alert_sent title=%s", alert.title)
            return True
        except Exception as e:
            logger.error("slack_alert_failed error=%s", e)
            return False

    def _send_pagerduty(self, alert: Alert) -> bool:
        """Send PagerDuty Events API v2 alert."""
        if not self.pagerduty_key:
            logger.warning("pagerduty_key_not_configured")
            return False

        severity_map = {"info": "info", "warning": "warning", "critical": "critical"}

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[NeuroSynth] {alert.title}: {alert.message[:200]}",
                "severity": severity_map.get(alert.priority.value, "warning"),
                "source": alert.source,
                "timestamp": alert.timestamp,
                "custom_details": alert.metadata or {},
            },
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                )
                resp.raise_for_status()
            logger.info("pagerduty_alert_sent title=%s", alert.title)
            return True
        except Exception as e:
            logger.error("pagerduty_alert_failed error=%s", e)
            return False

    @staticmethod
    def _send_log(alert: Alert) -> bool:
        """Log the alert using structured logging."""
        log_fn = {
            AlertPriority.INFO: logger.info,
            AlertPriority.WARNING: logger.warning,
            AlertPriority.CRITICAL: logger.critical,
        }.get(alert.priority, logger.info)

        log_fn(
            "alert title=%s message=%s priority=%s source=%s metadata=%s",
            alert.title, alert.message, alert.priority.value, alert.source,
            json.dumps(alert.metadata or {}),
        )
        return True


def create_drift_alert(drift_report: Any) -> Alert:
    """Create an Alert from a DriftReport."""
    severity_map = {
        "NO_DRIFT": AlertPriority.INFO,
        "MINOR": AlertPriority.INFO,
        "WARNING": AlertPriority.WARNING,
        "CRITICAL": AlertPriority.CRITICAL,
    }

    raw_sev = getattr(drift_report, "overall_severity", "WARNING")
    sev = raw_sev.value if hasattr(raw_sev, "value") else str(raw_sev)

    return Alert(
        title=f"Data Drift Detected — {sev}",
        message=getattr(drift_report, "recommendation", "Drift detected in production data."),
        priority=severity_map.get(sev, AlertPriority.WARNING),
        source="neurosynth-drift-detector",
        metadata={
            "total_features": getattr(drift_report, "total_features", 0),
            "drifted_features": getattr(drift_report, "drifted_features", 0),
            "severity": sev,
            "timestamp": getattr(drift_report, "timestamp", ""),
        },
    )
