"""
tools/notifier.py
──────────────────
Sends the generated blog draft to your Telegram for review.

What you receive on Telegram after each run:
  1. Quality score and topic title
  2. First 500 characters of the blog (preview)
  3. The full LinkedIn post (ready to copy-paste)
  4. The full blog saved as a .txt message or file

You then decide: post it to Medium or discard it.

Also handles the /topic command so you can message the bot
with a topic request and it queues it for the next run.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger


class Notifier:
    """Sends Telegram messages and files for human review."""

    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.warning(
                "Telegram not configured — notifications disabled.\n"
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    # ── Internal send methods ─────────────────────────────────────

    def _send_text(self, text: str) -> bool:
        """Send a plain text message to Telegram."""
        if not self.enabled:
            return False
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = httpx.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Telegram text send failed: {e}")
            return False

    def _send_file(self, file_path: str, caption: str = "") -> bool:
        """Send a file (e.g. the blog .md) to Telegram."""
        if not self.enabled:
            return False
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.token}/sendDocument"
            with open(file_path, "rb") as f:
                resp = httpx.post(url, data={
                    "chat_id": self.chat_id,
                    "caption": caption[:1000],
                }, files={"document": f}, timeout=30)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Telegram file send failed: {e}")
            return False

    # ── Public notification methods ───────────────────────────────

    def send_draft_ready(
        self,
        topic_title: str,
        quality_score: int,
        blog_preview: str,
        linkedin_post: str,
        blog_file_path: str,
    ) -> None:
        """
        Send the complete review package to Telegram.
        Called after every successful pipeline run.

        The human receives:
          Message 1 — summary card with score + blog preview
          Message 2 — full LinkedIn post (ready to copy)
          File      — the complete blog as a .md file
        """
        if not self.enabled:
            logger.info("Telegram disabled — skipping notification")
            logger.info(f"Blog saved to: {blog_file_path}")
            return

        # ── Message 1: Summary card ───────────────────────────────
        score_bar = "★" * quality_score + "☆" * (10 - quality_score)
        summary = (
            f"<b>Content Agent — New Draft Ready</b>\n\n"
            f"<b>Topic:</b> {topic_title}\n"
            f"<b>Quality:</b> {score_bar} ({quality_score}/10)\n\n"
            f"<b>Blog preview:</b>\n"
            f"<i>{blog_preview[:500]}...</i>\n\n"
            f"Review the full blog in the file below.\n"
            f"Copy the LinkedIn post from the next message."
        )
        self._send_text(summary)
        logger.info("Telegram: sent summary card")

        # ── Message 2: LinkedIn post ──────────────────────────────
        linkedin_msg = (
            f"<b>LinkedIn Post — ready to copy:</b>\n\n"
            f"{linkedin_post[:2500]}"
        )
        self._send_text(linkedin_msg)
        logger.info("Telegram: sent LinkedIn post")

        # ── File: Full blog markdown ──────────────────────────────
        if Path(blog_file_path).exists():
            self._send_file(
                blog_file_path,
                caption=f"Full blog: {topic_title}"
            )
            logger.info("Telegram: sent blog file")

        logger.success("Telegram: all notifications sent")

    def send_error(self, error_message: str) -> None:
        """Send a pipeline error alert to Telegram."""
        msg = (
            f"<b>Content Agent — Pipeline Error</b>\n\n"
            f"<code>{error_message[:500]}</code>"
        )
        self._send_text(msg)

    def send_skipped(self, reason: str, topic: str = "") -> None:
        """Notify when a run was skipped (e.g. duplicate topic)."""
        msg = (
            f"<b>Content Agent — Run Skipped</b>\n\n"
            f"<b>Reason:</b> {reason}\n"
            + (f"<b>Topic:</b> {topic}" if topic else "")
        )
        self._send_text(msg)

    # ── Topic command handler ─────────────────────────────────────

    def check_for_topic_command(self, topic_file: str = "topic.txt") -> Optional[str]:
        """
        Check if the user sent a /topic command to the bot via Telegram.
        If found, saves it to topic.txt and returns the topic string.

        This lets you message your bot from your phone:
          You:  /topic explain how LoRA fine-tuning works
          Bot:  Topic queued for next run!

        Returns the topic string if a command was found, else None.
        """
        if not self.enabled:
            return None

        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            resp = httpx.get(url, params={"limit": 10}, timeout=10)
            resp.raise_for_status()
            updates = resp.json().get("result", [])

            for update in reversed(updates):
                msg = update.get("message", {})
                text = msg.get("text", "")

                if text.startswith("/topic "):
                    topic = text[7:].strip()
                    if topic:
                        # Save to topic.txt so the pipeline picks it up
                        Path(topic_file).write_text(topic, encoding="utf-8")

                        # Acknowledge to the user
                        self._send_text(
                            f"Topic queued for next run!\n\n"
                            f"<b>{topic}</b>"
                        )
                        logger.info(f"Topic command received: '{topic}'")
                        return topic

        except Exception as e:
            logger.warning(f"Telegram topic check failed: {e}")

        return None