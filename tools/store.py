"""
tools/store.py
──────────────
Tracks every pipeline run in outputs/history.json.

Two jobs:
  1. Duplicate detection — prevents writing the same topic twice.
     If you ran a blog about "Flash Attention 3" last week, the agent
     won't pick that topic again this week even if it's still trending.

  2. Run history — keeps a log of every run so you can see what
     was generated, when, and whether it was sent to Telegram.

The file outputs/history.json is committed to your GitHub repo
so the history persists across GitHub Actions runs — no database needed.
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

HISTORY_FILE = Path("outputs/history.json")


class Store:
    def __init__(self, path: Path = HISTORY_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # ── Load / save ───────────────────────────────────────────────

    def _load(self) -> dict:
        """Load history from disk. Start fresh if file is missing or broken."""
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("history.json unreadable — starting fresh")
        return {"runs": [], "published_titles": []}

    def _save(self):
        """Write history to disk."""
        self.path.write_text(
            json.dumps(self._data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Duplicate detection ───────────────────────────────────────

    def is_duplicate(self, title: str) -> bool:
        """
        Returns True if a very similar topic was already written recently.

        Uses word overlap — if 55% or more words match a past title,
        it's considered a duplicate and the agent picks a different topic.

        Example:
          Past title:    "Flash Attention 3 cuts memory by 40%"
          New title:     "Flash Attention 3 reduces memory usage"
          Overlap:       Flash(1) Attention(1) 3(1) memory(1) = 4/6 = 67% → DUPLICATE

          Past title:    "Flash Attention 3 cuts memory by 40%"
          New title:     "Mamba beats transformers on long context tasks"
          Overlap:       0/6 = 0% → NOT duplicate → OK to write
        """
        words = set(title.lower().split())

        for past_title in self._data.get("published_titles", []):
            past_words = set(past_title.lower().split())
            overlap = len(words & past_words) / max(len(words), 1)
            if overlap > 0.55:
                logger.warning(
                    f"Duplicate detected ({overlap:.0%} overlap)\n"
                    f"  Past:    {past_title}\n"
                    f"  Current: {title}"
                )
                return True

        return False

    # ── Recording runs ────────────────────────────────────────────

    def record(
        self,
        topic_title: str,
        quality_score: int,
        blog_path: str,
        notified: bool = False,
        skipped: bool = False,
        skip_reason: str = "",
    ):
        """
        Save this pipeline run to history.json.

        Args:
            topic_title:   The blog topic that was selected
            quality_score: LLM quality score 1-10
            blog_path:     Path to the saved .md file
            notified:      True if Telegram notification was sent
            skipped:       True if the run was skipped (e.g. duplicate)
            skip_reason:   Why it was skipped
        """
        entry = {
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "topic": topic_title,
            "quality": quality_score,
            "blog_path": blog_path,
            "notified": notified,
            "skipped": skipped,
            "skip_reason": skip_reason,
        }

        self._data["runs"].append(entry)

        # Only add to duplicate-detection list if actually written
        if not skipped and quality_score > 0:
            self._data["published_titles"].append(topic_title)
            # Keep last 60 titles — enough to cover a year of weekly posts
            self._data["published_titles"] = self._data["published_titles"][-60:]

        self._save()
        logger.info(f"Run recorded in history.json")

    # ── Reporting ─────────────────────────────────────────────────

    def summary(self):
        """Print last 5 runs to the terminal log."""
        runs = self._data.get("runs", [])

        if not runs:
            logger.info("No runs recorded yet.")
            return

        logger.info(f"Total runs: {len(runs)}")
        logger.info("Last 5 runs:")

        for r in runs[-5:]:
            if r.get("skipped"):
                status = f"SKIPPED ({r.get('skip_reason', '?')})"
            elif r.get("notified"):
                status = "Sent to Telegram"
            else:
                status = f"Draft saved (Q={r.get('quality', '?')}/10)"

            logger.info(f"  [{r['date']}] {r['topic'][:55]} — {status}")

    @property
    def total_runs(self) -> int:
        return len(self._data.get("runs", []))