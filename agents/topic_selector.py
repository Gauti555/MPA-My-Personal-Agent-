"""
agents/topic_selector.py
─────────────────────────
Picks the best topic to write about.

Priority order:
  1. Manual topic from topic.txt or CLI --topic flag
  2. Auto-discovered topics from scrapers (ranked by score)

When auto-discovering, sends top 5 candidates to the LLM and asks
it to pick the best one based on novelty, depth, and relevance.
Returns structured data: title, angle, key points, reason.
"""

import json
from typing import List, Optional

from loguru import logger

from tools.hf_client import HFClient
from tools.scraper import Topic


SELECTOR_SYSTEM = """You are an expert AI/ML content strategist.
Your job is to pick the BEST topic for a technical blog post
targeting ML engineers and AI practitioners.

A good topic is:
- NEW: published or trending in the last 7 days
- DEEP: has real technical substance, not just announcement news
- PRACTICAL: engineers can learn something actionable
- SPECIFIC: not vague like "AI is improving" but concrete like
  "Qwen2.5 uses grouped query attention to reduce KV cache by 60%"

Respond ONLY with valid JSON. No explanation. No markdown. No extra text."""

SELECTOR_PROMPT = """Here are today's trending AI/ML topics:

{topics_text}

Pick the single BEST topic for a 1300-word technical blog post.

Respond with this exact JSON structure:
{{
  "selected_index": <integer, 0-based>,
  "topic_title": "<clear specific blog-friendly title>",
  "angle": "<one sentence: the specific hook or lens for this article>",
  "key_points": [
    "<technical point 1>",
    "<technical point 2>",
    "<technical point 3>",
    "<technical point 4>"
  ],
  "reason": "<one sentence: why this beats the other options>"
}}"""


class TopicSelectorAgent:
    def __init__(self, llm: HFClient, max_candidates: int = 5):
        self.llm = llm
        self.max_candidates = max_candidates

    def select(
        self,
        topics: List[Topic],
        manual_topic: Optional[str] = None,
    ) -> dict:
        """
        Select the best topic to write about.

        If manual_topic is provided (from topic.txt or --topic flag),
        skip the LLM selection and use it directly.

        Returns dict with: topic_title, angle, key_points, reason,
                           source_url, was_manual
        """

        # ── Manual override — always wins ─────────────────────────
        if manual_topic and manual_topic.strip():
            logger.info(f"Using manual topic: '{manual_topic[:80]}'")
            return {
                "topic_title": manual_topic.strip(),
                "angle": "A technical deep dive into this topic for ML practitioners",
                "key_points": [
                    "Background and context",
                    "How it works technically",
                    "Practical implications for engineers",
                    "Key takeaways and next steps",
                ],
                "reason": "Manually specified by user",
                "source_url": "",
                "was_manual": True,
            }

        # ── Auto-selection from scraped topics ────────────────────
        if not topics:
            raise ValueError(
                "No topics found from any scraper and no manual topic provided.\n"
                "Check your scraper settings or add a topic to topic.txt"
            )

        candidates = topics[: self.max_candidates]

        topics_text = "\n\n".join(
            f"[{i}] Source: {t.source}\n"
            f"    Title: {t.title}\n"
            f"    Summary: {t.summary[:200]}\n"
            f"    Score: {t.score} | Published: {t.published}"
            for i, t in enumerate(candidates)
        )

        logger.info(f"TopicSelector: asking LLM to pick from {len(candidates)} candidates...")

        raw = self.llm.generate(
            SELECTOR_PROMPT.format(topics_text=topics_text),
            system=SELECTOR_SYSTEM,
        )

        result = self._parse_json(raw, candidates)
        logger.info(f"Selected: {result['topic_title']}")
        logger.info(f"Reason:   {result['reason']}")
        return result

    def _parse_json(self, raw: str, candidates: List[Topic]) -> dict:
        """Extract JSON from LLM response. Falls back gracefully."""
        # Find the JSON block
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
                idx = int(data.get("selected_index", 0))
                # Clamp index to valid range
                idx = max(0, min(idx, len(candidates) - 1))
                data["source_url"] = candidates[idx].url
                data["was_manual"] = False
                return data
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback — use top scored topic
        logger.warning("TopicSelector: JSON parse failed — using top topic as fallback")
        return {
            "selected_index": 0,
            "topic_title": candidates[0].title,
            "angle": "A technical deep dive for ML practitioners",
            "key_points": [
                "Background and context",
                "Technical details",
                "Practical implications",
                "Key takeaways",
            ],
            "reason": "Highest scored topic (fallback)",
            "source_url": candidates[0].url,
            "was_manual": False,
        }