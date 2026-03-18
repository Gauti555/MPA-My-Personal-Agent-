"""
agents/editor.py
─────────────────
Reviews the blog post and generates SEO metadata + quality score.

What it produces:
  - seo_title: optimised title under 60 chars
  - meta_description: 150-160 char description for search engines
  - slug: URL-friendly version of title
  - tags: 5 relevant tags for Medium / Dev.to
  - reading_time_minutes: estimated reading time
  - quality_score: 1-10 score (pipeline skips sending if too low)
  - quality_notes: one sentence on what could be improved

The quality_score is used as a gate — if the blog scores below
the minimum set in config.yaml, Telegram is not notified and the
draft is saved locally but not sent for review.
"""

import json
from loguru import logger
from tools.hf_client import HFClient


EDITOR_SYSTEM = """You are a technical editor and SEO specialist for AI/ML content.
Respond ONLY with valid JSON. No markdown. No explanation. No extra text.
Your JSON must be parseable by Python's json.loads()."""

EDITOR_PROMPT = """Review this blog post and return SEO metadata and a quality score.

BLOG POST (first 2500 chars):
{blog_content}

Return this EXACT JSON and nothing else:
{{
  "seo_title": "<optimised title, max 60 chars, includes main keyword>",
  "meta_description": "<compelling description, 150-160 chars, includes keyword>",
  "slug": "<url-friendly-slug-with-hyphens-only>",
  "tags": ["<tag1>", "<tag2>", "<tag3>", "<tag4>", "<tag5>"],
  "reading_time_minutes": <integer>,
  "quality_score": <integer 1-10>,
  "quality_notes": "<one sentence on what could be improved>"
}}

Quality score guide:
  9-10: Exceptional — clear, accurate, original insight, great code example
  7-8:  Good — solid technical content, well structured
  5-6:  Acceptable — some depth but could be more specific
  3-4:  Weak — too vague or too short
  1-2:  Poor — mostly filler, no real technical value"""


class EditorAgent:
    def __init__(self, llm: HFClient):
        self.llm = llm

    def edit(self, blog: str, topic_title: str) -> dict:
        """
        Generate SEO metadata and quality score for the blog.

        Returns:
            dict with 'metadata' (dict) and 'blog' (string with improvements applied)
        """
        logger.info("Editor: reviewing blog quality and generating SEO metadata...")

        raw = self.llm.generate(
            EDITOR_PROMPT.format(blog_content=blog[:2500]),
            system=EDITOR_SYSTEM,
        )

        metadata = self._parse_json(raw)
        improved_blog = self._apply_improvements(blog, metadata)

        score = metadata.get("quality_score", 0)
        notes = metadata.get("quality_notes", "")
        logger.info(f"Editor: quality score {score}/10 — {notes}")

        return {"metadata": metadata, "blog": improved_blog}

    def _apply_improvements(self, blog: str, metadata: dict) -> str:
        """
        Apply light structural improvements to the blog.
        Does not rewrite content — just fixes structure.
        """
        lines = blog.strip().split("\n")
        seo_title = metadata.get("seo_title", "")
        reading_time = metadata.get("reading_time_minutes", 5)

        # Replace or insert H2 title with SEO-optimised version
        if seo_title:
            if lines and lines[0].startswith("## "):
                lines[0] = f"## {seo_title}"
            elif lines and lines[0].startswith("# "):
                lines[0] = f"# {seo_title}"
            else:
                lines.insert(0, f"## {seo_title}")

        # Add reading time badge after the title
        if len(lines) > 1 and not lines[1].strip().startswith("*"):
            lines.insert(1, f"\n*{reading_time} min read*\n")

        return "\n".join(lines)

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response with fallback defaults."""
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Editor: JSON parse failed — using fallback metadata")
        return {
            "seo_title": "Latest AI/ML Insights",
            "meta_description": "A technical deep dive into recent AI and ML developments.",
            "slug": "ai-ml-insights",
            "tags": ["AI", "Machine Learning", "Deep Learning", "LLM", "Python"],
            "reading_time_minutes": 6,
            "quality_score": 6,
            "quality_notes": "Metadata could not be generated — check LLM output",
        }