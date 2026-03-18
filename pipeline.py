"""
pipeline.py
────────────
Orchestrates all agents in sequence. Called by main.py.

Phase order:
  1.  Check for manual topic (topic.txt or --topic argument)
  2.  Scrape trending topics (if no manual topic)
  3.  Initialise LLM client
  4.  Select best topic
  5.  Research the topic
  6.  Write blog post + LinkedIn post
  7.  Edit + generate SEO metadata + quality score
  8.  Save all outputs to outputs/ folder
  9.  Quality gate — skip notify if score too low
  10. Send to Telegram for human review
  11. Record run in history.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger
from slugify import slugify

from tools.scraper import TrendScraper, read_manual_topic
from tools.hf_client import HFClient
from tools.notifier import Notifier
from tools.store import Store
from agents.topic_selector import TopicSelectorAgent
from agents.researcher import ResearchAgent
from agents.writer import WriterAgent
from agents.editor import EditorAgent


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_outputs(data: dict, output_dir: Path) -> dict:
    """
    Save blog, LinkedIn post, and metadata to the outputs folder.
    Returns dict of file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.utcnow().strftime("%Y-%m-%d")
    slug = slugify(data.get("topic_title", "post"))[:50]
    prefix = f"{date}_{slug}"

    # ── Blog markdown ─────────────────────────────────────────────
    blog_path = output_dir / f"{prefix}_blog.md"
    blog_path.write_text(data["blog"], encoding="utf-8")

    # ── LinkedIn post ─────────────────────────────────────────────
    li_path = output_dir / f"{prefix}_linkedin.txt"
    li_path.write_text(data["linkedin_post"], encoding="utf-8")

    # ── Full metadata JSON ────────────────────────────────────────
    meta_path = output_dir / f"{prefix}_meta.json"
    safe_data = {
        k: v for k, v in data.items()
        if k not in ("blog", "linkedin_post")
    }
    meta_path.write_text(
        json.dumps(safe_data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Outputs saved → outputs/{prefix}_*")
    return {
        "blog": str(blog_path),
        "linkedin": str(li_path),
        "meta": str(meta_path),
        "prefix": prefix,
    }


def run(cfg: dict, manual_topic: str = "") -> dict:
    """
    Run the full content generation pipeline.

    Args:
        cfg:          Loaded config dict from config/config.yaml
        manual_topic: Topic string from --topic CLI flag.
                      If empty, checks topic.txt, then auto-discovers.

    Returns:
        dict with all results — topic, paths, quality score, etc.
    """
    notifier = Notifier()
    store = Store()
    result = {}

    try:
        # ── Phase 1: Check for manual topic ──────────────────────
        logger.info("━━━ Phase 1 — Topic Input ━━━")

        # Priority: CLI flag > topic.txt > auto-discover
        topic_input = manual_topic.strip() if manual_topic else ""

        if not topic_input:
            # Check topic.txt file
            topic_file = cfg.get("topic_input", {}).get("topic_file", "topic.txt")
            topic_input = read_manual_topic(topic_file) or ""

        if topic_input:
            logger.info(f"Manual topic: '{topic_input[:80]}'")
        else:
            logger.info("No manual topic — will auto-discover from scrapers")

        # ── Phase 2: Scrape (only if no manual topic) ─────────────
        topics = []
        if not topic_input:
            logger.info("━━━ Phase 2 — Scraping ━━━")
            scraper = TrendScraper(cfg)
            topics = scraper.run()

            if not topics:
                msg = (
                    "No topics found from any scraper.\n"
                    "Add a topic to topic.txt to run without scrapers."
                )
                notifier.send_error(msg)
                raise RuntimeError(msg)

            logger.info(f"Top 5 scraped topics:")
            for i, t in enumerate(topics[:5], 1):
                logger.info(f"  {i}. {t}")
        else:
            logger.info("Phase 2 — Scraping skipped (manual topic provided)")

        # ── Phase 3: LLM init ─────────────────────────────────────
        logger.info("━━━ Phase 3 — LLM Init ━━━")
        llm_cfg = cfg.get("llm", {})
        llm = HFClient(
            model=llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 1500),
            temperature=llm_cfg.get("temperature", 0.7),
            timeout=llm_cfg.get("timeout", 120),
        )

        # ── Phase 4: Topic selection ──────────────────────────────
        logger.info("━━━ Phase 4 — Topic Selection ━━━")
        selector = TopicSelectorAgent(
            llm,
            max_candidates=cfg.get("pipeline", {}).get("max_topic_candidates", 5),
        )
        selection = selector.select(topics, manual_topic=topic_input)

        topic_title = selection["topic_title"]
        angle       = selection["angle"]
        key_points  = selection.get("key_points", [])
        source_url  = selection.get("source_url", "")

        # Duplicate check (skip for manual topics)
        if not selection.get("was_manual") and store.is_duplicate(topic_title):
            msg = f"Duplicate topic skipped: {topic_title}"
            logger.warning(msg)
            notifier.send_skipped("duplicate topic", topic_title)
            store.record(
                topic_title=topic_title,
                quality_score=0,
                blog_path="",
                skipped=True,
                skip_reason="duplicate",
            )
            return {"skipped": True, "reason": "duplicate", "topic": topic_title}

        result["topic_title"] = topic_title
        result["angle"] = angle

        # ── Phase 5: Research ─────────────────────────────────────
        logger.info("━━━ Phase 5 — Research ━━━")
        researcher = ResearchAgent(llm)
        research = researcher.research(topic_title, angle, source_url)
        result["research_brief"] = research["brief"]

        # ── Phase 6: Write ────────────────────────────────────────
        logger.info("━━━ Phase 6 — Writing ━━━")
        writer = WriterAgent(llm, cfg.get("linkedin", {}))
        blog = writer.write_blog(
            topic_title, angle, key_points, research["brief"]
        )
        linkedin_post = writer.write_linkedin(topic_title, angle, blog)

        result["blog"]          = blog
        result["linkedin_post"] = linkedin_post

        # ── Phase 7: Edit ─────────────────────────────────────────
        logger.info("━━━ Phase 7 — Editing ━━━")
        editor = EditorAgent(llm)
        edited = editor.edit(blog, topic_title)

        result["blog"]          = edited["blog"]
        result["metadata"]      = edited["metadata"]
        result["quality_score"] = edited["metadata"].get("quality_score", 0)

        logger.info(f"Quality score: {result['quality_score']}/10")

        # ── Phase 8: Save outputs ─────────────────────────────────
        logger.info("━━━ Phase 8 — Saving Outputs ━━━")
        output_dir = Path(cfg.get("pipeline", {}).get("output_dir", "outputs"))
        paths = save_outputs(result, output_dir)
        result["paths"] = paths

        # ── Phase 9: Quality gate ─────────────────────────────────
        min_score = cfg.get("pipeline", {}).get("min_quality_score", 6)

        if result["quality_score"] < min_score:
            logger.warning(
                f"Quality score {result['quality_score']}/10 is below "
                f"minimum {min_score}.\n"
                f"Draft saved to {paths['blog']} but NOT sent to Telegram.\n"
                f"Lower min_quality_score in config.yaml to send anyway."
            )
            store.record(
                topic_title=topic_title,
                quality_score=result["quality_score"],
                blog_path=paths["blog"],
                notified=False,
            )
            return result

        # ── Phase 10: Notify via Telegram ─────────────────────────
        logger.info("━━━ Phase 10 — Telegram Notification ━━━")
        notifier.send_draft_ready(
            topic_title=topic_title,
            quality_score=result["quality_score"],
            blog_preview=result["blog"][:500],
            linkedin_post=result["linkedin_post"],
            blog_file_path=paths["blog"],
        )

        # ── Phase 11: Record run ──────────────────────────────────
        store.record(
            topic_title=topic_title,
            quality_score=result["quality_score"],
            blog_path=paths["blog"],
            notified=True,
        )

        logger.success("━━━ Pipeline complete ━━━")
        store.summary()
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        notifier.send_error(str(e))
        raise