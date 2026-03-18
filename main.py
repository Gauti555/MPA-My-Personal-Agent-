"""
main.py
────────
Entry point. Run this to start the agent.

Usage examples:
  python main.py                        # auto-discover topic, full run
  python main.py --topic "LoRA explained for practitioners"
  python main.py --dry-run              # scrape topics only, no LLM
  python main.py --no-notify            # run fully but skip Telegram
  python main.py --history              # show past runs and exit
  python main.py --check-topic          # show what topic.txt contains
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env before anything else
load_dotenv()


def setup_logging():
    """Configure loguru — coloured output to terminal + file."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "{message}"
        ),
        colorize=True,
    )
    Path("outputs").mkdir(exist_ok=True)
    logger.add(
        "outputs/pipeline.log",
        rotation="1 week",
        level="DEBUG",
        encoding="utf-8",
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Content Agent — AI blog writer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --topic "Flash Attention 3 explained"
  python main.py --dry-run
  python main.py --history
  python main.py --check-topic
        """,
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Manually specify the blog topic (skips auto-discovery)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and show trending topics only — no LLM, no writing",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Run the full pipeline but skip Telegram notification",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show last 5 pipeline runs and exit",
    )
    parser.add_argument(
        "--check-topic",
        action="store_true",
        help="Show current contents of topic.txt and exit",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    # ── --history ─────────────────────────────────────────────────
    if args.history:
        from tools.store import Store
        Store().summary()
        return

    # ── --check-topic ─────────────────────────────────────────────
    if args.check_topic:
        topic_file = Path("topic.txt")
        content = topic_file.read_text(encoding="utf-8").strip()
        if content:
            logger.info(f"topic.txt contains:\n\n  {content}\n")
        else:
            logger.info("topic.txt is empty — agent will auto-discover a topic")
        return

    # ── Load config ───────────────────────────────────────────────
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── --dry-run ─────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN — scraping topics only, no LLM calls")
        from tools.scraper import TrendScraper
        scraper = TrendScraper(cfg)
        topics = scraper.run()

        if not topics:
            logger.error("No topics found — check your config and network")
            return

        logger.info(f"\nTop {min(8, len(topics))} topics found today:\n")
        for i, t in enumerate(topics[:8], 1):
            logger.info(f"  {i:2}. [{t.source:25}] {t.title[:60]}")

        logger.info(
            f"\nTo write about a specific topic, add it to topic.txt\n"
            f"or run: python main.py --topic \"your topic here\""
        )
        return

    # ── --no-notify: temporarily disable Telegram ─────────────────
    if args.no_notify:
        os.environ["TELEGRAM_BOT_TOKEN"] = ""
        os.environ["TELEGRAM_CHAT_ID"] = ""
        logger.info("Telegram notifications disabled for this run")

    # ── Full pipeline ─────────────────────────────────────────────
    logger.info("Starting Content Agent pipeline...")

    if args.topic:
        logger.info(f"Manual topic provided: '{args.topic}'")

    from pipeline import run
    result = run(cfg, manual_topic=args.topic)

    # ── Final summary ─────────────────────────────────────────────
    if result.get("skipped"):
        logger.info(f"Run skipped: {result.get('reason')}")
        return

    paths = result.get("paths", {})
    score = result.get("quality_score", "?")

    logger.success(
        f"\n{'━'*50}\n"
        f"  Done!\n"
        f"  Topic:   {result.get('topic_title', '')[:55]}\n"
        f"  Quality: {score}/10\n"
        f"  Blog:    {paths.get('blog', '')}\n"
        f"  LinkedIn:{paths.get('linkedin', '')}\n"
        f"{'━'*50}"
    )

    if paths.get("blog"):
        logger.info("Check your Telegram for the draft review message.")


if __name__ == "__main__":
    main()