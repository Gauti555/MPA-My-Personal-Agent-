"""
tools/scraper.py
────────────────
Collects trending AI/ML topics from 4 free sources:
  1. arXiv        — latest research papers (no API key)
  2. Hacker News  — trending AI discussions (no API key)
  3. RSS feeds    — official AI lab blogs (no API key)
  4. Google Trends — what people are searching (no API key)

Also handles the manual topic override from topic.txt.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import arxiv
import feedparser
import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Data model ───────────────────────────────────────────────────────

@dataclass
class Topic:
    """A single trending topic from any source."""
    title: str
    summary: str
    source: str
    url: str
    score: float = 0.0
    published: str = ""
    keywords: List[str] = field(default_factory=list)

    def __str__(self):
        return f"[{self.source}] {self.title[:65]} (score={self.score:.0f})"


# ── Manual override ──────────────────────────────────────────────────

def read_manual_topic(topic_file: str = "topic.txt") -> Optional[str]:
    """
    Check if the user has manually specified a topic in topic.txt.
    If found, read it, clear the file, and return the topic string.
    Returns None if the file is empty or doesn't exist.
    """
    path = Path(topic_file)

    if not path.exists():
        return None

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        return None

    # Clear the file so the same topic isn't used twice
    path.write_text("", encoding="utf-8")
    logger.info(f"Manual topic found: '{content[:80]}'")
    return content


# ── Source 1: arXiv ──────────────────────────────────────────────────

class ArxivScraper:
    """Fetches latest AI/ML papers from arXiv. No API key needed."""

    def __init__(self, categories: List[str], max_results: int = 10, days_back: int = 7):
        self.categories = categories
        self.max_results = max_results
        self.days_back = days_back

    def fetch(self) -> List[Topic]:
        topics = []
        query = " OR ".join(f"cat:{c}" for c in self.categories)
        cutoff = datetime.utcnow() - timedelta(days=self.days_back)

        logger.info(f"arXiv: searching {self.categories}...")

        try:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            for paper in search.results():
                # Skip papers older than cutoff
                published = paper.published.replace(tzinfo=None)
                if published < cutoff:
                    continue

                topics.append(Topic(
                    title=paper.title.strip(),
                    summary=paper.summary[:400].replace("\n", " "),
                    source="arxiv",
                    url=paper.entry_id,
                    score=10.0,
                    published=published.strftime("%Y-%m-%d"),
                    keywords=paper.categories[:3],
                ))

        except Exception as e:
            logger.warning(f"arXiv error: {e}")

        logger.info(f"arXiv: found {len(topics)} papers")
        return topics


# ── Source 2: Hacker News ────────────────────────────────────────────

class HackerNewsScraper:
    """
    Fetches trending AI stories from HN via Algolia API.
    Completely free — no signup, no API key.
    """

    API_URL = "https://hn.algolia.com/api/v1/search"

    def __init__(self, queries: List[str], min_score: int = 50, max_results: int = 20):
        self.queries = queries
        self.min_score = min_score
        self.max_results = max_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def fetch(self) -> List[Topic]:
        topics = []
        seen_urls = set()

        logger.info("HackerNews: fetching stories...")

        with httpx.Client(timeout=15) as client:
            for query in self.queries[:3]:   # limit to 3 queries
                try:
                    resp = client.get(self.API_URL, params={
                        "query": query,
                        "tags": "story",
                        "numericFilters": f"points>={self.min_score}",
                        "hitsPerPage": self.max_results // 3,
                    })
                    resp.raise_for_status()

                    for hit in resp.json().get("hits", []):
                        url = hit.get("url") or \
                              f"https://news.ycombinator.com/item?id={hit.get('objectID')}"

                        # Skip duplicates
                        if not hit.get("title") or url in seen_urls:
                            continue
                        seen_urls.add(url)

                        summary = (hit.get("story_text") or hit["title"])[:300]
                        summary = summary.replace("\n", " ")

                        topics.append(Topic(
                            title=hit["title"],
                            summary=summary,
                            source="hackernews",
                            url=url,
                            score=float(hit.get("points", 0)),
                            published=(hit.get("created_at") or "")[:10],
                            keywords=[query],
                        ))

                    time.sleep(0.5)   # be polite to the API

                except Exception as e:
                    logger.warning(f"HN query '{query}' failed: {e}")

        logger.info(f"HackerNews: found {len(topics)} stories")
        return topics


# ── Source 3: RSS Feeds ──────────────────────────────────────────────

class RSSFeedScraper:
    """
    Reads RSS feeds from major AI lab blogs.
    feedparser handles everything — no API keys, no authentication.
    """

    def __init__(self, feeds: List[dict], days_back: int = 7):
        """
        feeds: list of dicts with 'name' and 'url' keys
               e.g. [{"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml"}]
        """
        self.feeds = feeds
        self.days_back = days_back

    def fetch(self) -> List[Topic]:
        topics = []
        cutoff = datetime.utcnow() - timedelta(days=self.days_back)

        logger.info(f"RSS: reading {len(self.feeds)} feeds...")

        for feed_info in self.feeds:
            name = feed_info["name"]
            url = feed_info["url"]

            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:5]:   # latest 5 from each feed
                    # Parse published date
                    pub_date = self._parse_date(entry)
                    if pub_date and pub_date < cutoff:
                        continue

                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    # Extract summary — strip HTML tags
                    raw_summary = (
                        entry.get("summary") or
                        entry.get("description") or
                        title
                    )
                    summary = self._strip_html(raw_summary)[:400]

                    topics.append(Topic(
                        title=title,
                        summary=summary,
                        source=f"rss/{name}",
                        url=entry.get("link", url),
                        score=15.0,   # RSS from official sources = high priority
                        published=pub_date.strftime("%Y-%m-%d") if pub_date else "",
                        keywords=[name],
                    ))

            except Exception as e:
                logger.warning(f"RSS feed '{name}' failed: {e}")

        logger.info(f"RSS: found {len(topics)} posts")
        return topics

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        """Try multiple date fields that RSS entries might use."""
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            val = entry.get(field)
            if val:
                try:
                    return datetime(*val[:6])
                except Exception:
                    continue
        return None

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        try:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
        except Exception:
            return text


# ── Source 4: Google Trends ──────────────────────────────────────────

class GoogleTrendsScraper:
    """
    Queries Google Trends for rising AI/ML search terms.
    Uses pytrends — free, no API key.
    Note: Google sometimes rate-limits this. Failures are silent.
    """

    def __init__(self, keywords: List[str], geo: str = ""):
        self.keywords = keywords
        self.geo = geo

    def fetch(self) -> List[Topic]:
        topics = []

        logger.info("Google Trends: checking rising topics...")

        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))

            # Check keywords in batches of 5 (Google Trends limit)
            for i in range(0, len(self.keywords), 5):
                batch = self.keywords[i:i+5]
                try:
                    pytrends.build_payload(batch, geo=self.geo, timeframe="now 7-d")
                    related = pytrends.related_queries()

                    for keyword in batch:
                        data = related.get(keyword, {})
                        rising = data.get("rising")

                        if rising is not None and not rising.empty:
                            for _, row in rising.head(3).iterrows():
                                query_text = str(row.get("query", ""))
                                if not query_text:
                                    continue

                                topics.append(Topic(
                                    title=f"Trending: {query_text}",
                                    summary=f"Rising search term related to '{keyword}': {query_text}",
                                    source="google_trends",
                                    url=f"https://trends.google.com/trends/explore?q={query_text.replace(' ', '+')}",
                                    score=float(row.get("value", 50)),
                                    published=datetime.utcnow().strftime("%Y-%m-%d"),
                                    keywords=[keyword],
                                ))

                    time.sleep(1)   # avoid rate limiting

                except Exception as e:
                    logger.warning(f"Trends batch {batch} failed: {e}")

        except ImportError:
            logger.warning("pytrends not installed — skipping Google Trends")
        except Exception as e:
            logger.warning(f"Google Trends error: {e}")

        logger.info(f"Google Trends: found {len(topics)} rising topics")
        return topics


# ── Main Aggregator ──────────────────────────────────────────────────

class TrendScraper:
    """
    Runs all enabled scrapers, merges results,
    normalizes scores to 0-100, and returns ranked topics.
    """

    def __init__(self, cfg: dict):
        self.scrapers = []
        sources = cfg.get("scraper", {})

        if sources.get("arxiv", {}).get("enabled", True):
            ax = sources["arxiv"]
            self.scrapers.append(ArxivScraper(
                categories=ax.get("categories", ["cs.AI", "cs.LG"]),
                max_results=ax.get("max_results", 10),
                days_back=ax.get("days_back", 7),
            ))

        if sources.get("hackernews", {}).get("enabled", True):
            hn = sources["hackernews"]
            self.scrapers.append(HackerNewsScraper(
                queries=hn.get("queries", ["LLM", "machine learning"]),
                min_score=hn.get("min_score", 50),
                max_results=hn.get("max_results", 20),
            ))

        if sources.get("rss", {}).get("enabled", True):
            rss = sources["rss"]
            self.scrapers.append(RSSFeedScraper(
                feeds=rss.get("feeds", []),
                days_back=rss.get("days_back", 7),
            ))

        if sources.get("google_trends", {}).get("enabled", True):
            gt = sources["google_trends"]
            self.scrapers.append(GoogleTrendsScraper(
                keywords=gt.get("keywords", ["AI", "LLM"]),
                geo=gt.get("geo", ""),
            ))

    def run(self) -> List[Topic]:
        """Run all scrapers and return normalized, ranked topics."""
        all_topics: List[Topic] = []

        for scraper in self.scrapers:
            try:
                results = scraper.fetch()
                all_topics.extend(results)
            except Exception as e:
                logger.error(f"{scraper.__class__.__name__} crashed: {e}")

        if not all_topics:
            return []

        # Normalize all scores to 0-100
        max_score = max(t.score for t in all_topics) or 1.0
        for t in all_topics:
            t.score = round((t.score / max_score) * 100, 1)

        # Sort best first
        ranked = sorted(all_topics, key=lambda t: t.score, reverse=True)

        logger.info(f"Total topics collected: {len(ranked)}")
        return ranked