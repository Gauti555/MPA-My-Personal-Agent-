"""
agents/researcher.py
─────────────────────
Gathers supporting context for the chosen topic before writing.

Sources it searches:
  1. arXiv — finds related papers with their abstracts
  2. Source URL — fetches and extracts text from the original article
  3. HackerNews — finds engineer discussions about the topic

Then asks the LLM to synthesise everything into a clean research
brief that the writer agent uses to write the blog.

Better research = better blog. This is why blogs written by this
agent are more accurate than ones written from LLM knowledge alone.
"""

import re
from html.parser import HTMLParser

import arxiv
import httpx
from loguru import logger

from tools.hf_client import HFClient


RESEARCH_SYSTEM = """You are a technical research assistant specialising in AI and ML.
Given raw research notes, synthesise a clear factual brief for a writer.
Be specific — include paper names, dates, model names, benchmark numbers.
Avoid vague statements. Every sentence should contain a concrete fact."""

RESEARCH_PROMPT = """Topic to write about: {topic_title}
Article angle: {angle}

Raw research notes collected:
{notes}

Write a research brief (max 500 words) covering:
1. Core technical concepts the article must explain clearly
2. Most relevant recent papers or releases (names, dates, key results)
3. Real-world applications — who benefits and how
4. Specific numbers or benchmarks if available
5. One good Python code example idea that would illustrate the concept

Be factual and specific. Skip anything vague."""


class ResearchAgent:
    def __init__(self, llm: HFClient):
        self.llm = llm

    def research(self, topic_title: str, angle: str, source_url: str = "") -> dict:
        """
        Gather context from multiple sources and return a research brief.

        Returns:
            dict with 'brief' (string) and 'raw_length' (int)
        """
        logger.info(f"Researcher: gathering context for '{topic_title[:60]}'")

        notes = []

        # ── Source 1: arXiv related papers ───────────────────────
        arxiv_notes = self._search_arxiv(topic_title)
        if arxiv_notes:
            notes.append("=== Related arXiv Papers ===\n" + arxiv_notes)

        # ── Source 2: Original article page ──────────────────────
        if source_url and source_url.startswith("http"):
            page_notes = self._fetch_page(source_url)
            if page_notes:
                notes.append("=== Source Article Text ===\n" + page_notes)

        # ── Source 3: HackerNews discussions ─────────────────────
        hn_notes = self._search_hn(topic_title)
        if hn_notes:
            notes.append("=== Community Discussion (HN) ===\n" + hn_notes)

        combined = "\n\n".join(notes) if notes else f"Topic: {topic_title}"
        logger.info(f"Researcher: {len(combined)} chars of raw notes collected")

        # ── Ask LLM to synthesise a brief ────────────────────────
        brief = self.llm.generate(
            RESEARCH_PROMPT.format(
                topic_title=topic_title,
                angle=angle,
                notes=combined[:3000],
            ),
            system=RESEARCH_SYSTEM,
        )

        logger.info("Researcher: brief ready")
        return {"brief": brief, "raw_length": len(combined)}

    def _search_arxiv(self, query: str) -> str:
        """Find related papers on arXiv."""
        try:
            results = list(arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance,
            ).results())

            lines = [
                f"- {r.title} ({r.published.strftime('%Y-%m-%d')})\n"
                f"  {r.summary[:200]}"
                for r in results
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"arXiv research error: {e}")
            return ""

    def _fetch_page(self, url: str) -> str:
        """Fetch and extract plain text from a URL."""
        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                resp = client.get(
                    url,
                    headers={"User-Agent": "ContentAgent/1.0"},
                )
                resp.raise_for_status()

            # Strip HTML tags
            class _TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.chunks = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip and len(data.strip()) > 40:
                        self.chunks.append(data.strip())

            parser = _TextExtractor()
            parser.feed(resp.text)
            return " ".join(parser.chunks)[:1500]

        except Exception as e:
            logger.warning(f"Page fetch failed ({url}): {e}")
            return ""

    def _search_hn(self, query: str) -> str:
        """Find HackerNews comments about this topic."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "query": query,
                        "tags": "comment",
                        "hitsPerPage": 5,
                    },
                )
                resp.raise_for_status()

            hits = resp.json().get("hits", [])
            lines = [
                h["comment_text"][:200]
                for h in hits
                if h.get("comment_text")
            ]
            return "\n".join(lines[:3])

        except Exception as e:
            logger.warning(f"HN search error: {e}")
            return ""