"""
agents/writer.py
─────────────────
Writes the full blog post and LinkedIn summary using the LLM.

Blog post:
  - 1200-1400 words
  - Markdown format with headings
  - Starts with a strong hook (not "In this article...")
  - Includes one Python code example
  - Ends with Key Takeaways section

LinkedIn post:
  - Max 2500 characters
  - Opens with a bold claim or surprising fact
  - Short punchy paragraphs
  - Ends with a question + hashtags
"""

from loguru import logger
from tools.hf_client import HFClient


BLOG_SYSTEM = """You are a senior ML engineer and technical writer.
You write clear, insightful articles for other engineers.
Style: direct, precise, opinionated — like a great engineering blog.
No filler. No hype. Every sentence earns its place.
Format: Markdown."""

BLOG_PROMPT = """Write a complete technical blog post.

TOPIC: {topic_title}
ANGLE: {angle}

KEY POINTS TO COVER:
{key_points}

RESEARCH CONTEXT:
{research_brief}

REQUIREMENTS:
- Length: 1200 to 1400 words
- Format: Markdown with ## and ### headings
- First line: a hook — surprising fact, bold claim, or sharp question
  Do NOT start with "In this article", "Today we", or "Introduction"
- Include exactly one Python code block (```python) as a practical example
- Explain WHY this matters for engineers in production systems
- Final section must be "## Key Takeaways" with 4-5 bullet points
- Tone: senior engineer explaining to peers — no buzzwords, no hype
- Do NOT include an H1 title — the platform adds it automatically

Write the full blog post now:"""

LINKEDIN_SYSTEM = """You write high-performing LinkedIn posts for the AI/ML community.
Your posts feel like a thoughtful engineer sharing a real insight.
Short punchy sentences. Genuine technical value. No corporate speak."""

LINKEDIN_PROMPT = """Write a LinkedIn post based on this blog article.

ARTICLE TITLE: {topic_title}
ARTICLE ANGLE: {angle}

ARTICLE CONTENT (first 1500 chars):
{blog_excerpt}

REQUIREMENTS:
- Maximum 2500 characters total
- Line 1: a BOLD CLAIM or SURPRISING FACT — NOT "I just published"
- Blank line between every paragraph
- 4 to 6 short paragraphs
- Second to last paragraph: what engineers should do with this info
- Last paragraph: one open question to spark comments
- Final line only: exactly these hashtags: {hashtags}

Write the LinkedIn post now:"""


class WriterAgent:
    def __init__(self, llm: HFClient, linkedin_cfg: dict):
        self.llm = llm
        self.hashtags = " ".join(linkedin_cfg.get("hashtags", ["#AI", "#MachineLearning"]))

    def write_blog(
        self,
        topic_title: str,
        angle: str,
        key_points: list,
        research_brief: str,
    ) -> str:
        """
        Write the full blog post. Returns Markdown string.
        """
        logger.info(f"Writer: drafting blog — '{topic_title[:60]}'")

        blog = self.llm.generate(
            BLOG_PROMPT.format(
                topic_title=topic_title,
                angle=angle,
                key_points="\n".join(f"- {p}" for p in key_points),
                research_brief=research_brief[:2000],
            ),
            system=BLOG_SYSTEM,
        )

        # Ensure there is at least a markdown heading
        if not blog.strip().startswith("#"):
            blog = f"## {topic_title}\n\n{blog}"

        word_count = len(blog.split())
        logger.info(f"Writer: blog done — {word_count} words")
        return blog

    def write_linkedin(
        self,
        topic_title: str,
        angle: str,
        blog: str,
    ) -> str:
        """
        Write the LinkedIn post. Returns plain text string.
        """
        logger.info("Writer: drafting LinkedIn post...")

        post = self.llm.generate(
            LINKEDIN_PROMPT.format(
                topic_title=topic_title,
                angle=angle,
                blog_excerpt=blog[:1500],
                hashtags=self.hashtags,
            ),
            system=LINKEDIN_SYSTEM,
        )

        # Append hashtags if LLM forgot them
        if "#" not in post:
            post = post.strip() + f"\n\n{self.hashtags}"

        logger.info(f"Writer: LinkedIn post done — {len(post)} chars")
        return post