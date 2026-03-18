# MPA — My Personal Agent

An autonomous AI content agent that scrapes trending AI/ML topics,
writes technical blog posts, and sends drafts to Telegram for review.

Built with Python, HuggingFace Inference API, and GitHub Actions.

## What it does

- Scrapes arXiv, Hacker News, and AI lab RSS feeds daily
- Uses Qwen2.5-7B LLM to pick the best topic and write a 1300-word blog
- Scores quality out of 10 before sending
- Delivers blog draft + LinkedIn post to Telegram for human review
- Runs automatically every Monday via GitHub Actions — zero maintenance

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Qwen/Qwen2.5-7B-Instruct via HuggingFace free API |
| Scraping | arXiv API, HackerNews Algolia API, RSS feeds |
| Notification | Telegram Bot API |
| Automation | GitHub Actions (cron schedule) |
| Language | Python 3.11 |

## Project structure
```
agents/          — topic selector, researcher, writer, editor
tools/           — scraper, LLM client, Telegram notifier, history store
prompts/         — all LLM prompts (easy to tune)
config/          — settings in config.yaml
outputs/         — generated blog drafts saved here
pipeline.py      — orchestrates all agents
main.py          — CLI entry point
```

## How to run locally
```bash
git clone https://github.com/Gauti555/MPA-My-Personal-Agent-.git
cd MPA-My-Personal-Agent-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your tokens
python main.py --dry-run
```

## Usage
```bash
python main.py                                    # auto-discover topic
python main.py --topic "explain LoRA fine-tuning" # manual topic
python main.py --dry-run                          # show trending topics only
python main.py --history                          # show past runs
```

## Secrets required

Add these in GitHub Settings → Secrets → Actions:

| Secret | Where to get it |
|---|---|
| HF_TOKEN | huggingface.co/settings/tokens |
| TELEGRAM_BOT_TOKEN | @BotFather on Telegram |
| TELEGRAM_CHAT_ID | api.telegram.org/bot{token}/getUpdates |