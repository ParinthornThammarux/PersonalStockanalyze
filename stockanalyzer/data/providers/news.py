"""Headlines from Google News RSS: no API key, and it covers Thai-language news that Yahoo does not."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from stockanalyzer.data.providers.base import ProviderError

# one edition per language: Thai for SET symbols, US English for the rest
EDITIONS = {
    "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    "th": {"hl": "th", "gl": "TH", "ceid": "TH:th"},
}


def _published(text: str | None) -> pd.Timestamp:
    try:
        return pd.Timestamp(parsedate_to_datetime(text)).tz_convert("UTC")
    except (TypeError, ValueError):
        return pd.NaT


def parse_rss(content: bytes | str) -> list[dict]:
    """RSS items -> ``title`` (without the trailing " - Source" Google appends), ``source``, ``url``, ``published``."""
    items = []
    for item in ET.fromstring(content).iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        if title:
            items.append({"title": title, "source": source, "url": (item.findtext("link") or "").strip(),
                          "published": _published(item.findtext("pubDate"))})
    return items


def newest_first(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda i: i["published"].value if pd.notna(i["published"]) else float("-inf"), reverse=True)


class GoogleNewsProvider:
    name = "google_news"
    URL = "https://news.google.com/rss/search"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def headlines(self, query: str, *, lang: str = "en", limit: int = 20) -> list[dict]:
        params = {"q": query, **EDITIONS.get(lang, EDITIONS["en"])}
        try:
            resp = requests.get(self.URL, params=params, timeout=self.timeout,
                                headers={"User-Agent": "Mozilla/5.0 (stockanalyzer)"})
            resp.raise_for_status()
            items = parse_rss(resp.content)
        except (requests.RequestException, ET.ParseError) as exc:
            raise ProviderError(f"news for {query!r}: {exc}") from exc
        return newest_first(items)[:limit]
