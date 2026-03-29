"""Tavily web search (optional API key)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": max_results},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        return list(data.get("results") or [])
    except Exception:
        return []


def web_search_fallback(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """DuckDuckGo text search when Tavily is unavailable."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("href") or "",
                    "content": (r.get("body") or "")[:800],
                }
            )
        return out
    except Exception:
        return []


def fetch_url(url: str, max_chars: int = 8000) -> str:
    try:
        r = httpx.get(url, timeout=20.0, follow_redirects=True, headers={"User-Agent": "autoresearcher/0.1"})
        r.raise_for_status()
        return (r.text or "")[:max_chars]
    except Exception as e:
        return f"[fetch failed: {e}]"
