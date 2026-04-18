from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


def _run_async(coro):
    """Run an async coroutine from sync code, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class LibraryRetriever:
    """MCP-based library retriever.

    Calls web_search / web_parse tools on a remote MCP server
    (default: http://127.0.0.1:8002/mcp).
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = self._load_project_config()
        self._mcp_url = self._resolve_mcp_url()
        self._search_defaults = self._load_web_defaults()

    def _load_project_config(self) -> Dict[str, Any]:
        path = self.project_root / "config.yaml"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_mcp_url(self) -> str:
        library_cfg = ((self.config.get("landau") or {}).get("library_config")) or {}
        configured = library_cfg.get("mcp_url")
        if configured:
            return str(configured).rstrip("/")
        return "http://127.0.0.1:8002/mcp"

    def _load_web_defaults(self) -> Dict[str, Any]:
        defaults = {
            "search_region": "us",
            "search_lang": "en",
            "parse_model": "DeepSeek/DeepSeek-V3-0324",
        }
        library_cfg = ((self.config.get("landau") or {}).get("library_config")) or {}
        defaults.update(
            {
                "search_region": library_cfg.get("search_region", defaults["search_region"]),
                "search_lang": library_cfg.get("search_lang", defaults["search_lang"]),
                "parse_model": library_cfg.get("parse_model", defaults["parse_model"]),
            }
        )
        return defaults

    # ── MCP call ───────────────────────────────────────────────────────
    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Open a short-lived MCP session, call one tool, return parsed result."""
        async with streamablehttp_client(self._mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if not result.content:
                    return {}
                text = result.content[0].text
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return text

    def _call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Synchronous wrapper around _call_mcp_tool."""
        return _run_async(self._call_mcp_tool(tool_name, arguments))

    # ── Public API (same interface as before) ──────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sources: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        return self.search(query=query, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: int = 5,
        region: str | None = None,
        lang: str | None = None,
        depth: int = 0,
    ) -> List[Dict[str, Any]]:
        result = self._call_tool_sync("web_search", {
            "query": query,
            "top_k": int(top_k),
            "region": region or self._search_defaults["search_region"],
            "lang": lang or self._search_defaults["search_lang"],
            "depth": int(depth),
        })

        if isinstance(result, dict):
            items = result.get("organic", [])
        elif isinstance(result, list):
            items = result
        else:
            return []

        normalized: List[Dict[str, Any]] = []
        for item in items[:top_k]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "text": item.get("snippet", ""),
                }
            )
        return normalized

    def parse(
        self,
        link: str,
        user_prompt: str,
        llm: str | None = None,
    ) -> Dict[str, Any]:
        result = self._call_tool_sync("web_parse", {
            "link": link,
            "user_prompt": user_prompt,
            "llm": llm or self._search_defaults["parse_model"] or "gpt-4o",
        })

        if isinstance(result, dict):
            result.setdefault("content", "")
            result.setdefault("urls", [])
            result.setdefault("score", 0.0)
            return result

        return {"content": str(result).strip(), "urls": [], "score": 0.5}

    # ── Formatting helpers (unchanged interface) ───────────────────────
    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            link = r.get("link", "")
            text = r.get("text", "") or r.get("snippet", "")
            lines.append(f"{i}. {title}\nURL: {link}\n{text}")
        return "\n".join(lines)

    def format_parsed_for_llm(self, parsed: Dict[str, Any]) -> str:
        if not parsed:
            return ""
        content = str(parsed.get("content", "")).strip()
        score = parsed.get("score", "")
        urls = parsed.get("urls", []) or []
        lines = [f"score: {score}", content]
        if urls:
            lines.append("related_urls:")
            for item in urls:
                if not isinstance(item, dict):
                    continue
                lines.append(f"- {item.get('url', '')}: {item.get('description', '')}")
        return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    lib = LibraryRetriever()
    print(f"MCP URL: {lib._mcp_url}")

    print("\n" + "=" * 50)
    print("Test 1: web_search")
    print("=" * 50)
    try:
        results = lib.search(query="Python async programming", top_k=3)
        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.get('title', '')}")
            print(f"     {r.get('link', '')}")
            print(f"     {r.get('snippet', '')[:100]}...")
        print("\nFormatted:")
        print(lib.format_for_llm(results)[:500])
    except Exception as e:
        print(f"Search failed: {e}")

    print("\n" + "=" * 50)
    print("Test 2: web_parse")
    print("=" * 50)
    try:
        parsed = lib.parse(
            link="https://docs.python.org/3/library/asyncio.html",
            user_prompt="What is asyncio and what are its main features?",
        )
        print(f"Score: {parsed.get('score')}")
        print(f"Content preview: {str(parsed.get('content', ''))[:300]}...")
    except Exception as e:
        print(f"Parse failed: {e}")

    print("\n" + "=" * 50)
    print("Done")
    print("=" * 50)