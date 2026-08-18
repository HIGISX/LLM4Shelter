"""Parse JSON from an LLM response without silently repairing it."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_response(raw_response: str) -> tuple[Any | None, str | None]:
    text = (raw_response or "").strip()
    if not text:
        return None, "empty_response"
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass
    return None, "invalid_json"
