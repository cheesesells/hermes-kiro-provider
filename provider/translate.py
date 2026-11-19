"""Small OpenAI-chat to Kiro conversationState translation layer."""
from __future__ import annotations

import json
import uuid
from typing import Any

_EMPTY = "Please proceed with the task."
_TOOL = "Tool results provided."
_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


_IMAGE_FORMATS = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg", "gif": "gif", "webp": "webp"}


def _images(content: Any) -> list[dict]:
    """Collect OpenAI-style data-URL image parts as Kiro `images` entries.

    Remote http(s) URLs are skipped: Kiro takes raw base64 bytes only.
    """
    found: list[dict] = []
    if not isinstance(content, list):
        return found
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image"}:
            continue
        ref = part.get("image_url")
        url = ref.get("url") if isinstance(ref, dict) else ref
        if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
            continue
        header, data = url.split(";base64,", 1)
        fmt = _IMAGE_FORMATS.get(header[len("data:image/"):].lower())
        if fmt and data:
            found.append({"format": fmt, "source": {"bytes": data}})
    return found


def _tool_specs(tools: Any, messages: list[dict]) -> list[dict]:
    """Keep definitions for replayed tool calls; Kiro rejects orphaned results."""
    specs: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if name:
            specs[name] = {"toolSpecification": {"name": name, "description": fn.get("description") or name, "inputSchema": {"json": fn.get("parameters") or _EMPTY_SCHEMA}}}
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name and name not in specs:
                specs[name] = {"toolSpecification": {"name": name, "description": name, "inputSchema": {"json": _EMPTY_SCHEMA}}}
    return list(specs.values())


def _assistant(message: dict) -> dict:
    uses = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        uses.append({"name": fn.get("name") or "", "toolUseId": call.get("id") or uuid.uuid4().hex, "input": arguments})
    item: dict[str, Any] = {"content": _text(message.get("content"))}
    if uses:
        item["toolUses"] = uses
    return {"assistantResponseMessage": item}


def _tool_result(message: dict) -> dict:
    return {"toolUseId": message.get("tool_call_id") or "", "status": "error" if message.get("is_error") else "success", "content": [{"text": _text(message.get("content")) or "(no output)"}]}


def _tool_turn(messages: list[dict], model: str) -> dict:
    return {"userInputMessage": {"content": _TOOL, "modelId": model, "origin": "KIRO_CLI", "userInputMessageContext": {"toolResults": [_tool_result(message) for message in messages]}}}


def build_request(messages: list[dict], tools: Any, model: str, effort: dict | None, conversation_id: str | None = None) -> dict:
    system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system").strip()
    conversation = [m for m in messages if m.get("role") != "system"]
    current = conversation.pop() if conversation and conversation[-1].get("role") in {"user", "tool"} else {"role": "user", "content": ""}
    current_tools = [current] if current.get("role") == "tool" else []
    while current_tools and conversation and conversation[-1].get("role") == "tool":
        current_tools.insert(0, conversation.pop())

    system_pending = system
    history: list[dict] = []
    index = 0
    while index < len(conversation):
        message = conversation[index]
        role = message.get("role")
        if role == "assistant":
            history.append(_assistant(message))
        elif role == "user":
            content = _text(message.get("content"))
            if system_pending:
                content = f"{system_pending}\n\n{content}".strip()
                system_pending = ""
            content = content or _EMPTY
            entry: dict[str, Any] = {"content": content, "modelId": model, "origin": "KIRO_CLI"}
            if images := _images(message.get("content")):
                entry["images"] = images
            history.append({"userInputMessage": entry})
        elif role == "tool":
            grouped = [message]
            while index + 1 < len(conversation) and conversation[index + 1].get("role") == "tool":
                index += 1
                grouped.append(conversation[index])
            history.append(_tool_turn(grouped, model))
        index += 1

    context: dict[str, Any] = {}
    if current_tools:
        user: dict[str, Any] = {"content": _TOOL, "modelId": model, "origin": "KIRO_CLI"}
        context["toolResults"] = [_tool_result(message) for message in current_tools]
    else:
        content = _text(current.get("content"))
        if system_pending:
            content = f"{system_pending}\n\n{content}".strip()
        content = content or _EMPTY
        user = {"content": content, "modelId": model, "origin": "KIRO_CLI"}
        if images := _images(current.get("content")):
            user["images"] = images
    if specs := _tool_specs(tools, conversation + current_tools + ([] if current_tools else [current])):
        context["tools"] = specs
    if context:
        user["userInputMessageContext"] = context
    state: dict[str, Any] = {"chatTriggerType": "MANUAL", "agentTaskType": "vibe", "conversationId": conversation_id or uuid.uuid4().hex, "currentMessage": {"userInputMessage": user}}
    if history:
        state["history"] = history
    body: dict[str, Any] = {"conversationState": state, "agentMode": "vibe"}
    if effort:
        body["additionalModelRequestFields"] = effort
    return body
