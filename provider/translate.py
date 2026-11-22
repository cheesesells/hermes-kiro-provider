"""Small OpenAI-chat to Kiro conversationState translation layer."""
from __future__ import annotations

import base64
import binascii
import json
import uuid
from typing import Any

_EMPTY = "Please proceed with the task."
_TOOL = "Tool results provided."
_EMPTY_SCHEMA = {"type": "object", "properties": {}}
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_LISTS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_CHILDREN = frozenset({
    "items", "additionalItems", "contains", "propertyNames", "not",
    "if", "then", "else", "unevaluatedItems", "unevaluatedProperties",
})


def _sanitize_schema(schema: Any) -> Any:
    """Drop JSON-Schema keys Kiro rejects with 400 REQUEST_BODY_INVALID.

    Kiro refuses any tool schema carrying ``additionalProperties`` (either
    value) and an empty ``required`` array; Hermes schemas include both.
    Neither key is needed for tool dispatch, so strip them recursively.
    Only schema-bearing keywords are traversed, so a property literally named
    ``additionalProperties``/``required`` and ``const``/``enum`` payloads
    survive untouched.
    """
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "required" and isinstance(value, (list, tuple)) and not value:
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            cleaned[key] = {name: _sanitize_schema(child) for name, child in value.items()}
        elif (key in _SCHEMA_LISTS or key == "items") and isinstance(value, list):
            cleaned[key] = [_sanitize_schema(child) for child in value]
        elif key in _SCHEMA_CHILDREN:
            cleaned[key] = _sanitize_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


# Kiro's ImageBlock formats (Bedrock Converse names); OpenAI parts carry a MIME type.
_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg", "image/gif": "gif", "image/webp": "webp"}


def _image_block(part: Any) -> dict | None:
    """OpenAI ``image_url`` part with a base64 data URL -> Kiro image block; anything else -> None.

    Remote http(s) URLs are skipped: Kiro only takes inline bytes, and Hermes' vision tools
    already inline local and fetched images as data URLs before they reach the provider.
    """
    if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image"}:
        return None
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        return None
    header, data = url[5:].split(",", 1)
    mime, _, encoding = header.partition(";")
    fmt = _IMAGE_FORMATS.get(mime.strip().lower())
    if fmt is None or encoding.strip().lower() != "base64":
        return None
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    return {"format": fmt, "source": {"bytes": data}}


def _images(*messages: dict) -> list[dict]:
    """Image blocks from the content lists of ``messages``, in order."""
    blocks: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks.extend(block for part in content if (block := _image_block(part)) is not None)
    return blocks


def _tool_specs(tools: Any, messages: list[dict]) -> list[dict]:
    """Keep definitions for replayed tool calls; Kiro rejects orphaned results."""
    specs: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if name:
            specs[name] = {"toolSpecification": {"name": name, "description": fn.get("description") or name, "inputSchema": {"json": _sanitize_schema(fn.get("parameters") or _EMPTY_SCHEMA)}}}
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
    item: dict[str, Any] = {"content": _TOOL, "modelId": model, "origin": "KIRO_CLI", "userInputMessageContext": {"toolResults": [_tool_result(message) for message in messages]}}
    if images := _images(*messages):
        item["images"] = images
    return {"userInputMessage": item}


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
            item = {"content": content, "modelId": model, "origin": "KIRO_CLI"}
            if images := _images(message):
                item["images"] = images
            history.append({"userInputMessage": item})
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
    if images := _images(*(current_tools or [current])):
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
