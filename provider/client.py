"""Direct Kiro native transport, shaped like OpenAI's chat client."""
from __future__ import annotations

import asyncio
import collections.abc
import json
import logging
import time
from datetime import datetime, timezone
import urllib.parse
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

from capabilities import (
    additional_model_request_fields_supported,
    mark_additional_model_request_fields_unsupported,
)
from credentials import KiroAuthError, get_credentials, profile_regions, remember_profile_arn, runtime_region
from transport import KiroHTTPError, request, request_json
from translate import build_request

_FALLBACK = ("claude-sonnet-4.5", "claude-haiku-4.5", "gpt-5.6-terra")
_LOG = logging.getLogger(__name__)
_END = object()
# Kiro reports a model that cannot take additionalModelRequestFields in two shapes:
#   "...additionalModelRequestFields is not supported for this model"  (older models)
#   "Invalid additionalModelRequestFields: property 'enabled' is not defined in the schema..."  (4.6+)
# Both arrive with reason REQUEST_BODY_INVALID, so match on the field name rather than one sentence.
_ADDITIONAL_FIELDS_KEY = "additionalmodelrequestfields"
# The branded runtime gateway rejects a profileArn-less Builder ID request with
# 400 "profileArn is required for this request". Builder ID has no discoverable
# profile, so official clients send this fixed placeholder on streaming calls
# (embedded in the official kiro-cli 2.8.0 binary, fig_api_client::profile_resolver).
# Do not attach it to REST calls: ListAvailableModels/getUsageLimits reject it.
BUILDER_ID_PROFILE_ARN = "arn:aws:codewhisperer:us-east-1:638616132270:profile/AAAACCCCXXXX"


def _rejects_disabled_thinking(exc: KiroHTTPError) -> bool:
    if exc.status != 400:
        return False
    text = exc.body.decode("utf-8", "replace") if isinstance(exc.body, (bytes, bytearray)) else str(exc.body)
    return "additionalModelRequestFields" in text and "enumeration" in text and '"adaptive"' in text.replace('\\"', '"')


_NATIVE_REASONING_KEYS = {"thinking", "output_config", "reasoning"}
_CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_GPT_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"minimal": "low", "ultra": "max"}


def _kiro_reasoning_fields(model: str, config: Any) -> dict | None:
    """Map Hermes' generic ``{"enabled", "effort"}`` reasoning config onto Kiro's per-family schema.

    Kiro validates ``additionalModelRequestFields`` strictly: Claude accepts
    ``thinking.type`` (adaptive|disabled) and ``output_config.effort``; GPT accepts
    ``reasoning.effort``. Anything else is a 400, which broke title generation
    (Hermes sends ``{"enabled": False}``). Native-shaped dicts pass through.
    """
    if not isinstance(config, dict) or not config:
        return None
    if set(config) <= _NATIVE_REASONING_KEYS:
        return dict(config)
    if not ({"enabled", "effort"} & set(config)):
        # Unknown shape: forward untouched so provider-specific configs keep working.
        return dict(config)
    family = (model or "").lower()
    effort = str(config.get("effort") or "").strip().lower()
    effort = _EFFORT_ALIASES.get(effort, effort)
    disabled = config.get("enabled") is False or effort == "none"
    if family.startswith("claude"):
        if disabled:
            return {"thinking": {"type": "disabled"}}
        fields: dict = {"thinking": {"type": "adaptive"}}
        if effort in _CLAUDE_EFFORTS:
            fields["output_config"] = {"effort": effort}
        return fields
    if family.startswith("gpt"):
        if disabled:
            return {"reasoning": {"effort": "none"}}
        return {"reasoning": {"effort": effort}} if effort in _GPT_EFFORTS else None
    return None


def _model_id(body: dict) -> str:
    try:
        model = body["conversationState"]["currentMessage"]["userInputMessage"]["modelId"]
    except (KeyError, TypeError):
        return ""
    return model if isinstance(model, str) else ""


def _rejects_additional_model_request_fields(exc: KiroHTTPError) -> bool:
    if exc.status != 400:
        return False
    try:
        payload = json.loads(exc.body)
    except (TypeError, ValueError):
        return False

    def strings(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    values = list(strings(payload))
    return "REQUEST_BODY_INVALID" in values and any(
        _ADDITIONAL_FIELDS_KEY in value.casefold() for value in values
    )


def _headers(creds, *, accept: str = "application/json", target: str = "") -> dict[str, str]:
    ua = "aws-sdk-python/1.0 KiroIDE-hermes"
    headers = {
        "Accept": accept,
        "Authorization": f"Bearer {creds.access_token}",
        "TokenType": "SSO_OIDC",
        "User-Agent": ua,
        "x-amz-user-agent": ua,
        "x-amzn-codewhisperer-optout": "true",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }
    if target:
        headers["Content-Type"] = "application/x-amz-json-1.0"
        headers["x-amz-target"] = target
    return headers


def _management(creds, target: str, payload: dict, *, region: str = "") -> dict:
    """Kiro's target-dispatched control plane."""
    region = runtime_region(region or creds.api_region)
    return request_json(
        "POST",
        f"https://management.{region}.kiro.dev/",
        body=json.dumps(payload).encode(),
        headers=_headers(creds, target=target),
    )


def _rest_host(region: str) -> str:
    """us-east-1 keeps its historical host; other regions are served by q.<region>."""
    return "codewhisperer.us-east-1.amazonaws.com" if region == "us-east-1" else f"q.{region}.amazonaws.com"


def _rest(creds, path: str, query: dict[str, str]) -> dict:
    return request_json(
        "GET",
        f"https://{_rest_host(creds.api_region)}/{path}?{urllib.parse.urlencode(query)}",
        headers=_headers(creds),
    )


def list_model_ids() -> list[str]:
    try:
        creds = get_credentials()
        profile_arn = _ensure_profile_arn(creds)
        query = {"origin": "AI_EDITOR", "maxResults": "50"}
        if profile_arn:
            query["profileArn"] = profile_arn
        data = _rest(creds, "ListAvailableModels", query)
        models = [str(m["modelId"]) for m in data.get("models") or [] if isinstance(m, dict) and isinstance(m.get("modelId"), str)]
        if not models:
            _LOG.warning("Kiro returned no models; using the built-in fallback list.")
        return models or list(_FALLBACK)
    except Exception as exc:
        _LOG.warning("Kiro model discovery failed (%s); using the built-in fallback list.", exc)
        return list(_FALLBACK)


def get_usage_limits() -> dict:
    creds = get_credentials()
    profile_arn = _ensure_profile_arn(creds)
    query = {"isEmailRequired": "true", "origin": "AI_EDITOR", "resourceType": "AGENTIC_REQUEST"}
    if profile_arn:
        query["profileArn"] = profile_arn
    return _rest(creds, "getUsageLimits", query)


def _bonuses(bucket: dict) -> list:
    """Active gifted/promo credit grants of one bucket as (name, used, limit, expires_at)."""
    out = []
    for bonus in bucket.get("bonuses") or []:
        if not isinstance(bonus, dict) or str(bonus.get("status") or "ACTIVE").upper() != "ACTIVE":
            continue
        try:
            used, limit = float(str(bonus.get("currentUsage"))), float(str(bonus.get("usageLimit")))
        except (TypeError, ValueError):
            continue
        expires = None
        try:
            stamp = float(str(bonus.get("expiresAt")))
            expires = datetime.fromtimestamp(stamp / 1000 if stamp > 10_000_000_000 else stamp, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pass
        out.append((str(bonus.get("displayName") or bonus.get("bonusCode") or "Bonus"), used, limit, expires))
    return out


def format_usage(data: dict) -> str:
    buckets = data.get("usageBreakdownList") or []
    if not isinstance(buckets, list) or not buckets:
        return "Kiro returned no usage buckets."

    def display(value: object) -> str:
        try:
            return f"{float(str(value)):,.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(value)

    lines = ["Kiro usage:"]
    for index, bucket in enumerate(buckets, 1):
        if not isinstance(bucket, dict):
            continue
        current, limit = bucket.get("currentUsage"), bucket.get("usageLimit")
        name = bucket.get("displayName") or bucket.get("usageType") or bucket.get("resourceType") or f"Allowance {index}"
        try:
            current_number, limit_number = float(str(current)), float(str(limit))
            included = min(current_number, limit_number)
            percent = f" ({included / limit_number * 100:.0f}%)" if limit_number > 0 else ""
        except (TypeError, ValueError):
            included, percent = current, ""
        reset = bucket.get("nextDateReset")
        if reset:
            try:
                reset = datetime.fromtimestamp(float(str(reset)), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except (TypeError, ValueError, OSError):
                pass
        lines.append(f"- {name}: {display(included)}/{display(limit)}{percent}" + (f"; resets {reset}" if reset else ""))
        for b_name, b_used, b_limit, b_expires in _bonuses(bucket):
            b_percent = f" ({b_used / b_limit * 100:.0f}%)" if b_limit > 0 else ""
            expiry = f"; expires {b_expires:%Y-%m-%d %H:%M UTC}" if b_expires else ""
            lines.append(f"  Bonus {b_name}: {display(b_used)}/{display(b_limit)}{b_percent}, {display(max(b_limit - b_used, 0))} left{expiry}")

        if any(key in bucket for key in ("currentOverages", "currentOveragesWithPrecision", "overageCap", "overageCharges")):
            overage = bucket.get("currentOveragesWithPrecision", bucket.get("currentOverages", 0))
            cap = bucket.get("overageCapWithPrecision", bucket.get("overageCap", 0))
            try:
                overage_number, cap_number = float(str(overage)), float(str(cap))
                cap_percent = f" ({overage_number / cap_number * 100:.0f}% of cap)" if cap_number > 0 else ""
            except (TypeError, ValueError):
                cap_percent = ""
            plural = str(bucket.get("displayNamePlural") or f"{name}s").lower()
            extra = f"  Extra usage: {display(overage)}/{display(cap)} {plural}{cap_percent}"
            rate, charges, currency = bucket.get("overageRate"), bucket.get("overageCharges"), bucket.get("currency")
            try:
                extra += f"; ${float(str(charges)):.2f} {currency or 'USD'} at ${display(rate)}/{str(name).lower()}"
            except (TypeError, ValueError):
                pass
            lines.append(extra)
    return "\n".join(lines)


def build_usage_snapshot(data: dict):
    """Normalize Kiro allowances for Hermes' native ``/usage`` renderer."""
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

    buckets = data.get("usageBreakdownList") or []
    if not isinstance(buckets, list):
        return None
    windows = []
    details = []
    for index, bucket in enumerate(buckets, 1):
        if not isinstance(bucket, dict):
            continue
        try:
            current = float(str(bucket.get("currentUsage")))
            limit = float(str(bucket.get("usageLimit")))
        except (TypeError, ValueError):
            continue
        included = min(max(current, 0.0), max(limit, 0.0))
        used_percent = included / limit * 100 if limit > 0 else None
        reset_at = None
        try:
            reset = float(str(bucket.get("nextDateReset")))
            reset_at = datetime.fromtimestamp(reset / 1000 if reset > 10_000_000_000 else reset, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pass
        name = str(
            bucket.get("displayName")
            or bucket.get("usageType")
            or bucket.get("resourceType")
            or f"Allowance {index}"
        )
        windows.append(AccountUsageWindow(
            label=name,
            used_percent=used_percent,
            reset_at=reset_at,
            detail=f"{included:g}/{limit:g}",
        ))
        for b_name, b_used, b_limit, b_expires in _bonuses(bucket):
            windows.append(AccountUsageWindow(
                label=f"{name} bonus: {b_name}",
                used_percent=b_used / b_limit * 100 if b_limit > 0 else None,
                reset_at=b_expires,
                detail=f"{b_used:g}/{b_limit:g}",
            ))
        overage = bucket.get("currentOveragesWithPrecision", bucket.get("currentOverages"))
        if overage is not None:
            cap = bucket.get("overageCapWithPrecision", bucket.get("overageCap"))
            charge = bucket.get("overageCharges")
            currency = str(bucket.get("currency") or "USD")
            text = f"{name} extra usage: {overage}"
            if cap is not None:
                text += f"/{cap}"
            if charge is not None:
                text += f"; {charge} {currency}"
            details.append(text)
    if not windows:
        return None
    return AccountUsageSnapshot(
        provider="kiro",
        source="kiro_usage_api",
        fetched_at=datetime.now(timezone.utc),
        title="Kiro usage",
        windows=tuple(windows),
        details=tuple(details),
        raw=data,
    )


def _ensure_profile_arn(creds) -> str:
    """Discover the IdC profile once; Builder ID must not make this denied call."""
    if creds.profile_arn:
        return creds.profile_arn
    if getattr(creds, "is_builder_id", False):
        return ""
    regions = profile_regions(creds.region)
    for region in regions:
        # ponytail: fall through on empty results only; HTTP/auth errors must propagate.
        data = _management(creds, "KiroControlPlaneBearerService.ListAvailableProfiles", {}, region=region)
        for profile in data.get("profiles") or []:
            if isinstance(profile, dict) and (arn := str(profile.get("profileArn") or profile.get("arn") or "").strip()):
                creds.profile_arn = arn
                remember_profile_arn(creds)
                return arn
    raise KiroAuthError(f"Kiro returned no profile ARN for this IAM Identity Center login in {' or '.join(regions)}")


def _chunk(model: str, delta: dict, finish: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=f"chatcmpl-{uuid.uuid4().hex}", object="chat.completion.chunk", created=int(time.time()), model=model, choices=[SimpleNamespace(index=0, delta=SimpleNamespace(**delta), finish_reason=finish)])


def _next_or_end(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _END


class _AwaitableResponse(SimpleNamespace):
    def __await__(self):
        async def done():
            return self
        return done().__await__()


class _HybridStream:
    """One native stream for Hermes' synchronous and auxiliary async callers."""
    def __init__(self, iterator: Iterator[SimpleNamespace]) -> None:
        self._iterator = iterator

    def __iter__(self):
        return self._iterator

    def __await__(self):
        async def done():
            return self
        return done().__await__()

    def __aiter__(self):
        async def iterate():
            while (item := await asyncio.to_thread(_next_or_end, self._iterator)) is not _END:
                yield item
        return iterate()


class _LazyResponse(collections.abc.Coroutine):
    """Result of a non-stream call made while an event loop is running on this thread.

    Awaiting it (or wrapping it in a task) runs the request in a worker thread, as before.
    Reading an attribute instead runs the request inline, for synchronous callers that happen
    to sit on a thread with a running loop (e.g. Hermes auxiliary title generation).
    """
    def __init__(self, fn, *args) -> None:
        self._fn, self._args = fn, args
        self._result = None
        self._coro = None

    def _resolve(self):
        if self._result is None:
            self._result = self._fn(*self._args)
        return self._result

    def _driver(self):
        if self._coro is None:
            self._coro = self._await_in_thread()
        return self._coro

    async def _await_in_thread(self):
        if self._result is None:
            self._result = await asyncio.to_thread(self._fn, *self._args)
        return self._result

    def __await__(self):
        return self._driver().__await__()

    def send(self, value):
        return self._driver().send(value)

    def throw(self, *args):
        return self._driver().throw(*args)

    def close(self):
        if self._coro is not None:
            self._coro.close()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


class KiroClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **_: Any) -> None:
        self.api_key = "kiro-oauth-local"
        self.base_url = "https://runtime.us-east-1.kiro.dev"
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def close(self) -> None:
        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _open(self, body: dict, force_refresh: bool = False, stale_access_token: str = ""):
        creds = get_credentials(force_refresh=force_refresh, stale_access_token=stale_access_token)
        # ponytail: Builder ID rejects profile discovery; corporate IdC requires the discovered ARN.
        if profile_arn := _ensure_profile_arn(creds):
            body["profileArn"] = profile_arn
        elif getattr(creds, "is_builder_id", False):
            body["profileArn"] = BUILDER_ID_PROFILE_ARN
        headers = _headers(
            creds,
            accept="application/vnd.amazon.eventstream",
            target="AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        )
        headers["x-amzn-kiro-agent-mode"] = "vibe"
        used_access_token = creds.access_token
        try:
            response = request(
                "POST",
                f"https://runtime.{creds.api_region}.kiro.dev/generateAssistantResponse",
                body=json.dumps(body).encode(),
                headers=headers,
                timeout=600,
                stream=True,
            )
        except KiroHTTPError as exc:
            exc.access_token = used_access_token
            raise
        return response, creds.access_token

    def _events(self, body: dict) -> Iterator[tuple[str, dict]]:
        response = None
        stale_access_token = ""
        retried_auth = False
        retried_without_additional_fields = False
        retried_adaptive_thinking = False
        while True:
            try:
                opened = self._open(body, force_refresh=retried_auth, stale_access_token=stale_access_token)
                response, stale_access_token = opened if isinstance(opened, tuple) else (opened, "")
                break
            except KiroHTTPError as exc:
                fields = body.get("additionalModelRequestFields")
                if (
                    not retried_adaptive_thinking
                    and isinstance(fields, dict)
                    and (fields.get("thinking") or {}).get("type") == "disabled"
                    and _rejects_disabled_thinking(exc)
                ):
                    # Some Claude models (e.g. opus-5.5) only allow adaptive thinking; keep it cheap instead.
                    body = dict(body)
                    body["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
                    retried_adaptive_thinking = True
                    continue
                if (
                    not retried_without_additional_fields
                    and "additionalModelRequestFields" in body
                    and _rejects_additional_model_request_fields(exc)
                ):
                    model = _model_id(body)
                    if model:
                        mark_additional_model_request_fields_unsupported(model)
                    body = dict(body)
                    body.pop("additionalModelRequestFields", None)
                    retried_without_additional_fields = True
                    continue
                if exc.status not in (401, 403) or retried_auth:
                    raise KiroAuthError(f"Kiro runtime failed ({exc.status}): {exc.body.decode('utf-8', 'replace')[:500]}", status=exc.status) from exc
                stale_access_token = exc.access_token
                retried_auth = True
        if response is None:
            raise KiroAuthError("Kiro runtime could not be reached")
        from botocore.eventstream import EventStreamBuffer

        buffer = EventStreamBuffer()
        try:
            while raw := response.read(8192):
                buffer.add_data(raw)
                while True:
                    try:
                        event = buffer.next()
                    except StopIteration:
                        break
                    if event is None:
                        break
                    try:
                        payload = json.loads(event.payload)
                    except (TypeError, ValueError):
                        payload = {}
                    if isinstance(payload, dict):
                        yield str(event.headers.get(":event-type") or ""), payload
        finally:
            release = getattr(response, "release_conn", None)
            if release:
                release()

    def _create(self, *, model: str, messages: list[dict], stream: bool = False, tools: Any = None, extra_body: dict | None = None, **_: Any):
        effort = _kiro_reasoning_fields(model, (extra_body or {}).get("reasoning"))
        if effort and not additional_model_request_fields_supported(model):
            effort = None
        body = build_request(messages, tools, model, effort)
        if stream:
            return self._stream(model, body)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._complete(model, body)
        return _LazyResponse(self._complete, model, body)

    @staticmethod
    def _add_tool_event(calls: dict[str, dict[str, str]], data: dict) -> None:
        call_id = str(data.get("toolUseId") or uuid.uuid4().hex)
        call = calls.setdefault(call_id, {"name": "", "arguments": ""})
        if data.get("name"):
            call["name"] = str(data["name"])
        if "input" not in data:
            return
        value = data["input"]
        fragment = json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value or "")
        call["arguments"] += fragment

    @staticmethod
    def _tool_calls(calls: dict[str, dict[str, str]]) -> list[SimpleNamespace]:
        return [SimpleNamespace(index=index, id=call_id, type="function", function=SimpleNamespace(name=value["name"], arguments=value["arguments"] or "{}")) for index, (call_id, value) in enumerate(calls.items())]

    def _stream(self, model: str, body: dict) -> _HybridStream:
        return _HybridStream(self._stream_impl(model, body))

    def _stream_impl(self, model: str, body: dict) -> Iterator[SimpleNamespace]:
        sent = False
        calls: dict[str, dict[str, str]] = {}
        tools_sent = False
        for event, data in self._events(body):
            if event == "assistantResponseEvent" and isinstance(data.get("content"), str) and data["content"]:
                yield _chunk(model, {"role": "assistant" if not sent else None, "content": data["content"]})
                sent = True
            elif event == "toolUseEvent":
                self._add_tool_event(calls, data)
            elif event == "metadataEvent" and data.get("stopReason") == "TOOL_USE":
                if calls:
                    yield _chunk(model, {"role": "assistant" if not sent else None, "tool_calls": self._tool_calls(calls)})
                    tools_sent = sent = True
                yield _chunk(model, {}, "tool_calls")
                return
            elif event == "metadataEvent" and str(data.get("stopReason") or "").upper() == "CONTENT_FILTERED":
                raise KiroAuthError("Kiro declined the request (CONTENT_FILTERED)")
            elif event in {"error", "exception"} or data.get("error"):
                raise KiroAuthError(f"Kiro stream error: {data.get('message') or data.get('error')}")
        if calls:
            yield _chunk(model, {"role": "assistant" if not sent else None, "tool_calls": self._tool_calls(calls)})
            tools_sent = sent = True
            yield _chunk(model, {}, "tool_calls")
            return
        if not sent and not tools_sent:
            raise KiroAuthError("Kiro closed a successful stream without an event")
        yield _chunk(model, {}, "stop")

    def _complete(self, model: str, body: dict) -> _AwaitableResponse:
        text: list[str] = []
        tool_calls: list[SimpleNamespace] = []
        finish = "stop"
        for chunk in self._stream(model, body):
            delta = chunk.choices[0].delta
            if content := getattr(delta, "content", None):
                text.append(content)
            tool_calls.extend(getattr(delta, "tool_calls", None) or [])
            finish = chunk.choices[0].finish_reason or finish
        message = SimpleNamespace(role="assistant", content="".join(text) or None, tool_calls=tool_calls or None, reasoning=None, reasoning_content=None, reasoning_details=None)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        return _AwaitableResponse(model=model, choices=[SimpleNamespace(index=0, message=message, finish_reason=finish)], usage=usage)
