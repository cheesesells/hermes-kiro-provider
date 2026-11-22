from __future__ import annotations

import io
import json
import sys
import asyncio
import struct
import subprocess
import threading
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "provider"))

import client
import credentials
from credentials import BUILDER_ID_START_URL, KiroAuthError, _next_login_choice, prompt_login_inputs, runtime_region, validate_start_url
from hermes_cli.auth_constants import AuthError
from translate import build_request


def _register_local_profile():
    """Finish core discovery, then exercise this checkout instead of an installed copy."""
    import provider
    import providers

    providers.get_provider_profile("kiro")
    providers.register_provider(provider.profile)
    return provider


def test_start_url_is_strict_and_region_maps():
    assert validate_start_url(BUILDER_ID_START_URL) == BUILDER_ID_START_URL
    assert validate_start_url("https://d-abc.awsapps.com/start/") == "https://d-abc.awsapps.com/start"
    assert runtime_region("eu-west-1") == "eu-central-1"
    with pytest.raises(KiroAuthError):
        validate_start_url("http://evil.example/start")
    with pytest.raises(KiroAuthError):
        validate_start_url("https://notawsapps.com/start")
    with pytest.raises(KiroAuthError):
        runtime_region("attacker.example")


def test_interactive_login_defaults_to_builder_id_or_accepts_custom_idc():
    defaults = iter(["", ""])
    custom = iter(["2", "https://d-abc.awsapps.com/start", "eu-west-1"])
    assert prompt_login_inputs(None, None, lambda _: next(defaults)) == (BUILDER_ID_START_URL, "us-east-1")
    assert prompt_login_inputs(None, None, lambda _: next(custom)) == ("https://d-abc.awsapps.com/start", "eu-west-1")


def test_login_selector_moves_with_arrow_keys_and_fallback_is_spaced():
    assert _next_login_choice(0, "down") == 1
    assert _next_login_choice(1, "up") == 0
    assert (_next_login_choice(3, "down"), _next_login_choice(4, "down"), _next_login_choice(0, "up")) == (4, 4, 0)
    prompts = []
    values = iter(["", ""])
    prompt_login_inputs(None, None, lambda text: prompts.append(text) or next(values))
    assert "\n  1. AWS Builder ID\n  2. IAM Identity Center\n  3. Google\n  4. GitHub\n  5. Google / GitHub (import the kiro-cli login)\n\n" in prompts[0]


def test_request_hoists_system_and_never_sends_empty_user_content():
    request = build_request([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": ""},
    ], [], "claude-sonnet-4.5", None, "c")
    message = request["conversationState"]["currentMessage"]["userInputMessage"]
    assert message["content"] == "Be concise."
    assert "system" not in json.dumps(request)


def test_nonstream_facade_returns_openai_shape(monkeypatch):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter([
        ("assistantResponseEvent", {"content": "hello"}),
        ("assistantResponseEvent", {"content": " world"}),
    ]))
    response = instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "hello world"
    assert response.usage.total_tokens == 0


def test_empty_success_stream_is_an_error(monkeypatch):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter(()))
    with pytest.raises(KiroAuthError, match="without an event"):
        list(instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}], stream=True))


def test_content_filtered_stream_is_an_error(monkeypatch):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter([
        ("metadataEvent", {"stopReason": "CONTENT_FILTERED"}),
    ]))
    with pytest.raises(KiroAuthError, match="CONTENT_FILTERED"):
        list(instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}], stream=True))


def test_http_status_is_visible_to_host_matchers(monkeypatch):
    error = KiroAuthError("rate limited", status=429)
    assert (error.status, error.status_code) == (429, 429)
    instance = client.KiroClient()
    def rejected(*args, **kwargs):
        raise client.KiroHTTPError(429, b"rate limited")
    monkeypatch.setattr(instance, "_open", rejected)
    with pytest.raises(KiroAuthError) as caught:
        list(instance._events({}))
    assert caught.value.status_code == 429


def test_provider_client_still_constructs_after_companion_cleanup():
    """companion_error plumbing is gone; the client must construct and raise through normal paths."""
    instance = client.KiroClient()
    assert instance.api_key and instance.base_url


def test_provider_registration_does_not_import_transport_dependencies(monkeypatch, tmp_path):
    import hermes_constants

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    repo_root = Path(__file__).parents[1]
    core_root = Path(hermes_constants.__file__).parent
    script = f"""
import builtins
import sys
sys.path[:0] = [{str(core_root)!r}, {str(repo_root)!r}]
import hermes_bootstrap
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in ("botocore", "urllib3") or name.startswith(("botocore.", "urllib3.")):
        raise ModuleNotFoundError(f"No module named '{{name}}'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import provider
assert provider.profile.name == "kiro"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_new_model_catalog_fetches_and_falls_back_when_empty(monkeypatch, tmp_path):
    """The provider must serve models even when its command registration could not happen."""
    import provider

    assert provider.profile.fallback_models  # non-empty fallback tuple exists
    monkeypatch.setattr(client, "list_model_ids", lambda: [])
    assert provider.profile.fetch_models() == list(provider.profile.fallback_models)


def test_commands_run_standalone_without_hermes_cli(monkeypatch, tmp_path):
    """commands.py stays plugin-API-free: it is the fallback when command registration breaks."""
    source = Path(__file__).parents[1] / "provider" / "commands.py"
    assert "hermes_cli" not in source.read_text()

    import runpy
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["commands.py", "--help"]  # exercises argparse wiring; --help exits cleanly
        with pytest.raises(SystemExit) as exit_info:  # argparse exits 0 on --help
            runpy.run_path(str(source), run_name="__main__")
        assert exit_info.value.code == 0
    finally:
        sys.argv = saved_argv


def test_native_profile_declares_auth_catalog_and_capabilities(monkeypatch):
    import provider
    from hermes_cli.models import provider_model_ids

    _register_local_profile()
    assert provider.profile.auth_type == "oauth_device_code"
    assert provider.profile.env_vars == ()
    assert provider.profile.auth_handler is credentials.auth_handler
    assert provider.profile.refresh_credential is credentials.refresh_credential
    assert provider.profile.model_capabilities["claude-haiku-4.5"] == {
        "supports_vision": True,
        "supports_tools": True,
    }
    assert provider.profile.model_capabilities.get("claude-opus-5.5")["supports_vision"] is True
    assert provider.profile.model_capabilities.get("deepseek-v3.2")["supports_vision"] is False
    assert provider.profile.model_capabilities.get("unlisted-family-1")["supports_vision"] is False
    assert provider.profile.model_capabilities.get("auto")["supports_vision"] is True
    monkeypatch.setattr(provider, "list_model_ids", lambda: ["live-model"])
    monkeypatch.setattr(provider, "_CATALOG_AT", 0.0)
    monkeypatch.setattr(provider, "_CATALOG_MODELS", provider._FALLBACK_MODELS)
    assert provider.profile.fetch_models() == ["live-model"]
    assert provider_model_ids("kiro") == ["live-model"]


def test_event_decoder_handles_a_frame_split_mid_prelude(monkeypatch):
    class Response:
        def __init__(self):
            payload = json.dumps({"content": "OK"}).encode()
            event_type = b"assistantResponseEvent"
            headers = b"\x0b:event-type\x07" + struct.pack(">H", len(event_type)) + event_type
            prelude = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
            prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
            frame = prelude + headers + payload
            frame += struct.pack(">I", zlib.crc32(frame) & 0xFFFFFFFF)
            self.reads = iter([frame[:5], frame[5:], b""])

        def read(self, _):
            return next(self.reads)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_open", lambda *_, **__: Response())
    assert list(instance._events({})) == [("assistantResponseEvent", {"content": "OK"})]


def test_builder_id_streams_fixed_profile_arn(monkeypatch):
    instance = client.KiroClient()
    captured = {}
    creds = SimpleNamespace(api_region="us-east-1", access_token="token", profile_arn="", is_builder_id=True)
    monkeypatch.setattr(client, "get_credentials", lambda **_: creds)
    monkeypatch.setattr(client, "request", lambda method, url, **kwargs: captured.update(method=method, url=url, **kwargs) or SimpleNamespace())
    instance._open({"conversationState": {}})
    # The branded runtime gateway requires an ARN even though Builder ID has no profile.
    assert json.loads(captured["body"])["profileArn"] == client.BUILDER_ID_PROFILE_ARN


def test_idc_discovers_profile_arn_without_prompting(monkeypatch):
    instance = client.KiroClient()
    captured = {}
    calls = []
    creds = SimpleNamespace(api_region="us-east-1", access_token="token", profile_arn="", is_builder_id=False)
    monkeypatch.setattr(client, "get_credentials", lambda **_: creds)
    monkeypatch.setattr(client, "remember_profile_arn", lambda _: None)
    monkeypatch.setattr(client, "_management", lambda *_: calls.append(True) or {"profiles": [{"arn": "arn:aws:codewhisperer:us-east-1:1:profile/team"}]})
    monkeypatch.setattr(client, "request", lambda method, url, **kwargs: captured.update(method=method, url=url, **kwargs) or SimpleNamespace())
    instance._open({"conversationState": {}})
    assert calls and creds.profile_arn.endswith("profile/team")
    assert json.loads(captured["body"])["profileArn"] == creds.profile_arn


def test_builder_id_uses_live_bare_model_catalog_and_usage(monkeypatch):
    creds = SimpleNamespace(api_region="us-east-1", access_token="token", profile_arn="", is_builder_id=True)
    calls = []
    monkeypatch.setattr(client, "get_credentials", lambda **_: creds)
    monkeypatch.setattr(client, "_rest", lambda _creds, path, query: calls.append((path, query)) or ({"models": [{"modelId": "live-model"}]} if path == "ListAvailableModels" else {"usageBreakdownList": [{"usageType": "Agent", "currentUsage": 3, "usageLimit": 10}]}))
    assert client.list_model_ids() == ["live-model"]
    assert client.get_usage_limits()["usageBreakdownList"][0]["currentUsage"] == 3
    assert calls == [
        ("ListAvailableModels", {"origin": "AI_EDITOR", "maxResults": "50"}),
        ("getUsageLimits", {"isEmailRequired": "true", "origin": "AI_EDITOR", "resourceType": "AGENTIC_REQUEST"}),
    ]
    assert "3/10 (30%)" in client.format_usage({"usageBreakdownList": [{"usageType": "Agent", "currentUsage": 3, "usageLimit": 10}]})


def test_rest_requests_use_the_account_region_host(monkeypatch):
    urls = []
    monkeypatch.setattr(client, "request_json", lambda method, url, **_: urls.append(url) or {})
    for region in ("us-east-1", "eu-central-1"):
        client._rest(SimpleNamespace(api_region=region, access_token="token"), "getUsageLimits", {"origin": "AI_EDITOR"})
    assert urls == [
        "https://codewhisperer.us-east-1.amazonaws.com/getUsageLimits?origin=AI_EDITOR",
        "https://q.eu-central-1.amazonaws.com/getUsageLimits?origin=AI_EDITOR",
    ]


def test_model_discovery_failure_falls_back_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(client, "get_credentials", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level("WARNING"):
        assert client.list_model_ids() == list(client._FALLBACK)
    assert "boom" in caplog.text


def test_usage_splits_included_credit_from_overage():
    usage = client.format_usage({"usageBreakdownList": [{
        "displayName": "Credit", "displayNamePlural": "Credits",
        "currentUsage": 1860, "usageLimit": 1000,
        "currentOveragesWithPrecision": 860.55, "overageCapWithPrecision": 10000,
        "overageCharges": 34.4222627, "overageRate": 0.04, "currency": "USD",
        "nextDateReset": 1790812800.0,
    }]})
    assert "- Credit: 1,000/1,000 (100%); resets 2026-10-01 00:00 UTC" in usage
    assert "  Extra usage: 860.55/10,000 credits (9% of cap); $34.42 USD at $0.04/credit" in usage

    snapshot = client.build_usage_snapshot({"usageBreakdownList": [{
        "displayName": "Credit", "currentUsage": 1860, "usageLimit": 1000,
        "currentOveragesWithPrecision": 860.55, "overageCapWithPrecision": 10000,
        "overageCharges": 34.4222627, "currency": "USD", "nextDateReset": 1790812800.0,
    }]})
    assert snapshot.provider == "kiro" and snapshot.source == "kiro_usage_api"
    assert snapshot.windows[0].used_percent == 100
    assert snapshot.windows[0].detail == "1000/1000"
    assert snapshot.raw["usageBreakdownList"][0]["currentUsage"] == 1860
    assert snapshot.details == ("Credit extra usage: 860.55/10000; 34.4222627 USD",)


def test_native_auth_add_persists_opaque_pool_metadata(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _register_local_profile()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        creds = credentials.Credentials(
            "access", "refresh", "client", "secret", "us-east-1",
            BUILDER_ID_START_URL, time.time() + 3600, 4_102_444_800,
        )
        monkeypatch.setattr(credentials, "prompt_login_inputs", lambda *_: (BUILDER_ID_START_URL, "us-east-1"))
        monkeypatch.setattr(credentials, "_device_login", lambda *_: (creds, "https://verify", "CODE"))
        assert credentials.auth_handler("add", SimpleNamespace(label="work", priority=0)) is True
        assert credentials.auth_handler("status", SimpleNamespace()) is False
        row = load_pool("kiro").entries()[0]
        assert (row.auth_type, row.label, row.access_token, row.refresh_token) == (
            "oauth", "work", "access", "refresh",
        )
        assert row.extra == {
            "client_id": "client",
            "client_secret": "secret",
            "region": "us-east-1",
            "start_url": BUILDER_ID_START_URL,
            "client_secret_expires_at": 4_102_444_800,
        }
        assert row.base_url == credentials.RUNTIME_BASE_URL
        runtime = resolve_runtime_provider(requested="kiro", target_model="claude-haiku-4.5")
        assert runtime["base_url"] == credentials.RUNTIME_BASE_URL
        assert runtime["api_key"] == "access"
    finally:
        reset_hermes_home_override(token)


def test_plugin_refresh_rotates_pool_row_and_preserves_metadata(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _register_local_profile()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        entry = credentials.persist_credentials(credentials.Credentials(
            "old", "refresh", "client", "secret", "us-east-1",
            BUILDER_ID_START_URL, time.time() + 3600,
        ))
        calls = []
        monkeypatch.setattr(credentials, "_post", lambda *_: calls.append(True) or {
            "accessToken": "new", "refreshToken": "new-refresh", "expiresIn": 3600,
        })
        refreshed = load_pool("kiro").try_refresh_matching(credential_id=entry.id)
        assert calls == [True]
        assert (refreshed.access_token, refreshed.refresh_token) == ("new", "new-refresh")
        assert refreshed.extra["client_secret"] == "secret"
    finally:
        reset_hermes_home_override(token)


def test_mixed_region_rows_remain_eligible_for_failover(monkeypatch, tmp_path):
    from agent.credential_pool import credential_pool_entry_serves_endpoint
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    try:
        eu = credentials.persist_credentials(credentials.Credentials(
            "eu", "refresh", "client", "secret", "eu-central-1",
            "https://example.awsapps.com/start", time.time() + 3600,
        ))
        assert eu.base_url == credentials.RUNTIME_BASE_URL
        assert credential_pool_entry_serves_endpoint(eu, credentials.RUNTIME_BASE_URL)
    finally:
        reset_hermes_home_override(token)


def test_concurrent_401_refreshes_once(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _register_local_profile()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    token = set_hermes_home_override(tmp_path / "profile")
    creds = credentials.Credentials("old", "refresh", "id", "secret", "us-east-1", BUILDER_ID_START_URL, time.time() + 3600)
    credentials.persist_credentials(creds)
    barrier = threading.Barrier(2)
    refreshes, retried, errors = [], [], []

    class Response:
        def read(self, _):
            return b""
        def release_conn(self):
            return None

    def fake_request(*_, headers, **__):
        token = headers["Authorization"]
        if token == "Bearer old":
            barrier.wait(timeout=2)
            raise client.KiroHTTPError(401, b"expired")
        retried.append(token)
        return Response()

    monkeypatch.setattr(credentials, "_post", lambda *_: refreshes.append(True) or {"accessToken": "new", "expiresIn": 3600})
    monkeypatch.setattr(client, "get_credentials", credentials.get_credentials)
    monkeypatch.setattr(client, "request", fake_request)

    def run():
        try:
            list(client.KiroClient()._events({"conversationState": {}}))
        except Exception as exc:
            errors.append(exc)

    try:
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert refreshes == [True]
        assert retried == ["Bearer new", "Bearer new"]
    finally:
        reset_hermes_home_override(token)



def test_credentials_and_cache_follow_hermes_profile_context(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    alpha = tmp_path / "profiles" / "alpha"
    beta = tmp_path / "profiles" / "beta"
    monkeypatch.setenv("HERMES_HOME", str(alpha))
    token_alpha = set_hermes_home_override(alpha)
    try:
        credentials.persist_credentials(credentials.Credentials("alpha-token", "r", "id", "secret", "us-east-1", BUILDER_ID_START_URL, time.time() + 3600))
        assert credentials.state_dir() == alpha / "kiro"
        token_beta = set_hermes_home_override(beta)
        try:
            monkeypatch.setenv("HERMES_HOME", str(beta))
            credentials.persist_credentials(credentials.Credentials("beta-token", "r", "id", "secret", "us-east-1", BUILDER_ID_START_URL, time.time() + 3600))
            assert credentials.get_credentials().access_token == "beta-token"
        finally:
            reset_hermes_home_override(token_beta)
            monkeypatch.setenv("HERMES_HOME", str(alpha))
        assert credentials.get_credentials().access_token == "alpha-token"
    finally:
        reset_hermes_home_override(token_alpha)

def test_logout_removes_only_hermes_pool_state(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    try:
        credentials.persist_credentials(credentials.Credentials(
            "a", "r", "id", "secret", "us-east-1", BUILDER_ID_START_URL, time.time() + 60,
        ))
        assert credentials.logout() is True
        assert not load_pool("kiro").entries()
    finally:
        reset_hermes_home_override(token)


def test_native_auth_rejects_wrong_explicit_type():
    with pytest.raises(KiroAuthError, match="device-code OAuth"):
        credentials.auth_handler("add", SimpleNamespace(auth_type="api-key"))


def test_tool_calls_round_trip_and_nonstream_is_awaitable(monkeypatch):
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter([
        ("toolUseEvent", {"name": "read_file", "toolUseId": "call_1", "input": {"path": "x"}}),
        ("metadataEvent", {"stopReason": "TOOL_USE"}),
    ]))
    response = instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}])
    call = response.choices[0].message.tool_calls[0]
    assert response.choices[0].finish_reason == "tool_calls"
    assert (call.id, call.function.name, call.function.arguments) == ("call_1", "read_file", '{"path":"x"}')

    async def get_response():
        return await instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}])

    assert asyncio.run(get_response()).choices[0].message.tool_calls[0].function.name == "read_file"

    async def get_stream():
        chunks = []
        async for chunk in await instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}], stream=True):
            chunks.append(chunk)
        return chunks

    assert asyncio.run(get_stream())[-1].choices[0].finish_reason == "tool_calls"


def test_async_nonstream_does_not_block_event_loop(monkeypatch):
    instance = client.KiroClient()

    def slow_events(_):
        time.sleep(0.2)
        yield "assistantResponseEvent", {"content": "OK"}

    monkeypatch.setattr(instance, "_events", slow_events)

    async def run():
        started = time.monotonic()
        task = asyncio.create_task(instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}]))
        await asyncio.sleep(0.01)
        assert time.monotonic() - started < 0.1
        return await task

    assert asyncio.run(run()).choices[0].message.content == "OK"


_PNG_B64 = "iVBORw0KGgo="


def _image_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_image_parts_are_forwarded_on_current_history_and_tool_result_turns():
    request = build_request([
        {"role": "user", "content": [{"type": "text", "text": "first"}, _image_part(f"data:image/png;base64,{_PNG_B64}")]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "seen"}, _image_part(f"data:image/jpeg;base64,{_PNG_B64}")]},
    ], [{"type": "function", "function": {"name": "look", "parameters": {"type": "object"}}}], "claude-sonnet-4.5", None)
    state = request["conversationState"]
    history_images = state["history"][0]["userInputMessage"]["images"]
    current = state["currentMessage"]["userInputMessage"]
    assert history_images == [{"format": "png", "source": {"bytes": _PNG_B64}}]
    assert current["images"] == [{"format": "jpeg", "source": {"bytes": _PNG_B64}}]
    assert current["userInputMessageContext"]["toolResults"]


def test_unusable_image_parts_are_dropped():
    request = build_request([
        {"role": "user", "content": [
            {"type": "text", "text": "hi"},
            _image_part("https://example.com/a.png"),
            _image_part("data:image/svg+xml;base64,AAAA"),
            _image_part("data:image/png;base64,not base64!"),
        ]},
    ], None, "claude-sonnet-4.5", None)
    assert "images" not in request["conversationState"]["currentMessage"]["userInputMessage"]


def test_image_parts_on_historical_tool_turns_are_forwarded():
    request = build_request([
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "seen"}, _image_part(f"data:image/png;base64,{_PNG_B64}")]},
        {"role": "user", "content": "next"},
    ], None, "claude-sonnet-4.5", None)
    tool_turn = request["conversationState"]["history"][2]["userInputMessage"]
    assert tool_turn["userInputMessageContext"]["toolResults"]
    assert tool_turn["images"] == [{"format": "png", "source": {"bytes": _PNG_B64}}]


def test_tool_history_keeps_specs_and_does_not_mix_system_with_result():
    request = build_request([
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Read x"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read_file", "arguments": '{"path":"x"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp/x"},
    ], [], "claude-sonnet-4.5", None, "c")
    current = request["conversationState"]["currentMessage"]["userInputMessage"]
    tools = current["userInputMessageContext"]["tools"]
    result = current["userInputMessageContext"]["toolResults"][0]
    assert tools[0]["toolSpecification"]["name"] == "read_file"
    assert result["content"] == [{"text": "/tmp/x"}]
    assert "SYSTEM" in request["conversationState"]["history"][0]["userInputMessage"]["content"]


def test_parallel_tool_results_share_one_user_turn():
    request = build_request([
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Read x and y"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_x", "function": {"name": "read_file", "arguments": '{"path":"x"}'}},
            {"id": "call_y", "function": {"name": "read_file", "arguments": '{"path":"y"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_x", "content": "x"},
        {"role": "tool", "tool_call_id": "call_y", "content": "y"},
    ], [], "claude-sonnet-4.5", None, "c")
    state = request["conversationState"]
    assert len(state["history"]) == 2
    assert [item["text"] for item in state["currentMessage"]["userInputMessage"]["userInputMessageContext"]["toolResults"][0]["content"]] == ["x"]
    assert len(state["currentMessage"]["userInputMessage"]["userInputMessageContext"]["toolResults"]) == 2


def test_tool_schemas_strip_fields_kiro_rejects():
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {
            "path": {"type": "string", "additionalProperties": True},
            "options": {"type": "object", "required": ["deep"], "properties": {"deep": {"type": "boolean"}}},
        },
    }
    request = build_request([
        {"role": "user", "content": "Read x"},
    ], [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": parameters}}],
        "claude-sonnet-4.5", None, "c")
    spec = request["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]["tools"][0]["toolSpecification"]
    encoded = json.dumps(spec)
    assert "additionalProperties" not in encoded
    assert '"required": []' not in encoded
    assert spec["inputSchema"]["json"]["properties"]["options"]["required"] == ["deep"]
    assert parameters["additionalProperties"] is False  # caller's schema is not mutated


def test_schema_sanitizer_preserves_property_names_and_const_payloads():
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {
            "additionalProperties": {"type": "string"},
            "required": {"type": "boolean"},
            "mode": {"const": {"additionalProperties": False, "required": []}},
            "kind": {"enum": [{"additionalProperties": True, "required": []}]},
            "options": {
                "type": "object",
                "additionalProperties": False,
                "required": [],
                "properties": {"required": {"type": "string"}},
            },
        },
    }
    request = build_request([
        {"role": "user", "content": "Read x"},
    ], [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": parameters}}],
        "claude-sonnet-4.5", None, "c")
    schema = request["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]["tools"][0]["toolSpecification"]["inputSchema"]["json"]
    assert set(schema["properties"]) == {"additionalProperties", "required", "mode", "kind", "options"}
    assert schema["properties"]["additionalProperties"] == {"type": "string"}
    assert schema["properties"]["required"] == {"type": "boolean"}
    assert schema["properties"]["mode"]["const"] == {"additionalProperties": False, "required": []}
    assert schema["properties"]["kind"]["enum"] == [{"additionalProperties": True, "required": []}]
    assert "additionalProperties" not in schema["properties"]["options"]
    assert "required" not in schema["properties"]["options"]
    assert schema["properties"]["options"]["properties"] == {"required": {"type": "string"}}


@pytest.mark.parametrize("message", [
    "additionalModelRequestFields is not supported for this model",
    "Invalid additionalModelRequestFields: property 'enabled' is not defined in the schema and the schema does not allow additional properties",
])
def test_unsupported_additional_fields_are_cached_and_retried_once(monkeypatch, tmp_path, message):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        instance = client.KiroClient()
        bodies = []

        def fake_open(body, **_):
            bodies.append(json.loads(json.dumps(body)))
            if len(bodies) == 1:
                raise client.KiroHTTPError(400, json.dumps({
                    "reason": "REQUEST_BODY_INVALID",
                    "message": message,
                }).encode())
            return SimpleNamespace(read=lambda _: b"", release_conn=lambda: None)

        monkeypatch.setattr(instance, "_open", fake_open)
        body = build_request([{"role": "user", "content": "hi"}], [], "minimax-m2.5", {"reasoningEffort": "high"})
        assert list(instance._events(body)) == []
        assert "additionalModelRequestFields" in bodies[0]
        assert "additionalModelRequestFields" not in bodies[1]
        assert capabilities.additional_model_request_fields_supported("minimax-m2.5") is False
    finally:
        reset_hermes_home_override(token)


def test_cached_model_omits_additional_fields_for_sync_async_and_stream(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        capabilities.mark_additional_model_request_fields_unsupported("minimax-m2.5")
        instance = client.KiroClient()
        bodies = []

        def fake_events(body):
            bodies.append(body)
            yield "assistantResponseEvent", {"content": "OK"}

        monkeypatch.setattr(instance, "_events", fake_events)
        kwargs = {
            "model": "minimax-m2.5",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"reasoning": {"reasoningEffort": "high"}},
        }
        assert instance.chat.completions.create(**kwargs).choices[0].message.content == "OK"

        async def nonstream():
            return await instance.chat.completions.create(**kwargs)

        assert asyncio.run(nonstream()).choices[0].message.content == "OK"
        assert list(instance.chat.completions.create(**kwargs, stream=True))[-1].choices[0].finish_reason == "stop"
        assert all("additionalModelRequestFields" not in body for body in bodies)
    finally:
        reset_hermes_home_override(token)


def test_capability_cache_persists_across_module_restart_and_profiles(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    alpha = tmp_path / "profiles" / "alpha"
    beta = tmp_path / "profiles" / "beta"
    alpha_token = set_hermes_home_override(alpha)
    try:
        capabilities.mark_additional_model_request_fields_unsupported("minimax-m2.5")
        cache_file = credentials.state_dir() / "model-capabilities.json"
        assert cache_file.exists()
        assert "minimax-m2.5" in cache_file.read_text()
        del sys.modules["capabilities"]
        import capabilities as restarted_capabilities
        assert restarted_capabilities.additional_model_request_fields_supported("minimax-m2.5") is False

        beta_token = set_hermes_home_override(beta)
        try:
            assert restarted_capabilities.additional_model_request_fields_supported("minimax-m2.5") is True
        finally:
            reset_hermes_home_override(beta_token)
        assert restarted_capabilities.additional_model_request_fields_supported("minimax-m2.5") is False
    finally:
        reset_hermes_home_override(alpha_token)


def test_unrelated_400_is_not_cached_or_retried(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        instance = client.KiroClient()
        bodies = []

        def fake_open(body, **_):
            bodies.append(json.loads(json.dumps(body)))
            raise client.KiroHTTPError(400, b'{"reason":"REQUEST_BODY_INVALID","message":"tool schema is invalid"}')

        monkeypatch.setattr(instance, "_open", fake_open)
        body = build_request([{"role": "user", "content": "hi"}], [], "compatible-model", {"reasoningEffort": "high"})
        with pytest.raises(KiroAuthError, match="Kiro runtime failed"):
            list(instance._events(body))
        assert len(bodies) == 1
        assert "additionalModelRequestFields" in bodies[0]
        assert capabilities.additional_model_request_fields_supported("compatible-model") is True
    finally:
        reset_hermes_home_override(token)


def test_compatible_models_keep_reasoning_fields(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        instance = client.KiroClient()
        bodies = []

        def fake_open(body, **_):
            bodies.append(json.loads(json.dumps(body)))
            return SimpleNamespace(read=lambda _: b"", release_conn=lambda: None)

        monkeypatch.setattr(instance, "_open", fake_open)
        body = build_request([{"role": "user", "content": "hi"}], [], "compatible-model", {"reasoningEffort": "high"})
        assert list(instance._events(body)) == []
        assert "additionalModelRequestFields" in bodies[0]
        assert capabilities.additional_model_request_fields_supported("compatible-model") is True
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("payload", [
    {"additionalModelRequestFieldsUnsupportedModels": 42},
    {"additionalModelRequestFieldsUnsupportedModels": "not-a-list"},
    {"additionalModelRequestFieldsUnsupportedModels": ["valid", 42]},
    ["not-an-object"],
])
def test_invalid_capability_cache_schema_is_empty(payload, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        cache_file = credentials.state_dir() / "model-capabilities.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text(json.dumps(payload))
        assert capabilities.additional_model_request_fields_supported("valid") is True
    finally:
        reset_hermes_home_override(token)


def test_capability_cache_hits_memory_and_invalidates_external_changes(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        cache_file = credentials.state_dir() / "model-capabilities.json"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text(json.dumps({"additionalModelRequestFieldsUnsupportedModels": ["first"]}))
        reads = []
        original_read_text = capabilities.Path.read_text

        def spy_read_text(path, *args, **kwargs):
            if path == cache_file:
                reads.append(path)
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(capabilities.Path, "read_text", spy_read_text)
        assert capabilities.additional_model_request_fields_supported("first") is False
        assert capabilities.additional_model_request_fields_supported("first") is False
        assert len(reads) == 1

        replacement = cache_file.with_suffix(".replacement")
        replacement.write_text(json.dumps({"additionalModelRequestFieldsUnsupportedModels": ["second"]}))
        replacement.replace(cache_file)
        assert capabilities.additional_model_request_fields_supported("first") is True
        assert capabilities.additional_model_request_fields_supported("second") is False
        cache_file.unlink()
        assert capabilities.additional_model_request_fields_supported("second") is True
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("failure", ["read", "stat", "write"])
def test_cache_io_failures_do_not_abort_capability_fallback(monkeypatch, tmp_path, failure):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        instance = client.KiroClient()
        bodies = []

        def fake_open(body, **_):
            bodies.append(json.loads(json.dumps(body)))
            if len(bodies) == 1:
                raise client.KiroHTTPError(400, json.dumps({
                    "reason": "REQUEST_BODY_INVALID",
                    "message": "additionalModelRequestFields is not supported for this model",
                }).encode())
            return SimpleNamespace(read=lambda _: b"", release_conn=lambda: None)

        monkeypatch.setattr(instance, "_open", fake_open)
        if failure == "read":
            monkeypatch.setattr(capabilities.Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")))
        elif failure == "stat":
            monkeypatch.setattr(capabilities.Path, "stat", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stat failed")))
        else:
            monkeypatch.setattr(capabilities, "_atomic_write", lambda *_args: (_ for _ in ()).throw(OSError("write failed")))

        body = build_request([{"role": "user", "content": "hi"}], [], "failed-cache-model", {"reasoningEffort": "high"})
        assert list(instance._events(body)) == []
        assert len(bodies) == 2
        assert "additionalModelRequestFields" not in bodies[1]
        assert client.additional_model_request_fields_supported("failed-cache-model") is False
    finally:
        reset_hermes_home_override(token)


def test_concurrent_capability_updates_keep_all_models(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    import capabilities

    token = set_hermes_home_override(tmp_path / "profile")
    try:
        cache_file = tmp_path / "concurrent" / "model-capabilities.json"
        monkeypatch.setattr(capabilities, "_cache_path", lambda: cache_file)
        barrier = threading.Barrier(2)

        def mark(model):
            barrier.wait()
            capabilities.mark_additional_model_request_fields_unsupported(model)

        threads = [threading.Thread(target=mark, args=(model,)) for model in ("one", "two")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert set(json.loads(cache_file.read_text())["additionalModelRequestFieldsUnsupportedModels"]) == {"one", "two"}
    finally:
        reset_hermes_home_override(token)


def test_sync_caller_inside_running_loop_gets_a_real_response(monkeypatch):
    """Hermes auxiliary tasks (e.g. title generation) call the sync client from a thread with a running loop."""
    instance = client.KiroClient()
    monkeypatch.setattr(instance, "_events", lambda _: iter([("assistantResponseEvent", {"content": "OK"})]))

    async def run():
        response = instance.chat.completions.create(model="claude-sonnet-4.5", messages=[{"role": "user", "content": "hi"}])
        return response.choices[0].message.content

    assert asyncio.run(run()) == "OK"


@pytest.mark.parametrize(
    ("model", "config", "expected"),
    [
        ("claude-haiku-5.5", {"enabled": False}, {"thinking": {"type": "disabled"}}),
        ("claude-haiku-5.5", {"enabled": True, "effort": "medium"}, {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}),
        ("claude-opus-5.5", {"effort": "ultra"}, {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}),
        ("claude-sonnet-5.5", {"effort": "minimal"}, {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}),
        ("gpt-5.6-terra", {"enabled": False}, {"reasoning": {"effort": "none"}}),
        ("gpt-5.6-terra", {"effort": "high"}, {"reasoning": {"effort": "high"}}),
        ("minimax-m2.5", {"enabled": False}, None),
        ("claude-haiku-5.5", {"thinking": {"type": "adaptive"}}, {"thinking": {"type": "adaptive"}}),
        ("claude-haiku-5.5", None, None),
        ("compatible-model", {"reasoningEffort": "high"}, {"reasoningEffort": "high"}),
    ],
)
def test_generic_reasoning_config_maps_to_kiro_schema(model, config, expected):
    """Hermes sends ``{"enabled", "effort"}``; Kiro 400s on anything but its per-family fields."""
    assert client._kiro_reasoning_fields(model, config) == expected


def test_title_style_disabled_thinking_reaches_wire_in_kiro_shape(monkeypatch):
    instance = client.KiroClient()
    bodies = []

    def fake_events(body):
        bodies.append(body)
        yield "assistantResponseEvent", {"content": "OK"}

    monkeypatch.setattr(instance, "_events", fake_events)
    response = instance.chat.completions.create(
        model="claude-haiku-5.5",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"enabled": False}},
    )
    assert response.choices[0].message.content == "OK"
    assert bodies[0]["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}


def test_model_without_disabled_thinking_retries_adaptive_low(monkeypatch):
    instance = client.KiroClient()
    bodies = []

    def fake_open(body, **_):
        bodies.append(body)
        if len(bodies) == 1:
            raise client.KiroHTTPError(400, json.dumps({
                "reason": "REQUEST_BODY_INVALID",
                "message": 'Invalid additionalModelRequestFields: does not have a value in the enumeration ["adaptive"]',
            }).encode())
        return SimpleNamespace(read=lambda _: b"", release_conn=lambda: None)

    monkeypatch.setattr(instance, "_open", fake_open)
    body = build_request([{"role": "user", "content": "hi"}], [], "claude-opus-5.5", {"thinking": {"type": "disabled"}})
    assert list(instance._events(body)) == []
    assert bodies[1]["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}


SOCIAL = "https://prod.us-east-1.auth.desktop.kiro.dev"
ARN = "arn:aws:codewhisperer:us-east-1:000000000000:profile/FAKE"
GOOGLE = ("social:google", "us-east-1")
DEVICE = {
    "deviceCode": "fake-device", "userCode": "FAKE-0000", "verificationUri": "https://kiro.invalid/device",
    "verificationUriComplete": "https://kiro.invalid/device?user_code=FAKE-0000",
    "expiresInMilliseconds": 300000, "intervalInMilliseconds": 5000,
}
PENDING = {"status": "authorization_pending", "accessToken": None, "refreshToken": None}
AUTHORIZED = {"status": "authorized", "accessToken": "fake-access", "refreshToken": "fake-refresh", "profileArn": ARN}
SLOW_DOWN = {"status": "slow_down"}


def fake_net(monkeypatch, *replies):
    """Replace credentials.request_json: record each call, replay replies or raise exceptions."""
    net = SimpleNamespace(calls=[], replies=list(replies))

    def fake(method, url, *, body=None, headers=None, timeout=30):
        net.calls.append(SimpleNamespace(method=method, url=url, body=json.loads(body), headers=headers))
        reply = net.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply() if callable(reply) else reply

    monkeypatch.setattr(credentials, "request_json", fake)
    return net


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Temp Hermes home with the provider registered, plus a fake credentials.time that records sleeps."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _register_local_profile()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    token = set_hermes_home_override(tmp_path / "profile")
    # credentials.py calls only time.monotonic, time.sleep and time.time, so this
    # namespace stands in for the whole module; sleeping advances the fake clock.
    clock = SimpleNamespace(now=0.0, sleeps=[], base=time.time())
    clock.monotonic, clock.time = lambda: clock.now, lambda: clock.base + clock.now
    clock.sleep = lambda seconds: (clock.sleeps.append(seconds), setattr(clock, "now", clock.now + seconds))
    monkeypatch.setattr(credentials, "time", clock)
    yield clock
    reset_hermes_home_override(token)


def rows():
    from agent.credential_pool import load_pool

    return load_pool("kiro").entries()


def advance(env, seconds, then=PENDING):
    """Build a fake reply that moves the fake clock forward by `seconds` and then answers `then`."""
    def reply():
        env.now += seconds
        return then

    return reply


def social_row(monkeypatch):
    net = fake_net(monkeypatch, DEVICE, AUTHORIZED)
    credentials.login(*GOOGLE)
    return rows()[0], net


@pytest.mark.parametrize("choice,start_url", [
    ("3", "social:google"),
    ("4", "social:github"),
    ("0", None),
    ("6", None),
    ("x", None),
])
def test_menu_choices_3_and_4_pick_social_login(choice, start_url):
    """Answers 3 and 4 select Google or GitHub without a region prompt; other answers name all four methods."""
    prompts = []
    ask = lambda text: prompts.append(text) or choice
    if start_url:
        assert prompt_login_inputs(None, None, ask) == (start_url, "us-east-1") and len(prompts) == 1
    else:
        with pytest.raises(KiroAuthError) as caught:
            prompt_login_inputs(None, None, ask)
        expected = "Choose 1 for AWS Builder ID, 2 for IAM Identity Center, 3 for Google, 4 for GitHub or 5 for Google / GitHub (import the kiro-cli login)"
        assert str(caught.value) == expected


@pytest.mark.parametrize("choice,wire,label,standalone", [
    ("3", "Google", "kiro-google", False),
    ("4", "Github", "kiro-github", False),
    ("4", "Github", "kiro-github", True),
])
def test_social_login_stores_row_for_google_and_github(monkeypatch, capsys, env, choice, wire, label, standalone):
    """`auth add` and the standalone login post the device-flow requests, print the code and store a social row."""
    net = fake_net(monkeypatch, DEVICE, PENDING, PENDING, AUTHORIZED)
    monkeypatch.setattr(sys, "stdin", io.StringIO(choice + "\n"))
    monkeypatch.setattr(sys, "argv", ["credentials.py", "login"])
    if standalone:
        credentials.main()
    else:
        assert credentials.auth_handler("add", SimpleNamespace(label=None, priority=None)) is True
    auth, *polls = net.calls
    assert (auth.method, auth.url) == ("POST", f"{SOCIAL}/oauth/device/authorization")
    assert auth.body == {"clientId": "hermes-kiro", "loginProvider": wire}
    assert auth.headers["Content-Type"] == "application/json"
    poll = ("POST", f"{SOCIAL}/oauth/device/poll", {"deviceCode": "fake-device", "clientId": "hermes-kiro"})
    assert [(c.method, c.url, c.body) for c in polls] == [poll] * 3
    assert env.sleeps == [5.0, 5.0]
    assert "Open: https://kiro.invalid/device?user_code=FAKE-0000\nCode: FAKE-0000\n" in capsys.readouterr().out
    (row,) = rows()
    assert (row.label, row.source, row.access_token, row.refresh_token) == (
        label, "manual:kiro_social_device_code", "fake-access", "fake-refresh",
    )
    assert row.extra == {"kiro_login": "social", "region": "us-east-1", "profile_arn": ARN}
    assert row.base_url == credentials.RUNTIME_BASE_URL
    assert row.expires_at_ms == int((env.time() + 3600) * 1000)


@pytest.mark.parametrize("status", ["invalid_token", "expired", "denied", "unlisted", "x" * 100])
def test_unexpected_poll_status_stops_the_login_without_more_polls(monkeypatch, env, status):
    """Any other poll status ends the login, is named cut to 40 characters, is kept whole as the code."""
    net = fake_net(monkeypatch, DEVICE, PENDING, {"status": status}, PENDING, AUTHORIZED)
    with pytest.raises(KiroAuthError) as caught:
        credentials.login(*GOOGLE)
    assert f"ended with status {status[:40]!r};" in str(caught.value)
    assert caught.value.code == status
    assert len(net.calls) == 3 and env.sleeps == [5.0]
    assert rows() == []


def test_slow_down_lengthens_the_sleep_by_two_seconds_each_time(monkeypatch, env):
    """Every `slow_down` adds 2 seconds to the interval and polling goes on until `authorized`."""
    net = fake_net(monkeypatch, DEVICE, PENDING, SLOW_DOWN, SLOW_DOWN, PENDING, AUTHORIZED)
    credentials.login(*GOOGLE)
    assert env.sleeps == [5.0, 7.0, 9.0, 9.0]
    assert len(net.calls) == 6 and len(rows()) == 1


@pytest.mark.parametrize("reply", [{}, {"status": None}, {"status": ""}])
def test_poll_without_a_status_fails_with_its_own_message(monkeypatch, env, reply):
    """A poll reply with a missing or empty status ends the login with a message that says no status came back."""
    net = fake_net(monkeypatch, DEVICE, reply, PENDING, AUTHORIZED)
    with pytest.raises(KiroAuthError, match="returned no status; run `hermes auth add kiro` again") as caught:
        credentials.login(*GOOGLE)
    assert caught.value.code == ""
    assert len(net.calls) == 2 and rows() == []


@pytest.mark.parametrize("interval,reply,sleeps", [
    (5000, PENDING, [5.0, 5.0, 2.0]),
    (1e9, PENDING, [12.0]),
    (5000, SLOW_DOWN, [7.0, 5.0]),
])
def test_unanswered_login_expires_at_the_server_deadline(monkeypatch, env, interval, reply, sleeps):
    """An unanswered login ends at the server deadline, with or without `slow_down`, and no sleep passes it."""
    device = {**DEVICE, "expiresInMilliseconds": 12000, "intervalInMilliseconds": interval}
    fake_net(monkeypatch, device, *[reply] * 10)
    with pytest.raises(KiroAuthError, match="expired"):
        credentials.login(*GOOGLE)
    assert env.sleeps == sleeps and env.now == 12.0


def test_poll_that_uses_up_the_remaining_time_expires_without_sleeping(monkeypatch, env):
    """A pending poll that ends past the deadline raises the expired error with no further sleep."""
    fake_net(monkeypatch, {**DEVICE, "expiresInMilliseconds": 10000}, advance(env, 10))
    with pytest.raises(KiroAuthError, match="expired"):
        credentials.login(*GOOGLE)
    assert env.sleeps == []


@pytest.mark.parametrize("interval", [0, 200])
def test_poll_interval_never_drops_below_one_second(monkeypatch, env, interval):
    """A zero or tiny server interval still sleeps at least one second."""
    fake_net(monkeypatch, {**DEVICE, "intervalInMilliseconds": interval}, PENDING, AUTHORIZED)
    credentials.login(*GOOGLE)
    assert env.sleeps and min(env.sleeps) >= 1.0


@pytest.mark.parametrize("missing", ["accessToken", "refreshToken", "profileArn"])
def test_incomplete_authorization_stores_nothing(monkeypatch, env, missing):
    """An `authorized` reply without a token or profile ARN raises and persists nothing."""
    fake_net(monkeypatch, DEVICE, {**AUTHORIZED, missing: None})
    with pytest.raises(KiroAuthError, match="omitted"):
        credentials.login(*GOOGLE)
    assert rows() == []


@pytest.mark.parametrize("start_url,replies,match,status,code", [
    ("social:google", [client.KiroHTTPError(400, b'{"error": "bad_provider"}')], r"failed \(400\)", 400,
     "bad_provider"),
    ("social:google", [DEVICE, OSError("no route")], "unreachable", None, ""),
    ("social:google", [{"userCode": "FAKE-0000"}], "device authorization details", None, ""),
    ("social:cognito", [], "Unsupported Kiro social login", None, ""),
])
def test_login_transport_and_protocol_errors_raise_kiro_auth_error(
    monkeypatch, env, start_url, replies, match, status, code
):
    """HTTP 400, a network failure, a reply without device details and an unknown provider raise KiroAuthError."""
    fake_net(monkeypatch, *replies)
    with pytest.raises(KiroAuthError, match=match) as caught:
        credentials.login(start_url, "us-east-1")
    assert (caught.value.status, caught.value.code) == (status, code)
    assert rows() == []


def test_login_label_default_trim_and_priority_zero_across_logins(monkeypatch, env):
    """A blank label gets the default, a padded one is trimmed, priority 0 goes first, each login adds a row."""
    fake_net(monkeypatch, *[DEVICE, AUTHORIZED] * 3)
    credentials.login(*GOOGLE, label="   ")
    credentials.login("social:github", "us-east-1", label="  work ")
    credentials.login(*GOOGLE, label="last", priority=0)
    assert [row.label for row in rows()] == ["last", "kiro-google", "work"]


@pytest.mark.parametrize("extra,refresh", [
    ({}, "fake-refresh"),
    ({"refreshToken": "rotated"}, "rotated"),
])
def test_social_refresh_stores_returned_token_and_keeps_metadata(monkeypatch, env, extra, refresh):
    """A pool refresh posts the refresh token, stores the fresh access token and keeps or rotates the refresh token."""
    row, net = social_row(monkeypatch)
    net.replies.append({"accessToken": "fresh-access", "expiresIn": 120, **extra})
    assert credentials.get_credentials(force_refresh=True).access_token == "fresh-access"
    (out,) = rows()
    sent = net.calls[-1]
    assert (sent.method, sent.url, sent.body) == ("POST", f"{SOCIAL}/refreshToken", {"refreshToken": "fake-refresh"})
    assert (out.access_token, out.refresh_token, out.extra) == ("fresh-access", refresh, row.extra)
    assert out.expires_at_ms == int((env.time() + 120) * 1000)


@pytest.mark.parametrize("reply,error,relogin", [
    pytest.param(client.KiroHTTPError(400, b"{}"), AuthError, True, id="http-400"),
    pytest.param(client.KiroHTTPError(401, b"{}"), AuthError, True, id="http-401"),
    pytest.param(client.KiroHTTPError(403, b"{}"), AuthError, True, id="http-403"),
    pytest.param(client.KiroHTTPError(429, b"slow"), KiroAuthError, False, id="http-429"),
    pytest.param(client.KiroHTTPError(503, b"down"), KiroAuthError, False, id="http-503"),
    pytest.param(OSError("no route"), KiroAuthError, False, id="network-error"),
    pytest.param({"expiresIn": 3600}, KiroAuthError, False, id="no-access-token"),
])
def test_social_refresh_failures_split_into_relogin_and_retryable(monkeypatch, env, reply, error, relogin):
    """400, 401 and 403 need a fresh login; 429, 503, a network failure and a missing access token do not."""
    row, net = social_row(monkeypatch)
    net.replies.append(reply)
    with pytest.raises(error) as caught:
        credentials.refresh_credential(row)
    assert net.calls[-1].url == f"{SOCIAL}/refreshToken"
    assert getattr(caught.value, "relogin_required", False) is relogin


def test_social_refresh_without_a_refresh_token_asks_for_a_fresh_login(monkeypatch, env):
    """An empty refresh token raises the relogin error without any request."""
    row, net = social_row(monkeypatch)
    row.refresh_token = ""
    sent = len(net.calls)
    with pytest.raises(AuthError) as caught:
        credentials.refresh_credential(row)
    assert caught.value.relogin_required is True
    assert len(net.calls) == sent


def test_idc_refresh_failure_requires_relogin_and_keeps_the_oidc_error_text(monkeypatch, env):
    """A rejected IdC refresh asks for a fresh login and its text begins `AWS OIDC token failed (400)`."""
    expiry = time.time() + 3600
    row = credentials.persist_credentials(
        credentials.Credentials("stale", "r", "id", "secret", "us-east-1", BUILDER_ID_START_URL, expiry)
    )
    net = fake_net(monkeypatch, client.KiroHTTPError(400, b'{"error": "invalid_grant"}'))
    with pytest.raises(AuthError) as caught:
        credentials.refresh_credential(row)
    assert net.calls[0].url == "https://oidc.us-east-1.amazonaws.com/token"
    assert caught.value.relogin_required is True
    assert str(caught.value).startswith("AWS OIDC token failed (400)")


@pytest.mark.parametrize("key", ["profile_arn", "region"])
def test_social_row_missing_metadata_asks_for_a_fresh_login(monkeypatch, env, key):
    """A social row without profile_arn or region raises the incomplete-metadata error."""
    row, _ = social_row(monkeypatch)
    row.extra.pop(key)
    with pytest.raises(KiroAuthError, match=f"{key}.*hermes auth add kiro"):
        credentials.refresh_credential(row)


def test_social_row_reaches_the_runtime_request_without_profile_discovery(monkeypatch, env):
    """The client sends the row's profileArn and never calls profile discovery."""
    social_row(monkeypatch)
    sent = {}
    fake_request = lambda method, url, **kwargs: sent.update(url=url, **kwargs) or SimpleNamespace()
    monkeypatch.setattr(client, "_management", lambda *_: pytest.fail("profile discovery must not run"))
    monkeypatch.setattr(client, "request", fake_request)
    client.KiroClient()._open({"conversationState": {}})
    assert json.loads(sent["body"])["profileArn"] == ARN
    assert sent["headers"]["Authorization"] == "Bearer fake-access"


# --- kiro-cli import and bonus credits (local additions on top of upstream) ------------------

_CLI_ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/TESTPROFILE"


def _kiro_cli_db(path, **overrides):
    import sqlite3

    token = {
        "access_token": "cli-access", "refresh_token": "cli-refresh",
        "expires_at": "2099-01-01T00:00:00.123456789Z", "provider": "google", "profile_arn": _CLI_ARN,
    }
    token.update(overrides)
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO auth_kv VALUES (?, ?)", ("kirocli:social:token", json.dumps(token)))
    con.close()
    return path


def test_kiro_cli_token_is_read_without_modifying_the_database(tmp_path):
    db = _kiro_cli_db(tmp_path / "data.sqlite3")
    before = db.read_bytes()
    creds, provider = credentials.read_kiro_cli_social(db)
    assert db.read_bytes() == before
    assert (provider, creds.region, creds.is_social) == ("google", "us-east-1", True)
    assert (creds.access_token, creds.refresh_token, creds.profile_arn) == ("cli-access", "cli-refresh", _CLI_ARN)
    assert creds.expires_at == pytest.approx(4_070_908_800.123456)


def test_kiro_cli_token_errors_never_leak_token_values(tmp_path):
    import sqlite3

    with pytest.raises(KiroAuthError, match="kiro-cli login"):
        credentials.read_kiro_cli_social(tmp_path / "missing.sqlite3")
    empty = tmp_path / "empty.sqlite3"
    con = sqlite3.connect(empty)
    con.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT)")
    con.commit()
    con.close()
    with pytest.raises(KiroAuthError, match="no Google/GitHub sign-in"):
        credentials.read_kiro_cli_social(empty)
    for index, bad in enumerate(({"refresh_token": ""}, {"profile_arn": "not-an-arn"})):
        with pytest.raises(KiroAuthError, match="incomplete") as caught:
            credentials.read_kiro_cli_social(_kiro_cli_db(tmp_path / f"bad{index}.sqlite3", **bad))
        assert "cli-refresh" not in str(caught.value) and "cli-access" not in str(caught.value)
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"this is not a sqlite database" * 20)
    with pytest.raises(KiroAuthError, match="Could not read") as caught:
        credentials.read_kiro_cli_social(garbage)
    assert "this is not" not in str(caught.value)


def test_menu_choice_five_imports_kiro_cli_without_asking_for_a_region():
    asked = []
    values = iter(["5"])
    assert prompt_login_inputs(None, None, lambda text: asked.append(text) or next(values)) == (
        credentials.KIRO_CLI_LOGIN, "us-east-1")
    assert len(asked) == 1


def test_kiro_cli_import_round_trips_through_the_pool(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("KIRO_CLI_DB", str(_kiro_cli_db(tmp_path / "data.sqlite3")))
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        posts = []

        def fake_request(service, url, payload):
            posts.append((url, payload))
            return {"accessToken": f"access-{len(posts)}", "expiresIn": 3600}

        monkeypatch.setattr(credentials, "_request", fake_request)
        monkeypatch.setattr(credentials, "prompt_login_inputs", lambda *_: (credentials.KIRO_CLI_LOGIN, "us-east-1"))
        assert credentials.auth_handler("add", SimpleNamespace(label=None, priority=0)) is True
        assert posts == [(f"{credentials.SOCIAL_BASE_URL}/refreshToken", {"refreshToken": "cli-refresh"})]

        row = load_pool("kiro").entries()[0]
        assert (row.label, row.access_token, row.refresh_token) == ("kiro-google", "access-1", "cli-refresh")
        # Same row shape as upstream's device login, so one code path reads both.
        assert row.extra == {"kiro_login": "social", "region": "us-east-1", "profile_arn": _CLI_ARN}
        creds = credentials.get_credentials()
        assert (creds.is_social, creds.profile_arn, creds.access_token) == (True, _CLI_ARN, "access-1")
    finally:
        reset_hermes_home_override(token)


def test_row_saved_by_the_earlier_kiro_cli_import_still_loads(tmp_path):
    from agent.credential_pool import PooledCredential

    legacy = PooledCredential(
        provider="kiro", id="abc123", label="kiro-google", auth_type="oauth", priority=0,
        source="manual:kiro_device_code", access_token="a", refresh_token="r", expires_at_ms=4_070_908_800_000,
        extra={"auth_method": "social", "social_provider": "google", "region": "us-east-1", "profile_arn": _CLI_ARN},
    )
    creds = credentials._entry_credentials(legacy)
    assert (creds.is_social, creds.profile_arn, creds.region, creds.refresh_token) == (True, _CLI_ARN, "us-east-1", "r")


def test_usage_reports_active_bonus_credits():
    data = {"usageBreakdownList": [{
        "displayName": "Credit", "currentUsage": 0, "usageLimit": 50, "nextDateReset": 1796083200.0,
        "bonuses": [
            {"displayName": "Promo", "status": "ACTIVE", "currentUsage": 208.55, "usageLimit": 1000, "expiresAt": 1796112000.0},
            {"displayName": "Old", "status": "EXPIRED", "currentUsage": 5, "usageLimit": 10, "expiresAt": 1.0},
        ],
    }]}
    usage = client.format_usage(data)
    assert "  Bonus Promo: 208.55/1,000 (21%), 791.45 left; expires 2026-12-01 08:00 UTC" in usage
    assert "Old" not in usage
    snapshot = client.build_usage_snapshot(data)
    assert [w.label for w in snapshot.windows] == ["Credit", "Credit bonus: Promo"]
    assert snapshot.windows[1].detail == "208.55/1000"
