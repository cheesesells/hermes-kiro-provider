from __future__ import annotations

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
    prompts = []
    values = iter(["", ""])
    prompt_login_inputs(None, None, lambda text: prompts.append(text) or next(values))
    assert "\n  1. AWS Builder ID\n  2. IAM Identity Center\n  3. Google / GitHub (use kiro-cli login)\n\n" in prompts[0]
    assert _next_login_choice(1, "down") == 2 and _next_login_choice(2, "down") == 2


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
        "supports_vision": False,
        "supports_tools": True,
    }
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


def test_builder_id_request_omits_profile_arn(monkeypatch):
    instance = client.KiroClient()
    captured = {}
    creds = SimpleNamespace(api_region="us-east-1", access_token="token", profile_arn="", is_builder_id=True)
    monkeypatch.setattr(client, "get_credentials", lambda **_: creds)
    monkeypatch.setattr(client, "request", lambda method, url, **kwargs: captured.update(method=method, url=url, **kwargs) or SimpleNamespace())
    instance._open({"conversationState": {}})
    assert "profileArn" not in json.loads(captured["body"])


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


def test_unsupported_additional_fields_are_cached_and_retried_once(monkeypatch, tmp_path):
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


# --- Google / GitHub sign-in imported from kiro-cli ---------------------------------------

_SOCIAL_ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/TESTPROFILE"


def _kiro_cli_db(path, **overrides):
    import sqlite3

    token = {
        "access_token": "social-access", "refresh_token": "social-refresh",
        "expires_at": "2099-01-01T00:00:00.123456789Z", "provider": "google", "profile_arn": _SOCIAL_ARN,
    }
    token.update(overrides)
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO auth_kv VALUES (?, ?)", ("kirocli:social:token", json.dumps(token)))
    con.close()
    return path


def test_kiro_cli_social_token_is_read_without_modifying_the_database(tmp_path):
    db = _kiro_cli_db(tmp_path / "data.sqlite3")
    before = db.read_bytes()
    creds = credentials.read_kiro_cli_social(db)
    assert db.read_bytes() == before
    assert (creds.auth_method, creds.social_provider, creds.region) == ("social", "google", "us-east-1")
    assert (creds.access_token, creds.refresh_token, creds.profile_arn) == ("social-access", "social-refresh", _SOCIAL_ARN)
    assert creds.is_social and not creds.is_builder_id
    assert creds.expires_at == pytest.approx(4_070_908_800.123456)


def test_kiro_cli_social_token_errors_never_leak_token_values(tmp_path):
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
    for bad in ({"refresh_token": ""}, {"profile_arn": "not-an-arn"}):
        with pytest.raises(KiroAuthError, match="incomplete") as caught:
            credentials.read_kiro_cli_social(_kiro_cli_db(tmp_path / f"bad{len(bad['refresh_token'] if 'refresh_token' in bad else 'x')}.sqlite3", **bad))
        assert "social-refresh" not in str(caught.value) and "social-access" not in str(caught.value)
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"this is not a sqlite database" * 20)
    with pytest.raises(KiroAuthError, match="Could not read") as caught:
        credentials.read_kiro_cli_social(garbage)
    assert "this is not" not in str(caught.value)


def test_social_refresh_posts_refresh_token_to_the_desktop_auth_service(monkeypatch):
    seen = {}

    def fake_request_json(method, url, *, body=None, headers=None, **_):
        seen.update(method=method, url=url, body=json.loads(body), headers=headers)
        return {"accessToken": "a2", "refreshToken": "r2", "expiresIn": 3600, "profileArn": _SOCIAL_ARN}

    monkeypatch.setattr(credentials, "request_json", fake_request_json)
    out = credentials._refresh_social_tokens("us-east-1", "social-refresh")
    assert out["accessToken"] == "a2"
    assert seen["method"] == "POST"
    assert seen["url"] == "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken"
    assert seen["body"] == {"refreshToken": "social-refresh"}
    assert seen["headers"]["User-Agent"].startswith("KiroIDE-")


@pytest.mark.parametrize(("status", "terminal"), [(400, True), (401, True), (403, True), (429, False), (503, False)])
def test_social_refresh_failures_only_force_relogin_when_the_token_is_dead(monkeypatch, tmp_path, status, terminal):
    from hermes_cli.auth_constants import AuthError
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    try:
        entry = credentials.persist_credentials(credentials.read_kiro_cli_social(_kiro_cli_db(tmp_path / "d.sqlite3")))
        monkeypatch.setattr(credentials, "request_json", lambda *_a, **_k: (_ for _ in ()).throw(
            credentials.KiroHTTPError(status, b'{"message":"nope"}')))
        expected = AuthError if terminal else KiroAuthError
        with pytest.raises(expected) as caught:
            credentials.refresh_credential(entry)
        assert isinstance(caught.value, AuthError) == terminal
        assert "social-refresh" not in str(caught.value)
    finally:
        reset_hermes_home_override(token)


def test_login_menu_choice_three_imports_kiro_cli_without_asking_for_a_region():
    asked = []
    values = iter(["3"])
    assert prompt_login_inputs(None, None, lambda text: asked.append(text) or next(values)) == (
        credentials.SOCIAL_LOGIN, "us-east-1")
    assert len(asked) == 1


def test_social_login_round_trips_through_the_credential_pool(monkeypatch, tmp_path):
    from agent.credential_pool import load_pool
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _register_local_profile()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("KIRO_CLI_DB", str(_kiro_cli_db(tmp_path / "data.sqlite3")))
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        posts = []

        def fake_refresh(region, refresh_token):
            posts.append((region, refresh_token))
            return {"accessToken": f"access-{len(posts)}", "expiresIn": 3600, "profileArn": _SOCIAL_ARN}

        monkeypatch.setattr(credentials, "_refresh_social_tokens", fake_refresh)
        monkeypatch.setattr(credentials, "prompt_login_inputs", lambda *_: (credentials.SOCIAL_LOGIN, "us-east-1"))
        assert credentials.auth_handler("add", SimpleNamespace(label=None, priority=0)) is True
        assert posts == [("us-east-1", "social-refresh")]  # login proves the token works once

        row = load_pool("kiro").entries()[0]
        assert (row.label, row.access_token, row.refresh_token) == ("kiro-google", "access-1", "social-refresh")
        assert row.extra == {"auth_method": "social", "social_provider": "google", "region": "us-east-1", "profile_arn": _SOCIAL_ARN}
        assert "client_secret" not in row.extra

        creds = credentials.get_credentials()
        assert (creds.is_social, creds.profile_arn, creds.access_token) == (True, _SOCIAL_ARN, "access-1")

        refreshed = load_pool("kiro").try_refresh_matching(credential_id=row.id)
        assert posts[-1] == ("us-east-1", "social-refresh")
        assert (refreshed.access_token, refreshed.refresh_token) == ("access-2", "social-refresh")
        assert refreshed.extra["profile_arn"] == _SOCIAL_ARN and refreshed.extra["auth_method"] == "social"
    finally:
        reset_hermes_home_override(token)


def test_incomplete_social_pool_row_asks_for_a_new_login(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    try:
        creds = credentials.read_kiro_cli_social(_kiro_cli_db(tmp_path / "d.sqlite3"))
        creds.profile_arn = ""
        entry = credentials.persist_credentials(creds)
        with pytest.raises(KiroAuthError, match="profile_arn"):
            credentials._entry_credentials(entry)
    finally:
        reset_hermes_home_override(token)
