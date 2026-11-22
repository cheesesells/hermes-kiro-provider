"""Native Kiro provider for Hermes."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from client import KiroClient, build_usage_snapshot, get_usage_limits, list_model_ids  # noqa: E402
from credentials import auth_handler, refresh_credential  # noqa: E402

_FALLBACK_MODELS = ("claude-sonnet-4.5", "claude-haiku-4.5", "gpt-5.6-terra")
_CATALOG_MODELS = _FALLBACK_MODELS
_CATALOG_AT = 0.0
# Kiro rejects image blocks for its open-weight families; unknown ids default to
# text-only so Hermes summarizes the image instead of sending a block Kiro 400s.
_VISION_PREFIXES = ("claude", "gpt")


def _supports_vision(model: str) -> bool:
    """Claude and GPT models on Kiro accept inline images; ``auto`` routes to one of them."""
    name = str(model or "").strip().lower()
    return name == "auto" or name.startswith(_VISION_PREFIXES)


class _ModelCapabilities(dict):
    """Per-model capabilities, answered for any catalog id (the live catalog changes under us)."""

    def get(self, model, default=None):
        if model in self:
            return self[model]
        return {"supports_vision": _supports_vision(model), "supports_tools": True} if model else default


_MODEL_CAPABILITIES = _ModelCapabilities(
    {model: {"supports_vision": _supports_vision(model), "supports_tools": True} for model in _FALLBACK_MODELS}
)


def _catalog_models() -> tuple[str, ...]:
    """Cache the live catalog while keeping offline picker fallbacks."""
    global _CATALOG_MODELS, _CATALOG_AT
    if time.monotonic() - _CATALOG_AT >= 300:
        try:
            if models := tuple(list_model_ids()):
                _CATALOG_MODELS = models
        except Exception:
            pass
        _CATALOG_AT = time.monotonic()
    return _CATALOG_MODELS


class KiroProfile(ProviderProfile):
    @property
    def fallback_models(self) -> tuple[str, ...]:
        # Core reads fallback_models directly for non-api-key provider pickers.
        return _catalog_models()

    @fallback_models.setter
    def fallback_models(self, value: tuple[str, ...]) -> None:
        global _CATALOG_MODELS
        _CATALOG_MODELS = tuple(value or _FALLBACK_MODELS)

    def create_client(self, **kwargs):
        return KiroClient(**kwargs)

    def fetch_models(self, **kwargs):
        return list_model_ids() or list(_FALLBACK_MODELS)

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        return build_usage_snapshot(get_usage_limits())


profile = KiroProfile(
    name="kiro",
    aliases=("kiro-native",),
    display_name="Kiro (native)",
    description="Kiro via direct HTTPS; sign in with AWS Builder ID, IAM Identity Center, Google or GitHub.",
    signup_url="https://kiro.dev",
    auth_type="oauth_device_code",
    auth_handler=auth_handler,
    refresh_credential=refresh_credential,
    base_url="https://runtime.us-east-1.kiro.dev",
    hostname="runtime.us-east-1.kiro.dev",
    supports_health_check=False,
    # Image parts in tool results are forwarded as Kiro image blocks (see translate._images).
    supports_vision=True,
    fallback_models=_FALLBACK_MODELS,
    model_capabilities=_MODEL_CAPABILITIES,
    default_aux_model="claude-haiku-4.5",
)
register_provider(profile)


def register(ctx) -> None:
    # Model-provider discovery registers the profile above; no command-plugin shim is needed.
    return None
