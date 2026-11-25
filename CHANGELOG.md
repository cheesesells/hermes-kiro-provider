# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- `hermes auth add kiro` menu entry 5 imports the Google / GitHub sign-in that `kiro-cli login` stored (read-only read of `kirocli:social:token`, one proving refresh, then saved as the same pool row shape as the device login).
- `/usage` and the account-usage snapshot list active bonus credits (name, used/limit, expiry) under each bucket.

### Changed

- Pool rows written by the earlier kiro-cli import (`auth_method: social`) are still read.

## [1.2.1] - 2026-11-23

### Fixed

- Discover IAM Identity Center Kiro profiles across the supported commercial regions when the preferred region returns no profile.
- Route management, model discovery, usage and inference using the discovered profile ARN's allowlisted region, while keeping the original IdC region for login and token refresh.
- Preserve HTTP/auth failures and existing Builder ID and Google/GitHub routing; empty-profile errors name the regions searched.

No credential migration, new dependencies or configuration changes are required.


## [1.2.0] - 2026-11-17

### Added

- `hermes auth add kiro` offers Google and GitHub sign-in (menu entries 3 and 4) through Kiro's own device flow at `prod.us-east-1.auth.desktop.kiro.dev`; the plugin polls for authorization, stores the returned profile ARN, and refreshes the access token through `/refreshToken`. A rejected refresh (400/401/403) asks for `hermes auth add kiro` again. A `slow_down` poll reply makes the poll wait 2 seconds longer. Rows are labelled `kiro-google` / `kiro-github` unless `--label` is given.

### Changed

- The status lines (`hermes kiro status` and the standalone `credentials.py status`) print `region=...` where they printed `IdC region=...`.
- The `hermes kiro login` help text reads "Sign in to Kiro"; it does not name IAM Identity Center alone.
- The `--start-url` help text also accepts `social:google` / `social:github`, and the `--region` help text says Google and GitHub ignore it.
- The provider description names all four sign-in methods: AWS Builder ID, IAM Identity Center, Google and GitHub.

## [1.1.0] - 2026-11-13

### Added

- Base64 `image_url` / `input_image` parts are forwarded as Kiro image blocks on current, history and tool-result turns; Claude and GPT catalog models declare vision, while DeepSeek, MiniMax, GLM and Qwen stay text-only.

### Fixed

- Builder ID streaming requests now send the fixed Builder ID profile ARN that `runtime.*.kiro.dev` requires; the gateway otherwise rejects them with `400 profileArn is required for this request`.
- Tool schemas are stripped of `additionalProperties` and empty `required` arrays, which Kiro rejects with `400 REQUEST_BODY_INVALID`.
- A `CONTENT_FILTERED` stream stop is surfaced as an error instead of a normal completion.
- Provider errors expose the upstream HTTP status as `status_code` for Hermes' failover classifiers.
- The compatibility logout helper clears only the Kiro pool, removing a private `hermes_cli` import.
- Control-plane REST calls use the account region (`q.<region>.amazonaws.com` outside `us-east-1`) instead of a hardcoded `us-east-1` host, and model-discovery fallback is logged.
- Kiro's schema-invalid `additionalModelRequestFields` 400 is retried without the fields and the model is marked unsupported, instead of failing the request.
- Non-stream calls made by a synchronous caller on a thread with a running event loop return a real response instead of an unawaited coroutine.
- Hermes' generic reasoning config is mapped onto Kiro's per-family `additionalModelRequestFields` schema; models that only allow adaptive thinking are retried with adaptive thinking at low effort.

## [1.0.1] - 2026-09-25

### Fixed

- Deferred `botocore` and `urllib3` imports so Hermes can discover the provider while switching dependency environments during a source update.

## [1.0.0] - 2026-09-20

### Breaking

- Credentials created by pre-1.0 releases are no longer migrated or repaired. Every Hermes profile using Kiro must sign in again with `hermes auth add kiro`.
- Removed support for the legacy `$HERMES_HOME/kiro/credentials.json` store, `KIRO_AUTH` sentinel, and pre-v1.0 credential-pool row backfill.

### Added

- Native `hermes auth add|status|refresh|logout kiro` integration through Hermes provider hooks.
- Profile-scoped `auth.json` credential-pool storage with opaque Kiro metadata, multi-account priority, and cross-process refresh serialization.
- Native `/usage` account allowance and reset reporting.
- Main and auxiliary model-picker integration with live discovery, offline fallbacks, and per-model capability metadata.
- Terminal refresh-error classification that marks unusable grants for reauthentication while leaving transient failures retryable.

### Changed

- Store a canonical runtime base URL on new pool entries so Hermes can resolve Kiro sessions and preserve mixed-region account failover.
- Use the official catalog command, `hermes plugins install kiro-provider`.
- Document native multiplexed and per-profile gateway restart commands.

### Removed

- Custom `install.sh` installation flow.
- Plugin-owned command-registration and credential-storage workarounds replaced by Hermes' native provider APIs.

[1.1.0]: https://github.com/anpicasso/hermes-kiro-provider/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/anpicasso/hermes-kiro-provider/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/anpicasso/hermes-kiro-provider/compare/v0.1.7...v1.0.0
