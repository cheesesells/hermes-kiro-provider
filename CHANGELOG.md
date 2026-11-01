# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Google / GitHub sign-in (`hermes auth add kiro`, option 3): imports the social sign-in from Kiro CLI, stores it in Hermes' credential pool with the account's real profile ARN, and refreshes it through Kiro's desktop auth service.

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

[1.0.1]: https://github.com/anpicasso/hermes-kiro-provider/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/anpicasso/hermes-kiro-provider/compare/v0.1.7...v1.0.0
