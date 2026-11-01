# hermes-kiro-provider

Native Kiro model-provider plugin for Hermes. It connects directly to Kiro over HTTPS—no local proxy, daemon, helper binary, or `kiro-cli`.

## Native Hermes integration

- **Authentication:** `hermes auth add|status|refresh|logout kiro`
- **Credential storage:** OAuth tokens and Kiro metadata live in Hermes' profile-scoped `auth.json` credential pool, including multi-account priority and safe cross-process refresh
- **Usage:** the normal `/usage` command renders Kiro allowances and reset times
- **Models:** live Kiro models appear in main and auxiliary model pickers, with curated offline fallbacks
- **Capabilities:** Hermes knows the bundled fallback models support tools and are text-only
- **Transport:** direct Kiro event streams, tool calls/results, and automatic 401 recovery

```mermaid
flowchart LR
    U[User] --> A[hermes auth ... kiro]
    A --> S[$HERMES_HOME/auth.json]
    H[Hermes CLI / Gateway] --> P[kiro-provider]
    P --> S
    P --> T[Shared HTTPS pool]
    T --> K[Kiro APIs\nstream · models · usage]
```

## ⚠️ v1.0.0 requires signing in again

v1.0.0 removes compatibility with credentials created by earlier releases. After upgrading, replace each profile's existing Kiro login:

```bash
hermes auth logout kiro
hermes auth add kiro
```

For a named profile:

```bash
hermes --profile work auth logout kiro
hermes --profile work auth add kiro
```

The plugin no longer reads the old `$HERMES_HOME/kiro/credentials.json` file, the `KIRO_AUTH` sentinel, or repairs pre-v1.0 credential-pool rows.

## Install

Use Hermes' built-in plugin installer (Hermes Agent 0.21.3 or newer):

```bash
hermes plugins install kiro-provider
```

Plugins are profile-scoped. Install once per profile that should use Kiro:

```bash
hermes --profile work plugins install kiro-provider
```

Restart an already-running multiplexed gateway once after installation so it imports the provider:

```bash
hermes gateway restart
```

For a non-multiplexed gateway serving one named profile:

```bash
hermes --profile work gateway restart
```

## Authenticate

```bash
hermes auth add kiro
```

Choose AWS Builder ID, corporate IAM Identity Center or Google / GitHub interactively. Provider-specific auth values are requested by the plugin because `hermes auth` intentionally does not add plugin-defined flags.

**Google / GitHub.** This option does not run its own browser flow; it reuses the sign-in of [Kiro CLI](https://kiro.dev/docs/cli/). Run `kiro-cli login`, choose Google or GitHub, then run `hermes auth add kiro` and pick option 3. The plugin reads `kirocli:social:token` from kiro-cli's local database (read-only), proves it with one refresh, and keeps its own copy of the refresh token in Hermes' credential pool. It then refreshes through `prod.<region>.auth.desktop.kiro.dev/refreshToken`. Kiro does not rotate this refresh token, so the plugin and kiro-cli can each refresh independently; if kiro-cli is signed out or you log in again, repeat the two commands.

Useful native commands:

```bash
hermes auth status kiro
hermes auth list kiro
hermes auth refresh kiro
hermes auth logout kiro
```

For a named profile, use Hermes' global profile option:

```bash
hermes --profile work auth add kiro
```

Then select Kiro for the main model or any auxiliary task through the normal Hermes model pickers. When Kiro is active, `/usage` uses the provider's native account-usage hook.

## Development

```bash
HERMES_HOME="$(mktemp -d)" \
PYTHONPATH="$HOME/.hermes/hermes-agent:provider" \
  "$HOME/.hermes/hermes-agent/venv/bin/python3" -m pytest tests -q

hermes plugins doctor ./provider --ci
```

Kiro is an undocumented private API. Use your own entitlement; never commit credentials.
