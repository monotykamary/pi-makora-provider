# Changelog

## 1.12.26

- Preserve historical pnpm release-age exemptions in exact-version unions with Pi 1.0.0; verify the policy using pnpm 11.6.0 frozen-lockfile checks.
- Pin Pi SDK development dependencies to 1.0.0 while retaining wildcard host peers.
- Verify real manifest loading, provider catalogs, startup/shutdown and native/bundled Pi hosts offline.
- Exercise real transport adapters with Unicode text, tool calls, empty responses, usage, request hooks and cancellation; no live provider calls.

## 1.12.25

- Validate against Pi 0.99.0, including an offline real-host package-loading probe.
- Declare imported host packages as wildcard peers and pin development dependencies to Pi 0.99.0.
- Typecheck the provider and recovery guard; normalize clamped off-mode reasoning and malformed settings objects.
