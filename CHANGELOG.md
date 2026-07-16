# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.3]

### Fixed
- `LoggingOnlyPolicy` no longer flags every clean request as `WARN`. Its default
  `warn_threshold` was `0.0`, so a clean request scoring `0.0` satisfied
  `score >= warn_threshold` and was always flagged. Default is now `0.99`, so only
  near-critical scores trigger a warning.
- `pyproject.toml` `Homepage`/`Repository`/`Issues` URLs updated from the old
  `llm-security-toolkit` repo name to `github.com/siphalion/quisium`.
- README package-path references updated from `src/llm_security/...` to
  `src/quisium/...` (6 locations) to match the actual repository layout.

### Added
- `.github/workflows/ci.yml` — runs `ruff check` and `pytest` on push/PR across
  Python 3.9, 3.11, and 3.12.
- `tests/conftest.py` — shared `balanced`, `strict`, `logging_only` policy
  fixtures and an autouse handler-cleanup fixture, deduplicated out of six
  individual test files.
- `quisium.__init__` now re-exports the full exception hierarchy
  (`LLMSecurityError`, `PromptBlockedError`, `OutputBlockedError`,
  `PolicyNotFoundError`, `ProviderError`, `ProviderTimeoutError`, `GuardError`)
  alongside the previously exported `BlockedByPolicyError` and
  `InvalidToolCallError`, so callers no longer need to import from
  `quisium.exceptions` directly.

### Changed
- `LLMSecurityMiddleware` renamed to `QuisiumMiddleware` in both
  `quisium.middleware.fastapi` and `quisium.middleware.flask`.
  `LLMSecurityMiddleware` remains available as a deprecated alias.

## [0.1.2] and earlier

See [GitHub releases](https://github.com/siphalion/quisium/releases) for
prior version history.
