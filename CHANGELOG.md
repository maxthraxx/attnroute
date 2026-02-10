# Changelog

All notable changes to attnroute will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-02-10

### Added
- CHANGELOG.md for version history tracking

### Fixed
- Version sync between `__init__.py` and `pyproject.toml`

## [0.5.1] - 2026-02-10

### Changed
- Toned down README stats to more conservative "90%+" claims
- Fixed CI linting configuration

### Fixed
- Ruff linting errors (import sorting, deprecated type hints)

## [0.5.0] - 2026-02-09

### Added
- **Plugin System** - Extensible architecture for behavioral guardrails
  - `VerifyFirst` - Enforces read-before-write policy (addresses GitHub #23833)
  - `LoopBreaker` - Detects and breaks repetitive failure loops (addresses GitHub #21431)
  - `BurnRate` - Predicts rate limit exhaustion with early warnings (addresses GitHub #22435)
- Plugin CLI: `attnroute plugins list|enable|disable|status`
- Plugin state persistence in `~/.claude/plugins/`
- Entry points for external plugin discovery

### Changed
- Updated pyproject.toml to include plugin subpackages
- Added plugin hooks to context_router, session_init, and telemetry_record

## [0.4.0] - 2026-02-08

### Added
- Graph-based retrieval with PageRank ranking
- Tree-sitter AST parsing for 14+ languages
- Dependency graph caching
- `attnroute graph stats` command

## [0.3.0] - 2026-02-07

### Added
- Memory compression with Claude API (optional)
- 3-layer progressive retrieval (index → timeline → full)
- ChromaDB integration for semantic search
- `attnroute compress stats` command

## [0.2.0] - 2026-02-06

### Added
- BM25 keyword search (bm25s)
- Semantic search with model2vec embeddings
- Graceful degradation when optional deps missing
- `attnroute benchmark` command

## [0.1.0] - 2026-02-05

### Added
- Initial release
- HOT/WARM/COLD context tiering
- Exponential heat decay with co-activation boosting
- Attention state persistence
- Claude Code hook integration (UserPromptSubmit, SessionStart, Stop)
- `attnroute init` and `attnroute status` commands
- Zero required dependencies

[0.5.2]: https://github.com/jeranaias/attnroute/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jeranaias/attnroute/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jeranaias/attnroute/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jeranaias/attnroute/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeranaias/attnroute/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jeranaias/attnroute/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeranaias/attnroute/releases/tag/v0.1.0
