# attnroute

### Intelligent Context Routing for AI Coding Assistants

**Cut your token usage by 98%+** — attnroute learns which files matter and delivers only what's needed.

```
1,569,434 tokens  →  2,027 tokens  (99.87% reduction)
     556 files    →  symbol map    in 309ms
```

---

## Why attnroute?

AI coding assistants waste tokens by injecting entire files when you only need a few functions. attnroute fixes this by:

- **Learning your patterns** — tracks which files you actually use together
- **Predicting what's next** — pre-warms files before you ask for them
- **Compressing intelligently** — symbol-level summaries instead of full content
- **Working standalone** — zero required dependencies, optional enhancements

## Verified Performance

Measured with `tiktoken cl100k_base` on real codebases:

| Repository | Files | Before | After | Reduction | Time |
|:-----------|------:|-------:|------:|----------:|-----:|
| Go backend | 556 | 1,569,434 | 2,027 | **99.87%** | 309ms |
| Python lib | 30 | 94,991 | 2,072 | **97.82%** | 95ms |

> **Don't trust these numbers?** Run `attnroute benchmark` on your own codebase.

---

## Quick Start

```bash
# Install with all features
pip install attnroute[all]

# Initialize for your project
attnroute init

# Check what's available
attnroute status
```

### Installation Options

```bash
pip install attnroute              # Core only (zero dependencies)
pip install attnroute[search]      # + BM25 & semantic search
pip install attnroute[graph]       # + tree-sitter & PageRank
pip install attnroute[compression] # + Claude API memory
pip install attnroute[all]         # Everything
```

---

## How It Works

attnroute maintains a "working memory" of your codebase:

```
┌─────────────────────────────────────────────────────────────┐
│  HOT (score > 0.7)    Full content — active focus           │
│  WARM (0.3 - 0.7)     Symbols only — background awareness   │
│  COLD (< 0.3)         Not injected — out of context         │
└─────────────────────────────────────────────────────────────┘
```

**The pipeline:**

1. **Attention Tracking** — monitors file interactions in real-time
2. **Heat Decay** — recent files stay hot, old ones cool down
3. **Co-activation** — files used together get linked
4. **PageRank** — dependency graph ranks importance
5. **Token Budget** — fits everything within your limits

---

## Features

### Repo Mapping
Extracts function and class signatures using tree-sitter AST parsing. Ranks files by importance using PageRank on the dependency graph.

### Usage Learning
Learns prompt→file associations over time. Boosts files that historically match your query patterns.

### Smart Prediction
Predicts which files you'll need based on:
- Recent file sequences
- Keyword associations
- Co-occurrence patterns
- Project context

### Memory Compression
Optional long-term memory via Claude API. Compresses tool outputs into semantic summaries for retrieval across sessions.

---

## CLI Commands

```bash
attnroute init          # Set up for current project
attnroute status        # Show configuration and features
attnroute report        # Token efficiency metrics
attnroute benchmark     # Run performance benchmarks
attnroute diagnostic    # Generate bug report
attnroute history       # View attention history
attnroute version       # Show version info

# With optional dependencies:
attnroute graph stats   # Dependency graph info
attnroute compress stats # Memory compression stats
```

---

## Configuration

Create `.claude/keywords.json` in your project:

```json
{
  "keywords": {
    "src/api.py": ["api", "endpoint", "route", "handler"],
    "src/auth.py": ["auth", "login", "token", "session"],
    "docs/setup.md": ["install", "setup", "configure"]
  },
  "pinned": ["README.md", "src/config.py"]
}
```

See [`examples/keywords.json`](examples/keywords.json) for a complete example.

---

## Architecture

```
attnroute/
├── context_router.py   # Main routing logic, attention tracking
├── repo_map.py         # Tree-sitter parsing, PageRank ranking
├── learner.py          # Usage pattern learning
├── predictor.py        # File prediction model
├── compressor.py       # Memory compression (optional)
├── indexer.py          # BM25/semantic search (optional)
├── graph_retriever.py  # Dependency graph CLI (optional)
└── cli.py              # Command-line interface
```

---

## Benchmarks

### Methodology

Our benchmarks are designed for transparency and reproducibility:

- **tiktoken cl100k_base** — same tokenizer family as Claude
- **Multiple runs** — statistical analysis with 95% confidence intervals
- **Real content** — actual file tokens, not estimates
- **Full timing** — includes indexing and map generation

### Run Your Own

```bash
# Quick verification on current directory
attnroute benchmark

# Test on a specific repo
python benchmarks/verify_claims.py /path/to/repo

# Full statistical benchmark
python benchmarks/bulletproof_benchmark.py --runs 5

# Compare with Aider (if installed)
python benchmarks/aider_head_to_head.py /path/to/repo
```

---

## Contributing

```bash
git clone https://github.com/jeranaias/attnroute.git
cd attnroute
pip install -e ".[all,dev]"
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Issues & Support

**Found a bug?** Generate a diagnostic report:

```bash
attnroute diagnostic
```

Then [open an issue](https://github.com/jeranaias/attnroute/issues/new) with:
- What happened vs. what you expected
- Steps to reproduce
- The diagnostic report file

---

## Acknowledgments

attnroute builds on ideas from:

- [Aider](https://github.com/paul-gauthier/aider) — pioneered repo mapping with tree-sitter and PageRank
- [Claude Code](https://github.com/anthropics/claude-code) — Anthropic's CLI for Claude
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) — memory patterns for LLMs

---

## License

MIT

---

<p align="center">
  <i>Stop wasting tokens. Start routing attention.</i>
</p>
