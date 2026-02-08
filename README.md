# attnroute

**Context routing for AI coding assistants**

attnroute reduces token usage by 98-99% through intelligent context selection. It learns which files you actually use and predicts what you'll need next.

## Performance

Measured with tiktoken on real codebases, verified with multiple runs and 95% confidence intervals:

| Repository | Files | Baseline Tokens | Output Tokens | Reduction | Time |
|------------|-------|-----------------|---------------|-----------|------|
| Go project (556 files) | 556 | 1,569,434 | 2,627 | 99.83% | 253ms |
| Python project (36 files) | 36 | 134,449 | 2,842 | 97.89% | 139ms |

### Verify These Claims

Don't trust our numbers? Run the benchmark yourself:

```bash
# Using the CLI
attnroute benchmark

# Or run directly
python benchmarks/verify_claims.py /path/to/your/repo
```

## Features

- **Repo Mapping**: Symbol-level codebase summaries using tree-sitter and PageRank
- **Usage Learning**: Tracks which files you actually read, edit, and reference
- **Smart Prediction**: Predicts which files you'll need based on patterns
- **Memory Compression**: Optional long-term memory via Claude API
- **Zero Dependencies**: Core functionality works standalone

## Installation

```bash
# Core only (no external dependencies)
pip install attnroute

# With semantic search
pip install attnroute[search]

# With AST parsing and PageRank
pip install attnroute[graph]

# With memory compression
pip install attnroute[compression]

# Everything
pip install attnroute[all]
```

## Quick Start

```bash
# Initialize for your project
attnroute init

# Check status
attnroute status

# View efficiency metrics
attnroute report
```

## How It Works

attnroute maintains a "working memory" of your codebase:

1. **Attention Tracking**: Monitors which files you interact with
2. **Heat Decay**: Recent files stay "hot", older ones cool down
3. **Co-activation**: Files used together get linked
4. **Repo Mapping**: Generates symbol-level summaries ranked by PageRank
5. **Token Budgeting**: Fits context within limits, prioritizing by importance

### Context Injection Strategy

```
HOT files (score > 0.7):  Full content for first file, symbols for rest
WARM files (0.3 - 0.7):   Headers and key signatures only
COLD files (< 0.3):       Not injected
```

## Architecture

```
attnroute/
├── context_router.py   # Main routing logic, attention tracking
├── repo_map.py         # Tree-sitter parsing, PageRank ranking
├── compressor.py       # Memory compression (optional)
├── learner.py          # Usage pattern learning
├── predictor.py        # File prediction model
├── indexer.py          # BM25/semantic search (optional)
└── cli.py              # Command-line interface
```

## Configuration

Create `keywords.json` in your project's `.claude/` directory:

```json
{
  "keywords": {
    "docs/api.md": ["api", "endpoint", "route"],
    "docs/setup.md": ["install", "setup", "config"]
  },
  "pinned": ["docs/overview.md"]
}
```

## Benchmarks

### Methodology

Our benchmarks use:
- **tiktoken cl100k_base** for token counting (same family as Claude)
- **Multiple runs** with standard deviation and 95% confidence intervals
- **Real file content**, not estimates
- **Full timing** including indexing and generation

### Running Benchmarks

```bash
# Quick verification
python benchmarks/verify_claims.py

# Full statistical benchmark
python benchmarks/bulletproof_benchmark.py --runs 5

# Head-to-head with Aider (requires aider-chat installed)
python benchmarks/aider_head_to_head.py /path/to/git/repo
```

## Related Projects

attnroute builds on ideas from several excellent projects:

- [Aider](https://github.com/paul-gauthier/aider) - Pioneered repo mapping with tree-sitter and PageRank
- [Claude Code](https://github.com/anthropics/claude-code) - Anthropic's CLI for Claude
- [Continuous Memory](https://github.com/anthropics/anthropic-cookbook) - Memory patterns for LLMs

## Issues and Feedback

### Report a Bug

Generate a diagnostic report to include with your issue:

```bash
# Generate a shareable report
attnroute diagnostic

# Or for a specific repository
attnroute diagnostic /path/to/your/repo
```

This creates `attnroute_diagnostic.txt` with:
- System info (OS, Python version, installed dependencies)
- Repository stats (file count, types)
- Benchmark results (latency, token counts)
- Configuration status

[Open an issue](https://github.com/jeranaias/attnroute/issues/new) with:
- What happened vs what you expected
- Steps to reproduce
- The diagnostic report file

### Performance Not As Expected?

Run the benchmark on your repo and share the results:

```bash
attnroute benchmark
# Or: python benchmarks/verify_claims.py /path/to/your/repo
```

This helps us understand where attnroute can improve.

### Feature Requests

We'd love to hear your ideas! [Open an issue](https://github.com/jeranaias/attnroute/issues/new) describing:
- Your use case
- How you envision it working

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Quick start:
```bash
git clone https://github.com/jeranaias/attnroute.git
cd attnroute
pip install -e ".[all,dev]"
pytest tests/
```

## License

MIT

---

*Built for developers who want efficient AI-assisted coding without wasting tokens.*
