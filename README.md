# attnroute

### Intelligent Context Routing for Claude Code

**Cut your token usage by 98%+** — attnroute learns which files matter and delivers only what's needed.

```
1,569,434 tokens  →  2,027 tokens  (99.87% reduction)
     556 files    →  symbol map    in 309ms
```

---

## What is This?

attnroute is a **hook system for [Claude Code](https://github.com/anthropics/claude-code)** (Anthropic's CLI). It automatically injects smart context into every prompt, so Claude sees the right files without you copying/pasting anything.

**Before attnroute:** Claude reads your entire codebase (millions of tokens, slow, expensive)

**After attnroute:** Claude sees only relevant symbols and files (thousands of tokens, fast, cheap)

---

## 5-Minute Setup

### 1. Install

```bash
pip install attnroute[all]
```

### 2. Initialize

```bash
cd /path/to/your/project
attnroute init
```

This automatically configures Claude Code's hooks. That's it.

### 3. Verify It's Working

```bash
attnroute status
```

You should see:
```
attnroute Status
==================================================
Features: BM25 search, Semantic search, Graph retrieval...
Keywords: .claude/keywords.json
Telemetry: X turns recorded
```

### 4. Use Claude Code Normally

Just use Claude Code as you normally would. attnroute works invisibly in the background:

```bash
claude   # Start Claude Code - attnroute hooks activate automatically
```

Every prompt you send now includes intelligently-selected context. You'll notice:
- Faster responses (less tokens to process)
- Better answers (relevant files, not random ones)
- Lower costs (if using API directly)

---

## How Do I Know It's Working?

Run this after a coding session:

```bash
attnroute report
```

You'll see token savings, which files were injected, and efficiency metrics.

Or check the history:

```bash
attnroute history
```

---

## Installation Options

```bash
pip install attnroute              # Core only (zero dependencies)
pip install attnroute[search]      # + BM25 & semantic search
pip install attnroute[graph]       # + tree-sitter & PageRank
pip install attnroute[compression] # + Claude API memory
pip install attnroute[all]         # Everything (recommended)
```

---

## Verified Performance

Measured with `tiktoken cl100k_base` on real codebases:

| Repository | Files | Before | After | Reduction | Time |
|:-----------|------:|-------:|------:|----------:|-----:|
| Go backend | 556 | 1,569,434 | 2,027 | **99.87%** | 309ms |
| Python lib | 30 | 94,991 | 2,072 | **97.82%** | 95ms |

> **Skeptical?** Run `attnroute benchmark` on your own codebase.

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

1. **Attention Tracking** — monitors which files you interact with
2. **Heat Decay** — recent files stay hot, old ones cool down
3. **Co-activation** — files used together get linked
4. **PageRank** — dependency graph ranks importance
5. **Token Budget** — fits everything within limits

---

## CLI Reference

```bash
attnroute init          # Set up hooks for current project
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

## Configuration (Optional)

For better results, create `.claude/keywords.json` in your project:

```json
{
  "keywords": {
    "src/api.py": ["api", "endpoint", "route", "handler"],
    "src/auth.py": ["auth", "login", "token", "session"]
  },
  "pinned": ["README.md", "src/config.py"]
}
```

**pinned** files are always included. **keywords** help match prompts to files.

See [`examples/keywords.json`](examples/keywords.json) for a complete example.

---

## Troubleshooting

### "attnroute not found" after install
```bash
# Make sure it's in your PATH
pip show attnroute   # Check installation location
```

### Hooks not activating
```bash
# Re-run init
attnroute init

# Check Claude Code settings
cat ~/.claude/settings.json | grep attnroute
```

### Not seeing token savings
```bash
# Check if telemetry is recording
attnroute status

# View recent activity
attnroute history --last 10
```

---

## Requirements

- **Python 3.10+**
- **Claude Code** (Anthropic's CLI)
- Optional: tree-sitter, networkx, tiktoken for full features

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

Built on ideas from:

- [Aider](https://github.com/paul-gauthier/aider) — repo mapping with tree-sitter and PageRank
- [Claude Code](https://github.com/anthropics/claude-code) — Anthropic's CLI

---

## License

MIT

---

<p align="center">
  <i>Stop wasting tokens. Start routing attention.</i>
</p>
