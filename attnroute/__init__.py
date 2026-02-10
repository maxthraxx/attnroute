"""
attnroute - Context routing for AI coding assistants

Reduces token usage by 90%+ through intelligent context selection.
Learns which files you actually use and predicts what you'll need next.

Features:
- Repo mapping with tree-sitter and PageRank
- Usage pattern learning
- Smart file prediction
- Memory compression (optional)
- Zero required dependencies

Quick start:
    pip install attnroute[all]
    attnroute init
    attnroute status
"""

__version__ = "0.5.3"
__author__ = "jeranaias"

# Core exports
try:
    from attnroute.context_router import build_context_output, get_tier, update_attention
except ImportError:
    pass

try:
    from attnroute.repo_map import RepoMapper
except ImportError:
    pass

try:
    from attnroute.compressor import ObservationCompressor, ProgressiveRetriever
except ImportError:
    pass

try:
    from attnroute.learner import Learner
except ImportError:
    pass
