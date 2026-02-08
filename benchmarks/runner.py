#!/usr/bin/env python3
"""
attnroute benchmark runner - CLI entry point for benchmarks.

Usage:
    attnroute benchmark                    # Quick benchmark on current directory
    attnroute benchmark --scenario all     # Full benchmark suite
    attnroute benchmark --output results.json  # Save results to file
"""

import argparse
import sys
from pathlib import Path


def main(scenario: str = None):
    """Main benchmark entry point."""
    # Import verify_claims from same directory
    benchmark_dir = Path(__file__).parent
    sys.path.insert(0, str(benchmark_dir.parent))  # Add attnroute to path
    sys.path.insert(0, str(benchmark_dir))  # Add benchmarks to path

    if scenario == "all":
        # Run full bulletproof benchmark
        try:
            from bulletproof_benchmark import run_bulletproof_benchmark
            run_bulletproof_benchmark(use_public_repos=False, include_aider=True)
        except ImportError as e:
            print(f"Could not import bulletproof_benchmark: {e}")
            print("Falling back to verify_claims...")
            from verify_claims import verify_claims
            verify_claims()
    elif scenario == "single_file":
        # Quick single-file test
        from verify_claims import verify_claims
        verify_claims()
    elif scenario == "multi_file":
        # Multi-file test
        from verify_claims import verify_claims
        verify_claims()
    else:
        # Default: quick verification
        from verify_claims import verify_claims
        verify_claims()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attnroute benchmarks")
    parser.add_argument("--scenario", choices=["all", "quick", "single_file", "multi_file"],
                        default="quick", help="Benchmark scenario to run")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()
    main(scenario=args.scenario)
