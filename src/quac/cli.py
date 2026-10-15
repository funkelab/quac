"""Command line entry point for the three stages of QuAC."""

import argparse
import sys
import sqlite3

from .config import load_config


def main(argv=None) -> int:
    """Run one configured stage and print its resulting artifact path."""
    parser = argparse.ArgumentParser(
        prog="quac", description="Quantitative Attribution with Counterfactuals"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "explain", "summarize"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config",
            "-c",
            default="config.yaml",
            help="YAML configuration (default: config.yaml)",
        )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "train":
            from .pipeline import train

            result = train(config)
        elif args.command == "explain":
            from .pipeline import explain

            result = explain(config)
        else:
            try:
                from .plotting import write_summary
            except ImportError as error:
                raise ValueError(
                    'Interactive summaries require the plot extra: pip install "quac[plot]"'
                ) from error
            result = write_summary(
                config.output.report, config.output.summary, config.summary.top_k
            )
        print(result)
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"quac: {error}", file=sys.stderr)
        return 1
