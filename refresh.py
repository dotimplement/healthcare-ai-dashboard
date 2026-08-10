"""Quarterly refresh pipeline.

Regenerates every derived artifact from healthcare_data.csv, in dependency
order. See REFRESH.md for the full quarterly workflow.

    uv run python refresh.py                # everything
    uv run python refresh.py --only stars   # one step
    uv run python refresh.py --skip contributions
    uv run python refresh.py --dry-run      # show what would run

healthcare_data.csv itself is an INPUT, not an output: the category, subcategory
and standard columns are editorial judgements maintained by hand. Update it
before running this.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from healthcare_ai import config

# Ordered pipeline. Each step names the artifacts it produces so --dry-run and
# the summary can report them without running anything.
STEPS: list[tuple[str, str, list]] = []


def step(name: str, description: str, outputs: list):
    def register(fn: Callable[[argparse.Namespace], None]):
        STEPS.append((name, description, outputs))
        _IMPLS[name] = (fn, description, outputs)
        return fn

    return register


_IMPLS: dict[str, tuple[Callable, str, list]] = {}


@step(
    "classify",
    "Classify repository owners as organizations or users",
    [config.REPOS_CLASSIFIED_CSV],
)
def _classify(args):
    from healthcare_ai.github_api import classify_owners

    classify_owners(rate_limit_delay=args.rate_limit_delay)


@step(
    "contributions",
    "Fetch per-contributor commit and line statistics",
    [config.CONTRIBUTOR_STATS_CSV, config.CONTRIBUTOR_STATS_AGG_CSV],
)
def _contributions(args):
    from healthcare_ai.github_api import analyze_contributions

    analyze_contributions(rate_limit_delay=args.rate_limit_delay)


@step(
    "stars",
    "Fetch star history for the top repositories",
    [config.STAR_EVENTS_CSV],
)
def _stars(args):
    from healthcare_ai import stars

    stars.refresh(full=args.full_stars)


@step(
    "taxonomy",
    "Report owners that still need a label in owner_types.yaml",
    [],
)
def _taxonomy(args):
    from healthcare_ai import owner_types

    owner_types._report()


def check_preconditions() -> list[str]:
    """Return a list of problems that would make a refresh fail."""
    problems = []
    if not config.HEALTHCARE_DATA_CSV.exists():
        problems.append(
            f"Seed dataset missing: {config.HEALTHCARE_DATA_CSV}. "
            "This file is maintained by hand and is the input to the pipeline."
        )
    try:
        from healthcare_ai.github_api import get_token

        get_token()
    except RuntimeError as exc:
        problems.append(str(exc))
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=[name for name, _, _ in STEPS],
        help="run only these steps (repeatable)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=[name for name, _, _ in STEPS],
        default=[],
        help="skip these steps (repeatable)",
    )
    parser.add_argument(
        "--full-stars",
        action="store_true",
        help="refetch all star history instead of resuming incrementally",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=config.RATE_LIMIT_DELAY,
        help=f"seconds between API calls (default: {config.RATE_LIMIT_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the steps that would run, then exit",
    )
    args = parser.parse_args(argv)

    selected = [
        (name, desc, outs)
        for name, desc, outs in STEPS
        if (not args.only or name in args.only) and name not in args.skip
    ]

    if args.dry_run:
        print("Would run:\n")
        for i, (name, desc, outs) in enumerate(selected, 1):
            print(f"  {i}. {name:14} {desc}")
            for out in outs:
                print(f"       -> {out.name}")
        return 0

    problems = check_preconditions()
    if problems:
        print("Cannot refresh:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}\n", file=sys.stderr)
        return 1

    print(
        f"Refreshing {len(selected)} step(s). Seed data: "
        f"{config.HEALTHCARE_DATA_CSV.name}\n"
    )

    for i, (name, desc, _) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name}: {desc}")
        started = time.monotonic()
        try:
            _IMPLS[name][0](args)
        except Exception as exc:  # noqa: BLE001 - report and stop
            print(
                f"\n  FAILED after {time.monotonic() - started:.0f}s: {exc}\n",
                file=sys.stderr,
            )
            print(
                "Earlier steps completed; rerun with "
                f"--only {name} once the cause is fixed.",
                file=sys.stderr,
            )
            return 1
        print(f"  done in {time.monotonic() - started:.0f}s\n")

    print("Refresh complete. Next:")
    print("  1. Review owner_types.yaml if the taxonomy report flagged owners")
    print("  2. uv run pytest")
    print("  3. uv run marimo edit charts.py   (regenerates nice_plots/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
