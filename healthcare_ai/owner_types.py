"""Owner taxonomy: maps a GitHub owner to an ownership category.

Replaces the previous approach of chained `str.contains` calls over the full
"owner/repo" string, which had two defects: later rules silently overwrote
earlier ones, and patterns matched repo names as well as owners.

Run `python -m healthcare_ai.owner_types --report` after a refresh to see
which owners still need a label.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import yaml

from healthcare_ai import config

VALID_TYPES = (
    "Incumbent",
    "Research Lab",
    "Startup",
    "Community Project/Non-Profit",
    "User",
)


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, str]:
    """Return a {lowercased owner: type} mapping loaded from the YAML file."""
    with open(config.OWNER_TYPES_YAML) as fh:
        raw = yaml.safe_load(fh) or {}

    mapping: dict[str, str] = {}
    for owner_type, owners in raw.items():
        if owner_type not in VALID_TYPES:
            raise ValueError(
                f"{config.OWNER_TYPES_YAML.name}: unknown type {owner_type!r}. "
                f"Valid types: {', '.join(VALID_TYPES)}"
            )
        for owner in owners or []:
            key = owner.lower()
            if key in mapping and mapping[key] != owner_type:
                raise ValueError(
                    f"{config.OWNER_TYPES_YAML.name}: owner {owner!r} listed "
                    f"under both {mapping[key]!r} and {owner_type!r}"
                )
            mapping[key] = owner_type
    return mapping


def classify(owner: str, default: str | None = None) -> str | None:
    """Return the ownership type for `owner`, or `default` if unlisted."""
    if not isinstance(owner, str):
        return default
    return load_taxonomy().get(owner.lower(), default)


def unclassified_owners(df) -> list[tuple[str, int]]:
    """Return [(owner, repo_count)] for owners with no entry, most repos first."""
    taxonomy = load_taxonomy()
    counts = df["Org"].value_counts()
    missing = [(o, int(n)) for o, n in counts.items() if o.lower() not in taxonomy]
    return sorted(missing, key=lambda pair: -pair[1])


def _report() -> None:
    from healthcare_ai.data import load_repos

    df = load_repos()
    missing = unclassified_owners(df)
    covered = len(df) - sum(n for _, n in missing)

    print(f"Owners in taxonomy : {len(load_taxonomy())}")
    print(f"Owners unlabelled  : {len(missing)}")
    print(f"Repos covered      : {covered}/{len(df)} ({covered / len(df):.1%})")
    print(
        f"Threshold          : fails above "
        f"{config.MAX_UNCLASSIFIED_OWNER_SHARE:.0%} unclassified\n"
    )

    if missing:
        print("Unlabelled owners, most repos first:")
        for owner, n in missing[:60]:
            print(f"  {n:3d}  {owner}")
        if len(missing) > 60:
            print(f"  ... and {len(missing) - 60} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="list owners that have no entry in owner_types.yaml",
    )
    args = parser.parse_args()
    if args.report:
        _report()
    else:
        parser.print_help()
