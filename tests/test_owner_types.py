"""Tests for the owner taxonomy.

The coverage test is the one that does real work each quarter: it fails when
enough newly-added repos have unlabelled owners that the ownership charts stop
being representative.
"""

import pytest

from healthcare_ai import config, owner_types
from healthcare_ai.data import load_repos


@pytest.fixture(scope="module")
def df():
    return load_repos()


def test_taxonomy_loads_and_types_are_valid():
    taxonomy = owner_types.load_taxonomy()
    assert taxonomy
    assert set(taxonomy.values()) <= set(owner_types.VALID_TYPES)


def test_no_owner_listed_under_two_types():
    """load_taxonomy raises on conflicts; this asserts the file is clean."""
    owner_types.load_taxonomy.cache_clear()
    owner_types.load_taxonomy()


def test_classify_is_case_insensitive_and_exact():
    assert owner_types.classify("microsoft") == "Incumbent"
    assert owner_types.classify("MICROSOFT") == "Incumbent"
    assert owner_types.classify("not-a-real-owner") is None
    assert owner_types.classify("not-a-real-owner", default="Unknown") == "Unknown"


def test_matching_is_not_substring_based():
    """Regression guard for the original substring rules.

    `awslabs/fhir-works-on-aws-authz-smart` used to be classified as a
    community project because the repo name contained "smart", and
    `nhsconnect/careconnect-reference-implementation` as a startup because it
    contained "implement".
    """
    assert owner_types.classify("awslabs") == "Incumbent"
    assert owner_types.classify("smart-on-fhir") == "Community Project/Non-Profit"
    # A repo name fragment must never resolve on its own.
    assert owner_types.classify("careconnect-reference-implementation") is None
    assert owner_types.classify("apple-health-mcp-server") is None


def test_owner_coverage_above_threshold(df):
    """Fails when too many repos have an unlabelled owner.

    When this trips after a refresh, run:
        uv run python -m healthcare_ai.owner_types --report
    and add the listed owners to healthcare_ai/owner_types.yaml.
    """
    missing = owner_types.unclassified_owners(df)
    unclassified_repos = sum(n for _, n in missing)
    share = unclassified_repos / len(df)

    assert share <= config.MAX_UNCLASSIFIED_OWNER_SHARE, (
        f"{share:.1%} of repos have an unlabelled owner "
        f"(limit {config.MAX_UNCLASSIFIED_OWNER_SHARE:.0%}). "
        f"Top unlabelled: {[o for o, _ in missing[:10]]}"
    )


def test_curated_types_override_the_coarse_api_label(df):
    """Curated owners must not be left as bare Organization/User."""
    curated = df[df["has_curated_owner_type"]]
    assert not curated.empty
    assert not curated["owner_type"].isin(["Organization", "User", "Unknown"]).all()


def test_chart_owner_types_are_all_reachable(df):
    """Every type the ownership charts break out must actually appear."""
    present = set(df.loc[df["has_curated_owner_type"], "owner_type"])
    missing = set(config.CHART_OWNER_TYPES) - present
    assert not missing, f"no repos resolve to {sorted(missing)}"
