"""Contract tests for the dataset and its derived columns.

These are the tests that matter for a quarterly refresh: they fail when the
seed data changes shape, when a derivation silently breaks, or when the
pipeline's assumptions about category names stop holding.
"""

import pandas as pd
import pytest

from healthcare_ai import config
from healthcare_ai.data import (
    chart_year_range,
    filter_to_chart_years,
    load_repos,
    star_history_cutoff,
    year_bounds_from,
)

REQUIRED_SEED_COLUMNS = {
    "Repository",
    "Category",
    "Subcat",
    "Standard",
    "Stars",
    "Forks",
    "URL",
    "Language",
    "Created",
    "Last Commit",
    "Top Contributors",
}

REQUIRED_DERIVED_COLUMNS = {
    "Org",
    "days_since_last_commit",
    "is_active",
    "recent_activity_category",
    "active_until",
    "lifespan_days",
    "start_year",
    "has_standard",
    "standards_list",
    "owner_type",
    "has_curated_owner_type",
    "Created_years",
}


@pytest.fixture(scope="module")
def df():
    return load_repos()


def test_seed_csv_has_required_columns():
    raw = pd.read_csv(config.HEALTHCARE_DATA_CSV, nrows=1)
    missing = REQUIRED_SEED_COLUMNS - set(raw.columns)
    assert not missing, f"healthcare_data.csv is missing columns: {sorted(missing)}"


def test_derived_columns_present(df):
    missing = REQUIRED_DERIVED_COLUMNS - set(df.columns)
    assert not missing, f"load_repos did not produce: {sorted(missing)}"


def test_no_repos_lost_to_excluded_categories(df):
    assert len(df) > 0
    assert not df["Category"].isin(config.EXCLUDED_CATEGORIES).any()


def test_key_fields_never_null(df):
    for column in ("Repository", "Created", "Last Commit", "Stars", "Org"):
        assert df[column].notna().all(), f"{column} contains nulls"


def test_repository_names_are_owner_slash_repo(df):
    bad = df.loc[~df["Repository"].str.contains("/", regex=False), "Repository"]
    assert bad.empty, f"malformed repository names: {list(bad)[:5]}"


def test_dates_are_ordered(df):
    """Creation must precede the last commit."""
    inverted = df[df["Last Commit"] < df["Created"]]
    assert inverted.empty, (
        f"{len(inverted)} repos have a last commit before creation: "
        f"{list(inverted['Repository'])[:5]}"
    )


def test_as_of_comes_from_the_data(df):
    """Activity must be measured against the data, not wall-clock time.

    This is the regression guard for the original bug: using datetime.now()
    meant a frozen dataset drifted towards "everything is inactive".
    """
    as_of = df.attrs["as_of"]
    assert as_of.source == "data"
    assert as_of.date == df["Last Commit"].max()


def test_at_least_one_repo_is_active(df):
    """A dataset where nothing is active means the as-of logic has regressed."""
    assert df["is_active"].sum() > 0
    assert df["is_active"].sum() < len(df), "everything active is equally suspicious"


def test_activity_category_matches_is_active(df):
    expected = df["is_active"].map({True: "Active", False: "Inactive"})
    assert (df["recent_activity_category"] == expected).all()


def test_chart_year_range_tracks_the_data(df):
    """The upper bound must follow the data, never a hardcoded literal."""
    start, end = chart_year_range(df)
    assert start == config.CHART_START_YEAR
    latest = max(df["Created"].dt.year.max(), df["Last Commit"].dt.year.max())
    assert end == latest + 1, (
        "chart upper bound has drifted from the data -- a refresh would "
        "silently drop the newest year"
    )


def test_filter_to_chart_years_keeps_the_latest_year(df):
    """The exact failure the old `year < 2026` literals caused."""
    yearly = df.groupby(df["Created"].dt.year).size().reset_index()
    yearly.columns = ["year", "count"]
    kept = filter_to_chart_years(yearly)
    latest = int(yearly["year"].max())
    assert latest in set(kept["year"]), f"latest year {latest} was filtered out"


def test_year_bounds_from_is_inclusive_of_max(df):
    frame = pd.DataFrame({"year": [2020, 2021, 2030]})
    start, end = year_bounds_from(frame)
    assert start == config.CHART_START_YEAR
    assert end == 2031


def test_created_year_bucket_label_is_generated(df):
    """The open-ended cohort label must track the data, not say '2022-2025'."""
    latest_year = int(df["Created"].dt.year.max())
    labels = set(df["Created_years"])
    open_bucket_start = config.CREATED_YEAR_BUCKETS[-1][0]
    assert f"{open_bucket_start}-{latest_year}" in labels, (
        f"expected an open-ended bucket ending at {latest_year}, got {sorted(labels)}"
    )


def test_star_history_categories_exist_in_the_data(df):
    """Guards the 'HAI Engineering' vs 'HAI engineering' casing bug.

    The original notebook filtered on 'HAI Engineering', which matched nothing,
    silently excluding that whole category from star history.
    """
    present = set(pd.read_csv(config.HEALTHCARE_DATA_CSV)["Category"].unique())
    missing = set(config.STAR_HISTORY_CATEGORIES) - present
    assert not missing, (
        f"config.STAR_HISTORY_CATEGORIES names categories absent from the data: "
        f"{sorted(missing)}. Present: {sorted(present)}"
    )


@pytest.mark.skipif(
    not config.STAR_EVENTS_CSV.exists(), reason="star events not generated yet"
)
def test_star_history_cutoff_is_a_month_boundary():
    from healthcare_ai.data import load_star_events

    events = load_star_events()
    cutoff = star_history_cutoff(events)
    assert cutoff.day == 1
    assert cutoff <= events["date"].max()
