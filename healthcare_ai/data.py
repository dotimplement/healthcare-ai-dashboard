"""Loading and derivation for the healthcare AI repository dataset.

This is the single definition of the derived columns. `main.py` (the Streamlit
dashboard) and `charts.py` (the marimo notebook) both import from here, so the
dashboard and the published charts cannot disagree about what "Active" means.

The key design point is `as_of`: every time-relative column is computed against
the most recent commit date *in the data*, not against wall-clock time. The
previous code used `datetime.now()`, which meant a frozen dataset silently
drifted towards "everything is inactive" the longer it went without a refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from healthcare_ai import config, owner_types


@dataclass(frozen=True)
class AsOf:
    """The reference date for all time-relative derivations."""

    date: pd.Timestamp
    source: str  # "data" or "explicit"

    @property
    def year(self) -> int:
        return int(self.date.year)

    def __str__(self) -> str:
        return self.date.strftime("%Y-%m-%d")


def resolve_as_of(df: pd.DataFrame, explicit: pd.Timestamp | None = None) -> AsOf:
    """Determine the as-of date: the caller's, else the data's latest commit."""
    if explicit is not None:
        return AsOf(pd.Timestamp(explicit), "explicit")
    return AsOf(pd.Timestamp(df["Last Commit"].max()), "data")


def _created_year_label(start: int, latest_year: int) -> str:
    """Build the label for an open-ended creation cohort, e.g. '2022-2026'."""
    return f"{start}-{latest_year}"


def add_created_year_buckets(df: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    """Assign each repo a creation cohort from config.CREATED_YEAR_BUCKETS."""
    first_start, first_label = config.CREATED_YEAR_BUCKETS[0]
    df["Created_years"] = first_label

    for start, label in config.CREATED_YEAR_BUCKETS[1:]:
        if label is None:
            label = _created_year_label(start, latest_year)
        df.loc[df["Created"].dt.year >= start, "Created_years"] = label
    return df


def load_repos(as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load healthcare_data.csv and attach all derived columns.

    Args:
        as_of: Override the reference date. Defaults to the most recent
            "Last Commit" in the data.

    Returns:
        The repo dataframe. `df.attrs["as_of"]` holds the resolved `AsOf`.
    """
    df = pd.read_csv(config.HEALTHCARE_DATA_CSV)

    # Drop link collections and documentation -- not software projects.
    df = df[~df["Category"].isin(config.EXCLUDED_CATEGORIES)].copy()

    df["Created"] = pd.to_datetime(df["Created"])
    df["Last Commit"] = pd.to_datetime(df["Last Commit"])

    resolved = resolve_as_of(df, as_of)
    reference = resolved.date

    df["Org"] = df["Repository"].str.split("/").str[0]
    df["contributor_count"] = df["Top Contributors"].fillna("").str.split(",").apply(len)

    # Time-relative derivations, all measured from the as-of date.
    df["days_since_last_commit"] = (reference - df["Last Commit"]).dt.days
    df["is_active"] = df["days_since_last_commit"] < config.ACTIVITY_WINDOW_DAYS
    df["recent_activity_category"] = np.where(df["is_active"], "Active", "Inactive")
    df["active_until"] = df["Last Commit"] + timedelta(days=config.ACTIVITY_WINDOW_DAYS)

    df["first_commit"] = df["Created"]
    df["last_commit"] = df["Last Commit"]
    df["lifespan_days"] = (df["last_commit"] - df["first_commit"]).dt.days
    df["start_year"] = df["first_commit"].dt.year

    # Standards may be comma-separated; keep both the raw and exploded forms.
    df["Standard"] = df["Standard"].fillna("None/Unknown")
    df["has_standard"] = df["Standard"] != "None/Unknown"
    df["standards_list"] = df["Standard"].apply(
        lambda x: [s.strip() for s in str(x).split(",") if s.strip()]
    )

    df = _attach_ownership(df)
    df = add_created_year_buckets(df, latest_year=int(df["Created"].dt.year.max()))

    df.attrs["as_of"] = resolved
    return df


def _attach_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the coarse API classification, then the curated taxonomy on top."""
    try:
        classified = pd.read_csv(config.REPOS_CLASSIFIED_CSV)
    except FileNotFoundError:
        df["is_organization"] = None
        df["owner_type"] = "Unknown"
    else:
        lookup = (
            classified.drop_duplicates(subset=["owner"])
            .set_index("owner")[["is_organization", "owner_type"]]
            .to_dict("index")
        )
        df["is_organization"] = df["Org"].map(
            lambda o: lookup.get(o, {}).get("is_organization", None)
        )
        df["owner_type"] = df["Org"].map(
            lambda o: lookup.get(o, {}).get("owner_type", "Unknown")
        )

    # The curated four-way taxonomy overrides the coarse Organization/User
    # label where an owner has been labelled. Unlabelled owners keep the
    # coarse value and drop out of the ownership charts.
    curated = df["Org"].map(lambda o: owner_types.classify(o))
    df["owner_type"] = curated.fillna(df["owner_type"])
    df["has_curated_owner_type"] = curated.notna()
    return df


def load_contributor_stats() -> pd.DataFrame | None:
    """Load aggregated contributor stats, or None if not yet generated."""
    try:
        return pd.read_csv(config.CONTRIBUTOR_STATS_AGG_CSV)
    except FileNotFoundError:
        return None


def load_star_events() -> pd.DataFrame:
    """Load the star event history produced by `healthcare_ai.stars`."""
    events = pd.read_csv(config.STAR_EVENTS_CSV)
    events["date"] = pd.to_datetime(events["date"])
    return events.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Chart bounds -- derived, never hardcoded
# --------------------------------------------------------------------------


def chart_year_range(df: pd.DataFrame) -> tuple[int, int]:
    """Return the [start, end) year bounds for published charts.

    The upper bound follows the data, so a refresh that adds a new year does
    not silently drop it -- the failure mode of the previous `year < 2026`
    literals scattered through the notebook.
    """
    latest = int(max(df["Created"].dt.year.max(), df["Last Commit"].dt.year.max()))
    return config.CHART_START_YEAR, latest + 1


def year_bounds_from(df: pd.DataFrame, year_column: str = "year") -> tuple[int, int]:
    """Chart bounds [start, end) derived from a frame's own `year` column."""
    return config.CHART_START_YEAR, int(df[year_column].max()) + 1


def filter_to_chart_years(
    df: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    year_column: str = "year",
) -> pd.DataFrame:
    """Restrict a frame with a `year` column to `year_range` = [start, end).

    When `year_range` is omitted the upper bound is taken from the frame
    itself, so adding a new year of data extends the chart rather than being
    silently dropped -- the failure mode of the previous `year < 2026`
    literals. Pass an explicit range to force several charts onto shared axes.
    """
    start, end = (
        year_range if year_range is not None else year_bounds_from(df, year_column)
    )
    return df[(df[year_column] >= start) & (df[year_column] < end)]


def star_history_cutoff(events: pd.DataFrame) -> pd.Timestamp:
    """Last complete month in the star event data.

    Replaces the hardcoded `date < "2025-10-01"` bound. Star events are
    resampled monthly, so the final partial month is dropped to avoid a
    misleading dip at the right edge of the chart.
    """
    latest = pd.Timestamp(events["date"].max())
    return latest.to_period("M").to_timestamp()
