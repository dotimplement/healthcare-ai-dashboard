"""Star history fetching for the top healthcare AI repositories.

Extracted from the marimo notebook, where the fetch functions were defined but
never called and the `to_csv` was commented out -- meaning star_events_history.csv
could not actually be regenerated.

Produces a long-format event table:

    date, change, repo, owner_type, event_type

with `change` = +1 when a star is gained, and -1 at the point the repo is
considered inactive (last commit + ACTIVITY_WINDOW_DAYS), so that a cumulative
sum over the events gives "stars held by currently-active repos" over time.

Incremental by default: the stargazers API returns oldest-first, so for a repo
we already have N stars for, we resume from page N//100 + 1 rather than
refetching from scratch. Star *removals* are not visible through this API, so
counts can drift slightly over many quarters -- use `--full` to refetch.

Run via `python -m healthcare_ai.stars`, or through refresh.py.
"""

from __future__ import annotations

import argparse
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm

from healthcare_ai import config
from healthcare_ai.github_api import get_token

STAR_ACCEPT_HEADER = "application/vnd.github.v3.star+json"


def fetch_stargazers(
    owner: str, repo: str, token: str, start_page: int = 1
) -> list[dict]:
    """Fetch stargazer records for one repo, from `start_page` onwards.

    Returns a list of dicts each containing a 'starred_at' timestamp.
    """
    headers = {
        "Accept": STAR_ACCEPT_HEADER,
        "Authorization": f"token {token}",
    }
    stars: list[dict] = []
    page = start_page

    while True:
        response = requests.get(
            f"{config.GITHUB_API_BASE}/repos/{owner}/{repo}/stargazers",
            headers=headers,
            params={"per_page": config.PER_PAGE, "page": page},
            timeout=30,
        )

        if response.status_code == 403:
            print(
                f"  rate limited on {owner}/{repo}, waiting "
                f"{config.RATE_LIMIT_BACKOFF}s..."
            )
            sleep(config.RATE_LIMIT_BACKOFF)
            continue
        if response.status_code != 200:
            print(f"  error fetching {owner}/{repo}: HTTP {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        stars.extend(data)
        page += 1
        sleep(config.RATE_LIMIT_DELAY)

    return stars


def select_repos(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the repos to fetch star history for: top N by stars, in scope."""
    selected = df[df["Category"].isin(config.STAR_HISTORY_CATEGORIES)]
    selected = selected[selected["owner_type"] != "Unknown"]
    return selected.sort_values("Stars", ascending=False).head(config.STAR_HISTORY_TOP_N)


def _existing_star_counts(existing: pd.DataFrame | None) -> dict[str, int]:
    """How many 'star_gained' events we already hold, per repo."""
    if existing is None or existing.empty:
        return {}
    gained = existing[existing["event_type"] == "star_gained"]
    return gained["repo"].value_counts().to_dict()


def build_events(
    df: pd.DataFrame,
    token: str,
    existing: pd.DataFrame | None = None,
    full: bool = False,
) -> pd.DataFrame:
    """Fetch star histories and build the event table.

    Args:
        df: The processed repo dataframe (needs Repository, Created,
            active_until, owner_type, Stars, Category).
        token: GitHub personal access token.
        existing: Previously saved events, used to resume fetching.
        full: Ignore `existing` and refetch every repo from page 1.
    """
    selected = select_repos(df)
    have = {} if full else _existing_star_counts(existing)

    gained: list[dict] = []
    print(
        f"Fetching star history for {len(selected)} repos "
        f"({'full refetch' if full else 'incremental'})..."
    )

    for _, row in tqdm(selected.iterrows(), total=len(selected)):
        full_name = row["Repository"]
        if "/" not in full_name:
            print(f"  skipping malformed repo name: {full_name}")
            continue
        owner, repo = full_name.split("/", 1)

        known = have.get(full_name, 0)
        # Resume at the page containing the last star we already have, so a
        # partially-filled page is completed rather than skipped.
        start_page = (known // config.PER_PAGE) + 1

        for star in fetch_stargazers(owner, repo, token, start_page=start_page):
            starred_at = pd.to_datetime(star["starred_at"]).tz_localize(None)
            if starred_at < row["Created"].tz_localize(None):
                continue  # defensive: star predating repo creation
            gained.append(
                {
                    "date": starred_at,
                    "change": 1,
                    "repo": full_name,
                    "owner_type": row["owner_type"],
                    "event_type": "star_gained",
                }
            )

    new_gained = pd.DataFrame(gained)

    # Carry forward previously-fetched gains, then drop duplicates from the
    # overlapping resume page.
    if existing is not None and not existing.empty and not full:
        prior = existing[existing["event_type"] == "star_gained"].copy()
        prior["date"] = pd.to_datetime(prior["date"])
        new_gained = pd.concat([prior, new_gained], ignore_index=True)

    if new_gained.empty:
        return new_gained

    new_gained = new_gained.drop_duplicates(subset=["repo", "date"], keep="first")

    # Removal events are always recomputed: `active_until` moves every refresh
    # as repos receive new commits, so stale -1 events would be wrong.
    removals = _build_removal_events(new_gained, selected)

    events = pd.concat([new_gained, removals], ignore_index=True)
    return events.sort_values("date").reset_index(drop=True)


def _build_removal_events(gained: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Emit a -1 event at `active_until` for each star gained before then."""
    meta = selected.set_index("Repository")[["active_until", "owner_type"]]
    rows: list[dict] = []

    for full_name, group in gained.groupby("repo"):
        if full_name not in meta.index:
            continue
        active_until = pd.Timestamp(meta.loc[full_name, "active_until"]).tz_localize(None)
        n_before = int((group["date"] < active_until).sum())
        rows.extend(
            {
                "date": active_until,
                "change": -1,
                "repo": full_name,
                "owner_type": meta.loc[full_name, "owner_type"],
                "event_type": "star_removed_inactive",
            }
            for _ in range(n_before)
        )
    return pd.DataFrame(rows)


def refresh(full: bool = False, token: str = None) -> pd.DataFrame:
    """Fetch star history and write config.STAR_EVENTS_CSV."""
    from healthcare_ai.data import load_repos

    df = load_repos()
    existing = None
    if config.STAR_EVENTS_CSV.exists() and not full:
        existing = pd.read_csv(config.STAR_EVENTS_CSV)

    events = build_events(df, token or get_token(), existing=existing, full=full)
    if events.empty:
        print("No star events fetched; leaving existing file untouched.")
        return events

    events.to_csv(config.STAR_EVENTS_CSV, index=False)
    print(
        f"Wrote {len(events):,} events for {events['repo'].nunique()} repos "
        f"to {config.STAR_EVENTS_CSV.name}"
    )
    return events


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="refetch every repo from scratch instead of resuming",
    )
    args = parser.parse_args(argv)
    refresh(full=args.full)


if __name__ == "__main__":
    main()
