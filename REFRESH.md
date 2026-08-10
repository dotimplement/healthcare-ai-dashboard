# Quarterly refresh runbook

The whole loop, roughly 1–2 hours of wall time (most of it waiting on the
GitHub API).

## 0. Prerequisites

```bash
uv sync
echo "GITHUB_TOKEN=ghp_..." > .env    # public_repo scope is enough
```

## 1. Update the seed dataset (manual)

`healthcare_data.csv` is an **input**, not an output. Its `Category`, `Subcat`
and `Standard` columns are editorial judgements and no code generates them.

Before refreshing, add any new repositories and update the star/fork/commit
columns. Keep a copy of the previous quarter in `data_archive/` so you can diff
against it later:

```bash
cp healthcare_data.csv data_archive/healthcare_data_$(date +%Y%m%d).csv
```

The column contract is enforced by `tests/test_data.py` — if you rename or drop
a column, the tests will tell you before the dashboard breaks.

## 2. Run the pipeline

```bash
uv run python refresh.py --dry-run   # confirm the steps
uv run python refresh.py
```

Steps, in dependency order:

| Step | Produces | Notes |
|---|---|---|
| `classify` | `repos_classified*.csv` | One API call per unique owner (~385) |
| `contributions` | `contributor_detailed_stats*.csv` | Slowest step |
| `stars` | `star_events_history.csv` | Incremental by default |
| `taxonomy` | *(report only)* | Lists owners needing a label |

Any step can be run alone:

```bash
uv run python refresh.py --only stars
uv run python refresh.py --skip contributions
```

Star history resumes from what's already in `star_events_history.csv`, so a
normal quarterly run only fetches new stargazers. Star *removals* aren't visible
through the API, so counts drift slightly over time — do a full refetch
occasionally:

```bash
uv run python refresh.py --only stars --full-stars
```

## 3. Update the owner taxonomy

New repositories arrive with owners that have no entry in
`healthcare_ai/owner_types.yaml`, and those repos drop out of the ownership
charts. After a refresh:

```bash
uv run python -m healthcare_ai.owner_types --report
```

Add the listed owners under the appropriate type. Valid types: `Incumbent`,
`Research Lab`, `Startup`, `Community Project/Non-Profit`, `User`.

Matching is on the **owner** only, exact and case-insensitive. Don't add repo
name fragments — the previous substring-based rules classified
`awslabs/fhir-works-on-aws-authz-smart` as a community project because the repo
name contained "smart".

## 4. Verify

```bash
uv run pytest
```

The tests that matter here:

- **`test_owner_coverage_above_threshold`** — fails when too many repos have an
  unlabelled owner. This is the one that will trip after a refresh; fix it by
  doing step 3 properly.
- **`test_chart_year_range_tracks_the_data`** — guards against reintroducing
  hardcoded year bounds.
- **`test_as_of_comes_from_the_data`** — guards against reintroducing
  `datetime.now()` in the activity calculation.
- **`test_star_history_categories_exist_in_the_data`** — catches category names
  in `config.py` that no longer match the CSV.

## 5. Regenerate the charts

```bash
uv run marimo edit charts.py
```

Run all cells. Every `write_image` call refreshes a file in `nice_plots/`.
The notebook prints its as-of date and year range on load — check both before
exporting anything for publication.

## 6. EDA and write-up

```bash
uv run marimo edit charts.py       # explore this quarter vs last
uv run streamlit run main.py       # interactive dashboard
```

To diff against last quarter, load an archived copy alongside the current one:

```python
from healthcare_ai.data import load_repos
current = load_repos()
# previous quarter: point config.HEALTHCARE_DATA_CSV at the archived file,
# or read it directly and compare on Repository
```

## 7. Commit

```bash
git add -A
git commit -m "Q3 2026 refresh"
```

---

## Configuration

Every threshold, date bound, path and category list lives in
`healthcare_ai/config.py`. Notable knobs:

| Setting | Meaning |
|---|---|
| `ACTIVITY_WINDOW_DAYS` | Days since last commit before a repo is "Inactive" |
| `CHART_START_YEAR` | Left edge of published charts (right edge is derived) |
| `STAR_HISTORY_TOP_N` | How many repos to fetch star history for |
| `STAR_HISTORY_CATEGORIES` | Categories in scope for star history |
| `MAX_UNCLASSIFIED_OWNER_SHARE` | Coverage threshold enforced by the tests |

**Never hardcode a year or a date anywhere else.** The upper bound of every
chart is derived from the data (`data.chart_year_range`), and every
time-relative column is measured from the data's own most recent commit
(`data.resolve_as_of`), not from wall-clock time.
