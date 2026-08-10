"""GitHub API client for owner classification and contributor statistics.

Produces repos_classified*.csv and contributor_detailed_stats*.csv.
Run via `python -m healthcare_ai.github_api --mode ...`, or through refresh.py.

Requires GITHUB_TOKEN in the environment (or a .env file). Unauthenticated
requests are limited to 60/hour, which is not enough to classify 385 owners.
"""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from healthcare_ai import config


def get_token(required: bool = True) -> str | None:
    """Read GITHUB_TOKEN from the environment, loading .env if present."""
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token and required:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Add it to your environment or to a .env "
            "file in the project root:\n\n    GITHUB_TOKEN=ghp_...\n\n"
            "Create one at https://github.com/settings/tokens (public_repo "
            "scope is sufficient)."
        )
    return token


class GitHubContributionAnalyzer:
    def __init__(self, token: str = None):
        """
        Initialize the analyzer with optional GitHub token for higher rate limits.

        Args:
            token: GitHub personal access token (optional but recommended)
        """
        self.base_url = config.GITHUB_API_BASE
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def check_if_organization(self, owner: str) -> dict:
        """
        Check if a GitHub account is an organization or personal account.

        Args:
            owner: GitHub username/organization name

        Returns:
            Dictionary with account type and details
        """
        url = f"{self.base_url}/users/{owner}"

        try:
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                account_type = data.get("type", "Unknown")

                return {
                    "owner": owner,
                    "type": account_type,  # 'Organization' or 'User'
                    "is_organization": account_type == "Organization",
                    "name": data.get("name", ""),
                    "public_repos": data.get("public_repos", 0),
                    "status": "success",
                }
            else:
                return {
                    "owner": owner,
                    "type": "Unknown",
                    "is_organization": None,
                    "name": "",
                    "public_repos": 0,
                    "status": f"error_{response.status_code}",
                }

        except Exception as e:
            return {
                "owner": owner,
                "type": "Unknown",
                "is_organization": None,
                "name": "",
                "public_repos": 0,
                "status": f"exception: {str(e)}",
            }

    def classify_repos_from_csv(
        self,
        csv_path: str,
        repo_column: str = "Repository",
        output_path: str = "repos_classified.csv",
        rate_limit_delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Classify repositories as organization or personal from CSV.

        Args:
            csv_path: Path to input CSV
            repo_column: Name of the column containing repository names
            output_path: Path to save output CSV
            rate_limit_delay: Delay between API calls (seconds)

        Returns:
            DataFrame with classification results
        """
        print("=" * 80)
        print("GITHUB REPOSITORY CLASSIFIER")
        print("=" * 80)

        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"\n✓ Loaded {len(df)} repositories from CSV")

        # Extract unique owners
        unique_owners = set()
        for repo_full in df[repo_column]:
            if pd.notna(repo_full):
                try:
                    owner, repo = repo_full.split("/", 1)
                    unique_owners.add(owner)
                except ValueError:
                    print(f"⚠️  Skipping invalid repo format: {repo_full}")

        print(f"✓ Found {len(unique_owners)} unique owners to classify")
        print(
            f"⏱️  Estimated time: ~{len(unique_owners) * rate_limit_delay / 60:.1f} minutes"
        )
        print("=" * 80 + "\n")

        # Check each owner
        owner_info = {}
        for i, owner in enumerate(sorted(unique_owners), 1):
            print(f"[{i}/{len(unique_owners)}] Checking {owner}...")
            info = self.check_if_organization(owner)
            owner_info[owner] = info

            if i < len(unique_owners):
                time.sleep(rate_limit_delay)

        # Add classification to original dataframe
        df["owner"] = df[repo_column].apply(
            lambda x: x.split("/")[0] if pd.notna(x) and "/" in x else ""
        )
        df["owner_type"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("type", "Unknown")
        )
        df["is_organization"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("is_organization", None)
        )
        df["owner_name"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("name", "")
        )

        # Save results
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")

        # Print summary
        self.print_classification_summary(df)

        # Save separate files for org and personal repos
        org_repos = df[df["is_organization"] == True]
        personal_repos = df[df["is_organization"] == False]

        org_output = output_path.replace(".csv", "_organizations.csv")
        personal_output = output_path.replace(".csv", "_personal.csv")

        org_repos.to_csv(org_output, index=False)
        personal_repos.to_csv(personal_output, index=False)

        print(f"✓ Organization repos saved to: {org_output}")
        print(f"✓ Personal repos saved to: {personal_output}")

        return df

    def classify_repos_from_urls(
        self,
        repo_urls: list[str],
        output_path: str = "repos_classified.csv",
        rate_limit_delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Classify repositories from a list of GitHub URLs.

        Args:
            repo_urls: List of GitHub repository URLs
            output_path: Path to save output CSV
            rate_limit_delay: Delay between API calls (seconds)

        Returns:
            DataFrame with classification results
        """
        print("=" * 80)
        print("GITHUB REPOSITORY CLASSIFIER - URL MODE")
        print("=" * 80)

        # Parse URLs to extract owner/repo
        repos = []
        for url in repo_urls:
            # Handle various GitHub URL formats
            url = url.strip()
            if "github.com/" in url:
                # Extract owner/repo from URL
                parts = url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner = parts[0]
                    repo = parts[1].rstrip(".git")
                    repos.append(
                        {
                            "url": url,
                            "owner": owner,
                            "repo": repo,
                            "full_name": f"{owner}/{repo}",
                        }
                    )

        df = pd.DataFrame(repos)
        print(f"\n✓ Parsed {len(df)} repository URLs")

        # Extract unique owners
        unique_owners = df["owner"].unique()
        print(f"✓ Found {len(unique_owners)} unique owners to classify")
        print(
            f"⏱️  Estimated time: ~{len(unique_owners) * rate_limit_delay / 60:.1f} minutes"
        )
        print("=" * 80 + "\n")

        # Check each owner
        owner_info = {}
        for i, owner in enumerate(unique_owners, 1):
            print(f"[{i}/{len(unique_owners)}] Checking {owner}...")
            info = self.check_if_organization(owner)
            owner_info[owner] = info

            if i < len(unique_owners):
                time.sleep(rate_limit_delay)

        # Add classification to dataframe
        df["owner_type"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("type", "Unknown")
        )
        df["is_organization"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("is_organization", None)
        )
        df["owner_name"] = df["owner"].map(
            lambda x: owner_info.get(x, {}).get("name", "")
        )

        # Save results
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")

        # Print summary
        self.print_classification_summary(df)

        # Save separate files
        org_repos = df[df["is_organization"] == True]
        personal_repos = df[df["is_organization"] == False]

        org_output = output_path.replace(".csv", "_organizations.csv")
        personal_output = output_path.replace(".csv", "_personal.csv")

        org_repos.to_csv(org_output, index=False)
        personal_repos.to_csv(personal_output, index=False)

        print(f"✓ Organization repos saved to: {org_output}")
        print(f"✓ Personal repos saved to: {personal_output}")

        return df

    def print_classification_summary(self, df: pd.DataFrame):
        """Print classification summary."""
        print("\n" + "=" * 80)
        print("CLASSIFICATION SUMMARY")
        print("=" * 80 + "\n")

        org_count = (df["is_organization"] == True).sum()
        personal_count = (df["is_organization"] == False).sum()
        unknown_count = df["is_organization"].isna().sum()

        print(f"🏢 Organization repos: {org_count} ({org_count / len(df) * 100:.1f}%)")
        print(
            f"👤 Personal repos: {personal_count} ({personal_count / len(df) * 100:.1f}%)"
        )
        print(f"❓ Unknown: {unknown_count} ({unknown_count / len(df) * 100:.1f}%)")
        print(f"📊 Total: {len(df)}")

        print("\n" + "-" * 80)
        print("TOP ORGANIZATIONS:")
        print("-" * 80)
        org_repos = df[df["is_organization"] == True]
        if len(org_repos) > 0:
            top_orgs = org_repos["owner"].value_counts().head(10)
            for i, (org, count) in enumerate(top_orgs.items(), 1):
                org_name = df[df["owner"] == org]["owner_name"].iloc[0]
                display_name = f"{org} ({org_name})" if org_name else org
                print(f"{i:2d}. {display_name:40s} - {count} repos")

        print("\n" + "-" * 80)
        print("TOP PERSONAL ACCOUNTS:")
        print("-" * 80)
        personal_repos = df[df["is_organization"] == False]
        if len(personal_repos) > 0:
            top_users = personal_repos["owner"].value_counts().head(10)
            for i, (user, count) in enumerate(top_users.items(), 1):
                user_name = df[df["owner"] == user]["owner_name"].iloc[0]
                display_name = f"{user} ({user_name})" if user_name else user
                print(f"{i:2d}. {display_name:40s} - {count} repos")

        print("=" * 80 + "\n")

    def get_user_stats_for_repo(self, owner: str, repo: str, username: str) -> dict:
        """
        Get contribution statistics for a specific user in a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            username: GitHub username to check

        Returns:
            Dictionary with additions, deletions, and total commits
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/stats/contributors"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 202:
                # GitHub is computing stats, wait and retry
                print(f"  Stats computing for {owner}/{repo}, retrying...")
                time.sleep(2)
                response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                contributors = response.json()

                for contributor in contributors:
                    if contributor["author"]["login"].lower() == username.lower():
                        total_additions = sum(week["a"] for week in contributor["weeks"])
                        total_deletions = sum(week["d"] for week in contributor["weeks"])
                        total_commits = contributor["total"]

                        return {
                            "username": username,
                            "repo": f"{owner}/{repo}",
                            "additions": total_additions,
                            "deletions": total_deletions,
                            "commits": total_commits,
                            "net_lines": total_additions - total_deletions,
                            "status": "success",
                        }

                return {
                    "username": username,
                    "repo": f"{owner}/{repo}",
                    "additions": 0,
                    "deletions": 0,
                    "commits": 0,
                    "net_lines": 0,
                    "status": "not_contributor",
                }

            elif response.status_code == 404:
                return {
                    "username": username,
                    "repo": f"{owner}/{repo}",
                    "additions": 0,
                    "deletions": 0,
                    "commits": 0,
                    "net_lines": 0,
                    "status": "repo_not_found",
                }
            else:
                return {
                    "username": username,
                    "repo": f"{owner}/{repo}",
                    "additions": 0,
                    "deletions": 0,
                    "commits": 0,
                    "net_lines": 0,
                    "status": f"error_{response.status_code}",
                }

        except Exception as e:
            return {
                "username": username,
                "repo": f"{owner}/{repo}",
                "additions": 0,
                "deletions": 0,
                "commits": 0,
                "net_lines": 0,
                "status": f"exception: {str(e)}",
            }

    def load_contributors_from_csv(
        self,
        csv_path: str,
        contributor_column: str = "Top Contributors",
        repo_column: str = "Repository",
    ) -> list[tuple[str, str, str]]:
        """
        Load contributors from the healthcare CSV file.

        Args:
            csv_path: Path to the CSV file
            contributor_column: Name of the column containing contributors
            repo_column: Name of the column containing repository names

        Returns:
            List of tuples (username, owner, repo)
        """
        df = pd.read_csv(csv_path)

        user_repo_combinations = []
        seen_combinations = set()

        for idx, row in df.iterrows():
            if pd.notna(row[contributor_column]) and pd.notna(row[repo_column]):
                repo_full = row[repo_column]

                # Parse owner and repo from 'owner/repo' format
                try:
                    owner, repo = repo_full.split("/", 1)
                except ValueError:
                    print(f"⚠️  Skipping invalid repo format: {repo_full}")
                    continue

                # Parse contributors (comma-separated)
                contributors = str(row[contributor_column]).split(", ")

                for contributor in contributors:
                    contributor_clean = contributor.strip()

                    # Skip bots
                    if contributor_clean.lower() in [
                        "dependabot",
                        "dependabot[bot]",
                        "dependabot-preview[bot]",
                        "github-actions[bot]",
                    ]:
                        continue

                    # Create unique combination
                    combination = (contributor_clean, owner, repo)

                    if combination not in seen_combinations:
                        user_repo_combinations.append(combination)
                        seen_combinations.add(combination)

        print(
            f"\n✓ Loaded {len(user_repo_combinations)} unique contributor-repository combinations"
        )
        print(f"✓ From {len(df)} repositories")

        return user_repo_combinations

    def analyze_all_from_csv(
        self,
        csv_path: str,
        output_path: str = "contributor_detailed_stats.csv",
        rate_limit_delay: float = 0.5,
    ) -> pd.DataFrame:
        """
        Analyze all contributors from CSV and save detailed statistics.

        Args:
            csv_path: Path to input CSV
            output_path: Path to save output CSV
            rate_limit_delay: Delay between API calls (seconds)

        Returns:
            DataFrame with detailed contribution statistics
        """
        print("=" * 80)
        print("GITHUB CONTRIBUTION ANALYZER - CSV MODE")
        print("=" * 80)

        # Load user-repo combinations
        user_repo_list = self.load_contributors_from_csv(csv_path)

        if not user_repo_list:
            print("❌ No valid contributor-repository combinations found!")
            return pd.DataFrame()

        print(f"\n🔍 Starting analysis of {len(user_repo_list)} combinations...")
        print(
            f"⏱️  Estimated time: ~{len(user_repo_list) * rate_limit_delay / 60:.1f} minutes"
        )
        print("=" * 80 + "\n")

        # Analyze contributions
        results = []
        for i, (username, owner, repo) in enumerate(user_repo_list, 1):
            print(
                f"[{i}/{len(user_repo_list)}] Analyzing {username} in {owner}/{repo}..."
            )
            stats = self.get_user_stats_for_repo(owner, repo, username)
            results.append(stats)

            # Rate limiting
            if i < len(user_repo_list):
                time.sleep(rate_limit_delay)

            # Progress update every 50 repos
            if i % 50 == 0:
                print(
                    f"\n--- Progress: {i}/{len(user_repo_list)} ({i / len(user_repo_list) * 100:.1f}%) ---\n"
                )

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Create aggregated statistics per user
        print("\n" + "=" * 80)
        print("AGGREGATING USER STATISTICS")
        print("=" * 80 + "\n")

        user_stats = (
            df.groupby("username")
            .agg(
                {
                    "commits": "sum",
                    "additions": "sum",
                    "deletions": "sum",
                    "net_lines": "sum",
                    "repo": "count",
                }
            )
            .reset_index()
        )

        user_stats.columns = [
            "username",
            "total_commits",
            "total_additions",
            "total_deletions",
            "total_net_lines",
            "repo_count",
        ]

        # Calculate additional metrics
        user_stats["avg_additions_per_repo"] = (
            user_stats["total_additions"] / user_stats["repo_count"]
        ).round(0)
        user_stats["avg_commits_per_repo"] = (
            user_stats["total_commits"] / user_stats["repo_count"]
        ).round(1)
        user_stats["lines_per_commit"] = (
            (user_stats["total_net_lines"] / user_stats["total_commits"])
            .replace([float("inf"), -float("inf")], 0)
            .round(1)
        )

        # Sort by total additions (lines of code contributed)
        user_stats = user_stats.sort_values("total_additions", ascending=False)

        # Save detailed results
        df.to_csv(output_path, index=False)
        print(f"✓ Detailed results saved to: {output_path}")

        # Save aggregated results
        agg_output_path = output_path.replace(".csv", "_aggregated.csv")
        user_stats.to_csv(agg_output_path, index=False)
        print(f"✓ Aggregated results saved to: {agg_output_path}")

        # Print summary
        self.print_summary(user_stats)

        return df, user_stats

    def print_summary(self, user_stats: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("TOP CONTRIBUTORS SUMMARY")
        print("=" * 80 + "\n")

        print("🏆 TOP 10 BY TOTAL LINES ADDED:")
        print("-" * 80)
        top_10_additions = user_stats.head(10)
        for i, row in top_10_additions.iterrows():
            print(
                f"{row.name + 1:2d}. {row['username']:20s} | "
                f"Additions: {row['total_additions']:>10,} | "
                f"Repos: {row['repo_count']:>3.0f} | "
                f"Commits: {row['total_commits']:>6.0f}"
            )

        print("\n🎯 TOP 10 BY NET LINES CONTRIBUTED:")
        print("-" * 80)
        top_10_net = user_stats.nlargest(10, "total_net_lines")
        for i, (idx, row) in enumerate(top_10_net.iterrows(), 1):
            print(
                f"{i:2d}. {row['username']:20s} | "
                f"Net Lines: {row['total_net_lines']:>10,} | "
                f"Repos: {row['repo_count']:>3.0f} | "
                f"Commits: {row['total_commits']:>6.0f}"
            )

        print("\n📊 OVERALL STATISTICS:")
        print("-" * 80)
        print(f"Total unique contributors: {len(user_stats)}")
        print(f"Total lines added: {user_stats['total_additions'].sum():,}")
        print(f"Total lines deleted: {user_stats['total_deletions'].sum():,}")
        print(f"Total net lines: {user_stats['total_net_lines'].sum():,}")
        print(f"Total commits: {user_stats['total_commits'].sum():,.0f}")
        print(f"Average repos per contributor: {user_stats['repo_count'].mean():.1f}")
        print("=" * 80 + "\n")


def classify_owners(token: str = None, rate_limit_delay: float = None) -> pd.DataFrame:
    """Refresh repos_classified*.csv from the seed dataset."""
    analyzer = GitHubContributionAnalyzer(token=token or get_token())
    return analyzer.classify_repos_from_csv(
        csv_path=str(config.HEALTHCARE_DATA_CSV),
        output_path=str(config.REPOS_CLASSIFIED_CSV),
        rate_limit_delay=rate_limit_delay or config.RATE_LIMIT_DELAY,
    )


def analyze_contributions(
    token: str = None, rate_limit_delay: float = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh contributor_detailed_stats*.csv from the seed dataset."""
    analyzer = GitHubContributionAnalyzer(token=token or get_token())
    return analyzer.analyze_all_from_csv(
        csv_path=str(config.HEALTHCARE_DATA_CSV),
        output_path=str(config.CONTRIBUTOR_STATS_CSV),
        rate_limit_delay=rate_limit_delay or config.RATE_LIMIT_DELAY,
    )


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("classify", "contributions"),
        help="classify: owner org/user classification. "
        "contributions: per-contributor commit and line statistics.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=config.RATE_LIMIT_DELAY,
        help=f"seconds between API calls (default: {config.RATE_LIMIT_DELAY})",
    )
    args = parser.parse_args(argv)

    token = get_token()
    if args.mode == "classify":
        print("Classifying repository owners...\n")
        classify_owners(token, args.rate_limit_delay)
    else:
        print("Analyzing contributions...\n")
        analyze_contributions(token, args.rate_limit_delay)


if __name__ == "__main__":
    main()
