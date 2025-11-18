import git
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import sys

try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    tk = None
    messagebox = None


WINDOW_TITLE = "Git Elo Ratings Analyzer"


@dataclass
class EloConfig:
    repo_path: str
    branch: str = "master"
    initial_rating: int = 1500
    K: int = 32 # The K-factor, controls how much the Elo rating adjusts per day
    inactivity_threshold: int = 30 # Days without commits before being considered inactive
    stagnant_elo_threshold: int = 30 # Days at 1500 before hiding contributor
    days_to_show: int = 365
    top_contributors_count: int = 5
    name_mapping_file: str = "name_mapping.json"


@dataclass
class VizConfig:
    ylim_min: int | None = None
    ylim_max: int | None = None
    animation_interval_ms: int = 80
    top_mode: str = "max"


def parse_args() -> tuple[EloConfig, VizConfig]:
    parser = argparse.ArgumentParser(description="Elo Ratings for Git Contributors")
    parser.add_argument("--repo_path", type=str, default="", help="Path to the git repository")
    parser.add_argument("--branch", type=str, default="master", help="Branch to analyze commits")
    parser.add_argument("--initial_rating", type=int, default=1500, help="Initial Elo rating for contributors")
    parser.add_argument("--K", type=int, default=32, help="K-factor for Elo rating adjustments")
    parser.add_argument("--inactivity_threshold", type=int, default=30, help="Days without commits before being considered inactive")
    parser.add_argument("--stagnant_elo_threshold", type=int, default=30, help="Days at 1500 before hiding contributor")
    parser.add_argument("--days_to_show", type=int, default=365, help="Number of days of data to show in the animation")
    parser.add_argument("--top_contributors_count", type=int, default=5, help="Number of top contributors to display")
    parser.add_argument("--name_mapping_file", type=str, default="name_mapping.json", help="Path to JSON or text file for author name mapping")
    parser.add_argument("--ylim_min", type=int, default=None, help="Optional fixed lower y-limit for Elo plot")
    parser.add_argument("--ylim_max", type=int, default=None, help="Optional fixed upper y-limit for Elo plot")
    parser.add_argument("--animation_interval_ms", type=int, default=80, help="Animation interval in milliseconds")
    parser.add_argument("--top_mode", type=str, choices=["max", "current"], default="max", help="Ranking mode for top contributors box")

    args = parser.parse_args()

    repo_path = args.repo_path or "."

    elo_cfg = EloConfig(
        repo_path=repo_path,
        branch=args.branch,
        initial_rating=args.initial_rating,
        K=args.K,
        inactivity_threshold=args.inactivity_threshold,
        stagnant_elo_threshold=args.stagnant_elo_threshold,
        days_to_show=args.days_to_show,
        top_contributors_count=args.top_contributors_count,
        name_mapping_file=args.name_mapping_file,
    )

    viz_cfg = VizConfig(
        ylim_min=args.ylim_min,
        ylim_max=args.ylim_max,
        animation_interval_ms=args.animation_interval_ms,
        top_mode=args.top_mode,
    )

    return elo_cfg, viz_cfg


def load_name_mapping(mapping_path: Path) -> Dict[str, str]:
    if not mapping_path.is_file():
        print(f"Name mapping file not found: {mapping_path}")
        return {}

    try:
        with mapping_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"Name mapping file loaded from {mapping_path} ({len(data)} entries)")
            return {str(k): str(v) for k, v in data.items()}
        else:
            raise ValueError("JSON mapping must be an object (dict) of old->new names")
    except Exception as exc:
        print(f"Error reading name mapping file {mapping_path} as JSON: {exc}")
        return {}


def load_commits(cfg: EloConfig) -> pd.DataFrame:
    repo_path = Path(cfg.repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    try:
        repo = git.Repo(str(repo_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open git repository at {repo_path}: {exc}") from exc

    try:
        commits = list(repo.iter_commits(cfg.branch))
    except Exception as exc:
        raise RuntimeError(f"Failed to iterate commits on branch '{cfg.branch}': {exc}") from exc

    if not commits:
        raise RuntimeError(f"No commits found on branch '{cfg.branch}' in repository {repo_path}")

    mapping_path = Path(cfg.name_mapping_file)
    name_mapping = load_name_mapping(mapping_path)

    data: list[tuple[str, datetime]] = []
    for commit in commits:
        author = commit.author.name
        commit_date = datetime.fromtimestamp(commit.committed_date)
        canonical_author = name_mapping.get(author, author)
        data.append((canonical_author, commit_date))

    df = pd.DataFrame(data, columns=["Author", "Date"])
    df["Date"] = df["Date"].dt.date

    if df.empty:
        raise RuntimeError("No commit data found after processing repository history.")

    return df


def compute_elo(df: pd.DataFrame, cfg: EloConfig):
    daily_commits = df.groupby(["Date", "Author"]).size().unstack(fill_value=0)

    elo_ratings = {author: cfg.initial_rating for author in daily_commits.columns}
    elo_history = {author: [] for author in daily_commits.columns}

    K = cfg.K
    inactivity_threshold = cfg.inactivity_threshold
    stagnant_elo_threshold = cfg.stagnant_elo_threshold

    max_elo = {author: cfg.initial_rating for author in daily_commits.columns}
    days_at_1500 = {author: 0 for author in daily_commits.columns}

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    for day in daily_commits.index:
        daily_activity = daily_commits.loc[day]
        total_commits = daily_activity.sum()

        if total_commits == 0:
            for author in elo_ratings:
                elo_history[author].append(elo_ratings[author])
            continue

        for author in daily_activity.index:
            if daily_activity[author] == 0:
                elo_history[author].append(elo_ratings[author])
                continue

            actual_score = daily_activity[author] / total_commits
            opponents = [opponent for opponent in daily_activity.index if opponent != author]
            if not opponents:
                elo_history[author].append(elo_ratings[author])
                continue

            expected_scores = [expected_score(elo_ratings[author], elo_ratings[opponent]) for opponent in opponents]
            expected_total_score = sum(expected_scores) / len(expected_scores)

            elo_ratings[author] += K * (actual_score - expected_total_score)
            elo_history[author].append(elo_ratings[author])

            if elo_ratings[author] > max_elo[author]:
                max_elo[author] = elo_ratings[author]

            if elo_ratings[author] != cfg.initial_rating:
                days_at_1500[author] = 0
            else:
                days_at_1500[author] += 1

    elo_df = pd.DataFrame(elo_history, index=daily_commits.index)

    last_active_day = {author: daily_commits.index[0] for author in daily_commits.columns}
    for author in daily_commits.columns:
        for day in daily_commits.index:
            if daily_commits[author][day] > 0:
                last_active_day[author] = day

    return elo_df, max_elo, last_active_day, days_at_1500, inactivity_threshold, stagnant_elo_threshold


def create_animation(elo_df: pd.DataFrame, max_elo, last_active_day, days_at_1500, cfg: EloConfig, viz_cfg: VizConfig):
    plt.style.use("dark_background")
    fig = plt.figure(num=WINDOW_TITLE)
    ax = fig.add_subplot(111)

    authors = list(elo_df.columns)
    color_map = plt.get_cmap("tab20")
    colors = {author: color_map(i % 20) for i, author in enumerate(authors)}

    lines = {author: ax.plot([], [], label=author, color=colors[author])[0] for author in authors}

    min_elo = float(elo_df.min().min())
    max_elo_val = float(elo_df.max().max())
    y_min = viz_cfg.ylim_min if viz_cfg.ylim_min is not None else min(min_elo, 1200) - 50
    y_max = viz_cfg.ylim_max if viz_cfg.ylim_max is not None else max(max_elo_val, 1800) + 50
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Elo Rating", color="white")
    ax.set_title(WINDOW_TITLE, color="white")
    ax.tick_params(axis="x", colors="white", labelrotation=45)
    ax.tick_params(axis="y", colors="white")
    ax.grid(color="gray", alpha=0.3)

    text_labels = {author: ax.text(0, 0, "", color=lines[author].get_color(), fontsize=9) for author in authors}

    text_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"),
    )

    inactivity_threshold = cfg.inactivity_threshold
    stagnant_elo_threshold = cfg.stagnant_elo_threshold

    def update(frame: int):
        current_date = elo_df.index[frame]
        window_start = max(0, frame - cfg.days_to_show + 1)
        ax.set_xlim(elo_df.index[window_start], current_date)

        if viz_cfg.top_mode == "current":
            current_elos = elo_df.loc[current_date]
            ranking_source = current_elos.to_dict()
        else:
            ranking_source = max_elo

        sorted_contributors = sorted(ranking_source.items(), key=lambda x: x[1], reverse=True)

        top_contributors_text = (
            "Top Contributors (" + ("Current" if viz_cfg.top_mode == "current" else "Highest Elo") + "):\n"
        )
        for i, (author, elo_val) in enumerate(sorted_contributors[: cfg.top_contributors_count]):
            short_name = author if len(author) <= 25 else author[:22] + "..."
            top_contributors_text += f"{i+1}. {short_name}: {int(elo_val)}\n"
        text_box.set_text(top_contributors_text)

        for author, line in lines.items():
            inactive = (current_date - last_active_day[author]).days > inactivity_threshold
            stagnant = days_at_1500[author] > stagnant_elo_threshold

            y_val = elo_df[author].iloc[frame]

            if not inactive and not stagnant and y_val != cfg.initial_rating:
                line.set_alpha(1)
                line.set_data(elo_df.index[: frame + 1], elo_df[author].iloc[: frame + 1])

                x = elo_df.index[frame]
                y = y_val
                text_labels[author].set_position((x, y + 10))

                if viz_cfg.top_mode == "current":
                    current_elos = elo_df.loc[current_date]
                    ranking_today = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)
                    top_today = {a for a, _ in ranking_today[: cfg.top_contributors_count]}
                else:
                    top_today = {a for a, _ in sorted_contributors[: cfg.top_contributors_count]}

                if author in top_today:
                    text_labels[author].set_text(author)
                    text_labels[author].set_alpha(1)
                else:
                    text_labels[author].set_alpha(0)
            else:
                line.set_alpha(0.3 if y_val != cfg.initial_rating else 0)
                text_labels[author].set_alpha(0)

        return list(lines.values()) + list(text_labels.values()) + [text_box]

    ani = FuncAnimation(fig, update, frames=len(elo_df), blit=False, repeat=False, interval=viz_cfg.animation_interval_ms)
    return fig, ani


def run_with_config(elo_cfg: EloConfig, viz_cfg: VizConfig) -> None:
    try:
        df = load_commits(elo_cfg)
        elo_df, max_elo, last_active_day, days_at_1500, _, _ = compute_elo(df, elo_cfg)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    _, _ = create_animation(elo_df, max_elo, last_active_day, days_at_1500, elo_cfg, viz_cfg)
    plt.show()


def _interactive_cli() -> None:
    print("Starting interactive mode (text-based)...")
    repo_path = input("Repository path (default: .): ").strip() or "."
    branch = input("Branch (default: master): ").strip() or "master"

    def _read_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} (default: {default}): ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Invalid integer, using default.")
            return default

    initial_rating = _read_int("Initial Elo rating", 1500)
    K = _read_int("K-factor", 32)
    inactivity_threshold = _read_int("Inactivity threshold (days)", 30)
    stagnant_elo_threshold = _read_int("Stagnant Elo threshold (days)", 30)
    days_to_show = _read_int("Days to show in animation", 365)
    top_contributors_count = _read_int("Number of top contributors", 5)

    name_mapping_file = input("Name mapping file (default: name_mapping.json): ").strip() or "name_mapping.json"

    elo_cfg = EloConfig(
        repo_path=repo_path,
        branch=branch,
        initial_rating=initial_rating,
        K=K,
        inactivity_threshold=inactivity_threshold,
        stagnant_elo_threshold=stagnant_elo_threshold,
        days_to_show=days_to_show,
        top_contributors_count=top_contributors_count,
        name_mapping_file=name_mapping_file,
    )

    viz_cfg = VizConfig()
    run_with_config(elo_cfg, viz_cfg)


def run_interactive_ui() -> None:
    if tk is None:
        _interactive_cli()
        return

    root = tk.Tk()
    root.title(WINDOW_TITLE)

    fields = {
        "Repository path": (".", str),
        "Branch": ("master", str),
        "Initial Elo rating": ("1500", int),
        "K-factor": ("32", int),
        "Inactivity threshold (days)": ("30", int),
        "Stagnant Elo threshold (days)": ("30", int),
        "Days to show in animation": ("365", int),
        "Number of top contributors": ("5", int),
        "Name mapping file": ("name_mapping.json", str),
        "Y min (optional)": ("", int),
        "Y max (optional)": ("", int),
        "Animation interval (ms)": ("80", int),
        "Top mode (max/current)": ("max", str),
    }

    entries: dict[str, tk.Entry] = {}

    row = 0
    for label_text, (default_value, _type) in fields.items():
        label = tk.Label(root, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        entry = tk.Entry(root, width=40)
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, padx=5, pady=2)
        entries[label_text] = entry
        row += 1

    def on_run():
        try:
            repo_path = entries["Repository path"].get().strip() or "."
            branch = entries["Branch"].get().strip() or "master"

            def _get_int_field(key: str, default: int | None) -> int | None:
                raw = entries[key].get().strip()
                if raw == "":
                    return default
                return int(raw)

            initial_rating = _get_int_field("Initial Elo rating", 1500)
            K = _get_int_field("K-factor", 32)
            inactivity_threshold = _get_int_field("Inactivity threshold (days)", 30)
            stagnant_elo_threshold = _get_int_field("Stagnant Elo threshold (days)", 30)
            days_to_show = _get_int_field("Days to show in animation", 365)
            top_contributors_count = _get_int_field("Number of top contributors", 5)
            ylim_min = _get_int_field("Y min (optional)", None)
            ylim_max = _get_int_field("Y max (optional)", None)
            animation_interval_ms = _get_int_field("Animation interval (ms)", 80)

            name_mapping_file = entries["Name mapping file"].get().strip() or "name_mapping.json"

            top_mode_raw = entries["Top mode (max/current)"].get().strip().lower() or "max"
            top_mode = "current" if top_mode_raw == "current" else "max"

        except ValueError as exc:
            if messagebox is not None:
                messagebox.showerror("Invalid input", f"Please enter valid integers: {exc}")
            else:
                print(f"Invalid input: {exc}")
            return

        elo_cfg = EloConfig(
            repo_path=repo_path,
            branch=branch,
            initial_rating=int(initial_rating),
            K=int(K),
            inactivity_threshold=int(inactivity_threshold),
            stagnant_elo_threshold=int(stagnant_elo_threshold),
            days_to_show=int(days_to_show),
            top_contributors_count=int(top_contributors_count),
            name_mapping_file=name_mapping_file,
        )

        viz_cfg = VizConfig(
            ylim_min=ylim_min,
            ylim_max=ylim_max,
            animation_interval_ms=int(animation_interval_ms),
            top_mode=top_mode,
        )

        root.destroy()
        run_with_config(elo_cfg, viz_cfg)

    run_button = tk.Button(root, text="Start analysis", command=on_run)
    run_button.grid(row=row, column=0, columnspan=2, pady=10)

    root.mainloop()


def main() -> None:
    if len(sys.argv) == 1:
        run_interactive_ui()
        return

    elo_cfg, viz_cfg = parse_args()
    run_with_config(elo_cfg, viz_cfg)


if __name__ == "__main__":
    main()
