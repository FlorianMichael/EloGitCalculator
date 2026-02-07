# Elo Ratings for Git Contributors

This program analyzes commit history from a specified Git repository, calculates Elo ratings for contributors based on their activity, and visualizes the ratings over time using animated plots.

## Requirements

- Python 3.x
- Install dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive mode

Running the script without any arguments starts an interactive mode:

```bash
python git_ratings_analyzer.py
```

- If a graphical environment and `tkinter` are available, a small window opens where you can set:
  - Repository path
  - Branch
  - Elo parameters (initial rating, K-factor)
  - Inactivity and stagnant-Elo thresholds
  - Days to show
  - Number of top contributors
  - Name mapping file
- If `tkinter` is not available, the script falls back to simple text prompts in the terminal.

### Command-line mode

You can also use the script as a CLI tool by providing arguments. For example:

```bash
python git_ratings_analyzer.py \
  --repo_path /path/to/your/repo \
  --branch main \
  --initial_rating 1500 \
  --K 32 \
  --inactivity_threshold 30 \
  --stagnant_elo_threshold 30 \
  --days_to_show 365 \
  --top_contributors_count 5 \
  --name_mapping_file name_mapping.json
```

### Arguments

- `--repo_path`: Path to the Git repository (default: current directory).
- `--branch`: Branch to analyze commits (default: `master`).
- `--initial_rating`: Initial Elo rating for contributors (default: `1500`).
- `--K`: K-factor for Elo rating adjustments (default: `32`).
- `--inactivity_threshold`: Days without commits before being considered inactive (default: `30`).
- `--stagnant_elo_threshold`: Days at 1500 before hiding a contributor (default: `30`).
- `--days_to_show`: Number of days of data to show in the animation window (default: `365`).
- `--top_contributors_count`: Number of top contributors to display (default: `5`).
- `--name_mapping_file`: Path to a JSON file for author names, resolved relative to the current working directory. The JSON file must be a simple object mapping raw author names to canonical names, e.g.:

  ```json
  {
    "ExampleDude": "Example Dude",
    "OldName": "Example Dude"
  }
  ```

- `--ylim_min`: Optional fixed lower y-axis limit for the Elo plot.
- `--ylim_max`: Optional fixed upper y-axis limit for the Elo plot.
- `--animation_interval_ms`: Animation interval in milliseconds (default: `80`).
- `--top_mode`: Ranking mode for the top contributors box: `max` (highest Elo ever) or `current` (current Elo).

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [GitPython](https://gitpython.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
