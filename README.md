# Elo Ratings for Git Contributors

This program analyzes commit history from a specified Git repository, calculates Elo ratings for contributors based on their activity, and visualizes the ratings over time using animated plots.

## Features

- **Elo Rating Calculation**: Uses the Elo rating system to rank contributors based on their commit activity.
- **Dynamic Visualization**: Animates Elo ratings over time, allowing for a clear view of contributions and changes in ratings.
- **Customizable Settings**: Easily configure the repository path, branch, Elo parameters, inactivity thresholds, and more.

## Requirements

- Python 3.x
- Required Python libraries:
  - `gitpython`
  - `pandas`
  - `matplotlib`

You can install the required libraries using pip:

```bash
pip install gitpython pandas matplotlib
```

## Usage

To run the script, execute the following command in your terminal:

```bash
python your_script.py --repo_path /path/to/your/repo --branch your_branch_name --initial_rating 1500 --K 32 --inactivity_threshold 30 --stagnant_elo_threshold 30 --days_to_show 365 --top_contributors_count 5 --name_mapping_file name_mapping.json
```

### Arguments:

- `--repo_path`: Path to the Git repository.
- `--branch`: The branch to analyze commits (default: `master`).
- `--initial_rating`: Initial Elo rating for contributors (default: `1500`).
- `--K`: K-factor for Elo rating adjustments (default: `32`).
- `--inactivity_threshold`: Days without commits before being considered inactive (default: `30`).
- `--stagnant_elo_threshold`: Days at 1500 before hiding contributor (default: `30`).
- `--days_to_show`: Number of days of data to show in the animation (default: `365`).
- `--top_contributors_count`: Number of top contributors to display (default: `5`).
- `--name_mapping_file`: Path to a JSON file containing author name mappings, allowing you to consolidate multiple author names into a single canonical name. For example, a `name_mapping.json` file might contain:
  ```json
	{
		"EnZaXD": "EnZaXD",
		"FlorianMichael": "EnZaXD"
	}
  ```

## Example

```bash
python git_ratings_analyzer.py --repo_path /path/to/your/repo --branch main --initial_rating 1500 --K 32 --inactivity_threshold 30 --stagnant_elo_threshold 30 --days_to_show 365 --top_contributors_count 5 --name_mapping_file name_mapping.json
```

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [GitPython](https://gitpython.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)