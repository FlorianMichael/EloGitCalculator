import git
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import argparse
import json

parser = argparse.ArgumentParser(description='Elo Ratings for Git Contributors')
parser.add_argument('--repo_path', type=str, default='', help='Path to the git repository')
parser.add_argument('--branch', type=str, default='master', help='Branch to analyze commits')
parser.add_argument('--initial_rating', type=int, default=1500, help='Initial Elo rating for contributors')
parser.add_argument('--K', type=int, default=32, help='K-factor for Elo rating adjustments')
parser.add_argument('--inactivity_threshold', type=int, default=30, help='Days without commits before being considered inactive')
parser.add_argument('--stagnant_elo_threshold', type=int, default=30, help='Days at 1500 before hiding contributor')
parser.add_argument('--days_to_show', type=int, default=365, help='Number of days of data to show in the animation')
parser.add_argument('--top_contributors_count', type=int, default=5, help='Number of top contributors to display')
parser.add_argument('--name_mapping_file', type=str, default='name_mapping.json', help='Path to JSON file for author name mapping')

args = parser.parse_args()

repo_path = args.repo_path
repo = git.Repo(repo_path)

commits = list(repo.iter_commits(args.branch))

with open(args.name_mapping_file) as f:
    name_mapping = json.load(f)  # Load mappings from JSON file

data = []
for commit in commits:
    author = commit.author.name
    commit_date = datetime.fromtimestamp(commit.committed_date)
    
    # Apply the mapping
    canonical_author = name_mapping.get(author, author)
    data.append([canonical_author, commit_date])

df = pd.DataFrame(data, columns=['Author', 'Date'])

df['Date'] = df['Date'].dt.date  # Strip the time part from the datetime
daily_commits = df.groupby(['Date', 'Author']).size().unstack(fill_value=0)

elo_ratings = {author: args.initial_rating for author in daily_commits.columns}
elo_history = {author: [] for author in daily_commits.columns}

# Elo system parameters
K = args.K  # The K-factor, controls how much the Elo rating adjusts per day

# Inactivity and 1500 threshold settings
inactivity_threshold = args.inactivity_threshold  # Days without commits before being considered inactive
stagnant_elo_threshold = args.stagnant_elo_threshold  # Days at 1500 before hiding contributor

max_elo = {author: args.initial_rating for author in daily_commits.columns}

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
            # Contributor did not make any commits today
            elo_history[author].append(elo_ratings[author])
            continue
        
        actual_score = daily_activity[author] / total_commits
        expected_scores = {
            opponent: expected_score(elo_ratings[author], elo_ratings[opponent])
            for opponent in daily_activity.index if opponent != author
        }
        expected_total_score = sum(expected_scores.values()) / len(expected_scores)
        
        elo_ratings[author] += K * (actual_score - expected_total_score)
        elo_history[author].append(elo_ratings[author])
        
        if elo_ratings[author] > max_elo[author]:
            max_elo[author] = elo_ratings[author]
        
        # Reset the counter if their Elo rating changes from 1500
        if elo_ratings[author] != 1500:
            days_at_1500[author] = 0
        else:
            days_at_1500[author] += 1

elo_df = pd.DataFrame(elo_history, index=daily_commits.index)

last_active_day = {author: daily_commits.index[0] for author in daily_commits.columns}

for author in daily_commits.columns:
    for day in daily_commits.index:
        if daily_commits[author][day] > 0:
            last_active_day[author] = day

plt.style.use('dark_background')
fig, ax = plt.subplots()
lines = {author: ax.plot([], [], label=author)[0] for author in elo_df.columns}
ax.set_ylim(min(min(elo_df.min()), 1200), max(max(elo_df.max()), 1800))  # Elo ratings between 1200 and 1800
ax.set_xlabel('Date', color='white')
ax.set_ylabel('Elo Rating', color='white')
ax.set_title('Contributor Elo Ratings Over Time', color='white')
ax.tick_params(axis='x', colors='white')  # X-axis ticks color
ax.tick_params(axis='y', colors='white')  # Y-axis ticks color
ax.grid(color='gray')  # Grid color

text_labels = {author: ax.text(0, 0, '', color=line.get_color(), fontsize=9) for author, line in lines.items()}

text_box = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

def update(frame):
    current_date = elo_df.index[frame]
    window_start = max(0, frame - args.days_to_show + 1)  # Display the last configurable number of days of data
    ax.set_xlim(elo_df.index[window_start], current_date)
    
    # Sort contributors by their highest Elo rating ever
    sorted_contributors = sorted(max_elo.items(), key=lambda x: x[1], reverse=True)
    
    top_contributors_text = "Top Contributors (Highest Elo):\n"
    for i, (author, elo) in enumerate(sorted_contributors[:args.top_contributors_count]):
        top_contributors_text += f"{i+1}. {author}: {int(elo)}\n"
    text_box.set_text(top_contributors_text)
    
    for author, line in lines.items():
        inactive = (current_date - last_active_day[author]).days > inactivity_threshold
        stagnant = days_at_1500[author] > stagnant_elo_threshold
        
        if not inactive and not stagnant:
            if elo_history[author][frame] != args.initial_rating:
                line.set_alpha(1)  # Ensure the line is visible
                line.set_data(elo_df.index[:frame+1], elo_df[author][:frame+1])
                
                # Position the name of the contributor at the last point of the line
                if frame > 0:
                    x = elo_df.index[frame]
                    y = elo_df[author][frame]
                    text_labels[author].set_position((x, y))
                    text_labels[author].set_text(author)
                    text_labels[author].set_alpha(1 if y > 0 else 0)  # Hide label if no activity
            else:
                line.set_alpha(0)  # Hide the line if they haven't committed yet
                text_labels[author].set_alpha(0)  # Hide the text label
        else:
            # Stop updating the contributor's line but keep it visible at the last known point
            line.set_alpha(1)  # Keep the line visible
            text_labels[author].set_alpha(1)  # Keep the label visible at the last known point
    
    return list(lines.values()) + list(text_labels.values()) + [text_box]


ani = FuncAnimation(fig, update, frames=len(elo_df), blit=False, repeat=False)

plt.show()
