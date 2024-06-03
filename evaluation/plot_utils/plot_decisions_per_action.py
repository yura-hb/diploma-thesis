
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_decisions_per_action(data: pd.DataFrame, name: str, figsize=(8, 8), ax=None):
    data = data[data['reward'] != 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    actions = data["action"].unique()
    actions = np.sort(actions)

    ax.grid(True, zorder=0)

    # Create violins for each action
    for i, action in enumerate(actions):
        action_data = data[data["action"] == action]

        ax.bar(x=i, height=len(action_data), zorder=3)

    # Set labels and title
    # ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of actions ({name})")

    ax.set_xticks(np.arange(len(actions)), actions, rotation=45)

    # Add grid and adjust layout
    plt.tight_layout()

