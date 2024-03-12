
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_decisions_per_action(data: pd.DataFrame, name: str, figsize=(8, 8)):
    data = data[data['reward'] != 0]

    fig, ax = plt.subplots(figsize=figsize)

    actions = data["action"].unique()
    actions = np.sort(actions)

    # Create violins for each action
    for i, action in enumerate(actions):
        action_data = data[data["action"] == action]

        ax.bar(x=i, height=len(action_data))

    # Set labels and title
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title(f"Selected action count ({name})")

    ax.set_xticks(np.arange(len(actions)), actions, rotation=45)

    # Add grid and adjust layout
    ax.grid(True)
    plt.tight_layout()

    return fig
