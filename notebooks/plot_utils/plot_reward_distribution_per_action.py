import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_reward_distribution_per_action(data: pd.DataFrame, name: str, figsize=(8, 8)):
    data = data[data['reward'] != 0]

    fig, ax = plt.subplots(figsize=figsize)

    actions = data["action"].unique()
    actions = np.sort(actions)

    # Create violins for each action
    for i, action in enumerate(actions):
        action_data = data[data["action"] == action]["reward"]
        violin_parts = ax.violinplot(
            action_data,
            positions=[i],
            showmeans=True,
            showextrema=True,
            quantiles=[0.25, 0.5, 0.75],
        )

        violin_parts['bodies'][0].set_linewidth(2)

    # Set labels and title
    ax.set_xlabel("Action")
    ax.set_ylabel("Reward")
    ax.set_title(f"Violin Plot of Rewards by Action ({ name })")

    ax.set_xticks(np.arange(len(actions)), actions, rotation=45)

    # Add grid and adjust layout
    ax.grid(True)
    plt.tight_layout()

    return fig
