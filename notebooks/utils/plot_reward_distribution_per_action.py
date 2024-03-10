import pandas as pd
import matplotlib.pyplot as plt


def plot_reward_distribution_per_action(data: pd.DataFrame, figsize=(8, 8)):
    data = data[data['reward'] != 0]

    fig, ax = plt.subplots(figsize=figsize)

    # Create violins for each action
    for i, action in enumerate(data["action"].unique()):
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
    ax.set_title("Violin Plot of Rewards by Action")

    # Add grid and adjust layout
    ax.grid(True)
    plt.tight_layout()

    return fig
