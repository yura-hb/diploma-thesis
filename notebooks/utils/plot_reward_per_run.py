
import numpy as np
import matplotlib.pyplot as plt


def plot_reward_per_run(data, info, format_group):
    metric = info['metric']
    reward = info['reward']
    group = info['group']

    fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    for run in np.sort(data[group].unique()):
        filtered = data[data[group] == run]

        ax.plot(filtered[metric], filtered[reward], marker=info['marker'], ls='', label=format_group(run))

    ax.grid(True)
    ax.legend(ncols=info.get('ncols', 2),
              bbox_to_anchor=info.get('bbox_to_anchor', (-0.08, 1)),
              loc='best',
              fancybox=True,)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    plt.tight_layout()

    return fig