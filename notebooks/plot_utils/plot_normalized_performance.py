import matplotlib.pyplot as plt
import numpy as np


def plot_normalized_performance(df, info, ax=None):
    index_column = info['index']
    metric = info['metric']
    candidate_column = info['candidate_column']
    baseline = info['baseline']

    if ax is None:
        fig, ax = plt.subplots(figsize=info.get('figsize', (12, 6)))

    candidates = info.get('candidates', np.sort(df[candidate_column].unique()))
    candidates = [candidate for candidate in candidates if candidate != baseline]

    baseline_info = df[df[candidate_column] == baseline].set_index(index_column)

    def performance(candidate):
        candidate_info = df[df[candidate_column] == candidate].set_index(index_column)

        delta = (baseline_info[metric] - candidate_info[metric]) / (baseline_info[metric] + 1e-10)
        delta = delta[~np.isnan(delta)]

        return delta

    candidates = [(candidate, performance(candidate)) for candidate in candidates if candidate != baseline]

    if info.get('sort', True):
        candidates = sorted(candidates, key=lambda x: np.mean(x[1]), reverse=True)

        if top_k := info.get('top_k', None):
            candidates = candidates[:top_k]

    for index, candidate in enumerate(candidates):
        delta = candidate[1]
        delta *= 100

        ax.boxplot(delta, positions=[index], notch=True, vert=True, showmeans=True, widths=info.get('box_width', 0.4))

    ax.yaxis.grid(True)

    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    ax.set_title(info['title'])

    ax.set_ylim(bottom=info.get('bottom'), top=info.get('top'))

    start, end = ax.get_ylim()

    ax.yaxis.set_ticks(np.arange(start, end, info.get('y_step', 5)))
    ax.set_xticks(np.arange(len(candidates)), [candidate[0] for candidate in candidates], rotation=90)

    plt.tight_layout()

    if ax is None:
        return fig, candidates
    else:
        return candidates
