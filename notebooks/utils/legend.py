

def add_legend(ax, info):
    ax.legend(ncols=info.get('ncols', 2),
              bbox_to_anchor=info.get('bbox_to_anchor', (-0.08, 1)),
              loc='best',
              fancybox=True)
