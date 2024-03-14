

def edge(src, rel, dst):
    return src, rel, dst


def is_edge(key):
    return isinstance(key, tuple)


def components(edge):
    return edge[0], edge[1], edge[2]


