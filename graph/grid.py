import networkx as nx
import random

def get_graph(dim=(5,5), ingress=(0,0), egress=(4,4), seed=None):
    if seed is not None:
        random.seed(seed)

    # Undirected grid
    G = nx.grid_graph(dim)
    nodes = list(G.nodes())

    if ingress not in nodes:
        raise RuntimeError('Ingress node cannot fit in graph')
    if egress not in nodes:
        raise RuntimeError('Egress node cannot fit in graph')

    # Start with a fully bidirectional directed graph
    DG = nx.DiGraph()
    DG.add_nodes_from(nodes)

    for u, v in G.edges():
        DG.add_edge(u, v)
        DG.add_edge(v, u)

    # Randomize the order of edges to process
    directed_edges = list(DG.edges())
    random.shuffle(directed_edges)

    # Try removing the reverse direction while maintaining strong connectivity
    for u, v in directed_edges:
        if DG.has_edge(v, u):  # only consider edges that currently have a reverse
            DG.remove_edge(v, u)
            if not nx.is_strongly_connected(DG):
                # restoring to maintain strong connectivity
                DG.add_edge(v, u)

    return DG
