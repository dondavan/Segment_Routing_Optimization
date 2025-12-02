import networkx as nx
import random

def find_complex_ingress_egress_pairs(G, num_pairs=5, min_distance=3):
    # Find node pairs with long shortest paths (complex connectivity)
    all_pairs = dict(nx.all_pairs_shortest_path_length(G))
    candidates = []
    for u in G.nodes():
        for v in G.nodes():
            if u != v and v in all_pairs[u]:
                dist = all_pairs[u][v]
                if dist >= min_distance:
                    candidates.append((dist, u, v))
    # Sort by distance descending, pick top unique pairs
    candidates.sort(reverse=True)
    pairs = []
    used = set()
    for dist, u, v in candidates:
        if (u, v) not in used and (v, u) not in used:
            pairs.append((u, v))
            used.add((u, v))
            if len(pairs) >= num_pairs:
                break
    return pairs


def get_graph(dim=(5,5), ingress=(0,0), egress=(4,4), seed=None, extra_edges_factor=1.5, num_pairs=5, min_distance=3):
    if seed is not None:
        random.seed(seed)

    # Directed grid: edges only go right and down
    G = nx.DiGraph()
    rows, cols = dim
    for x in range(rows):
        for y in range(cols):
            if x < rows - 1:
                G.add_edge((x, y), (x+1, y))
            if y < cols - 1:
                G.add_edge((x, y), (x, y+1))

    nodes = list(G.nodes())
    if ingress not in nodes:
        raise RuntimeError('Ingress node cannot fit in graph')
    if egress not in nodes:
        raise RuntimeError('Egress node cannot fit in graph')

    # Add random extra unidirectional edges for complexity
    possible_edges = [(u, v) for u in nodes for v in nodes if u != v and not G.has_edge(u, v)]
    num_extra = int(extra_edges_factor * (rows + cols))
    random.shuffle(possible_edges)
    added = 0
    for u, v in possible_edges:
        G.add_edge(u, v)
        if nx.is_strongly_connected(G):
            added += 1
        else:
            G.remove_edge(u, v)
        if added >= num_extra:
            break

    # Find complex ingress/egress pairs
    pairs = find_complex_ingress_egress_pairs(G, num_pairs=num_pairs, min_distance=min_distance)
    # Ensure each pair is connected
    for ingress, egress in pairs:
        if not nx.has_path(G, ingress, egress):
            undirected = nx.Graph(G)
            try:
                sp = nx.shortest_path(undirected, ingress, egress)
                for u, v in zip(sp[:-1], sp[1:]):
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
            except nx.NetworkXNoPath:
                continue  # skip this pair if not possible
    return G, pairs
