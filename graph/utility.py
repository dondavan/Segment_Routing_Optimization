import networkx as nx
import numpy as np
import os
import hashlib
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Initialize Graph Edge Parameter
def init_edge_weight(G, edge_attributes, policy):
    
    for attribute in edge_attributes:
        weights = {}

        for edge in G.edges():
            weight = np.random.choice(policy[attribute])
            weights[edge] = weight
        
        nx.set_edge_attributes(G, weights, attribute)

# Initialize Graph Node Parameter
def init_node_weight(G, node_attributes, policy):
    
    for attribute in node_attributes:
        weights = {}

        for node in G.nodes():
            weight = np.random.choice(policy[attribute])
            weights[node] = weight

        nx.set_node_attributes(G, weights, attribute)


def draw_graph(G, node_attributes, edge_attributes, ingress, egress, save_dir=None):
    fig, ax = plt.subplots(figsize=(14, 12))  
    _node_size = 400  # Smaller nodes

    # Use one fixed layout
    pos = nx.spectral_layout(G, scale=10)  # Spread out more

    # --- Draw nodes and edges ---
    nx.draw_networkx_nodes(G, pos, node_size=_node_size, ax=ax)
    nx.draw_networkx_edges(G, pos, width=4, edge_color='black', ax=ax)  # Thicker, more visible edges

    # ================================================================
    # Draw node attributes (stacked under nodes)
    # ================================================================
    for idx, attr in enumerate(node_attributes):
        node_attribute = nx.get_node_attributes(G, attr)

        # vertical offset so labels don't overlap
        offset = (idx + 1.2) * 0.25  # More vertical offset
        label_pos = {n: (x, y + offset) for n, (x, y) in pos.items()}

        node_labels = {(u, v): f"{attr[0]}: {val.name}" for (u, v), val in node_attribute.items()}
        

        nx.draw_networkx_labels(
            G,
            label_pos,
            labels=node_labels,
            font_size=8,  # Smaller font
            font_family="sans-serif",
            ax=ax
        )

    # Highlight ingress & egress nodes
    if ingress is not None and egress is not None:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[ingress, egress], 
            node_size=_node_size, ax=ax, node_color="black"
        )

    # ================================================================
    # Draw edge attributes (stacked on edges)
    # ================================================================
    for idx, attr in enumerate(edge_attributes):
        edge_attr = nx.get_edge_attributes(G, attr)

        # Keep only entries for edges that actually exist in G
        edge_attr = {(u, v): val for (u, v), val in edge_attr.items() if G.has_edge(u, v)}

        # vertical offset for edge labels to avoid overlapping each other
        offset = (idx + 1.2) * 0.25  # More vertical offset
        label_pos = {n: (x, y + offset) for n, (x, y) in pos.items()}

        edge_labels = {(u, v): f"{attr[0]}: {val.name}" for (u, v), val in edge_attr.items()}
        nx.draw_networkx_edge_labels(
            G,
            label_pos,
            edge_labels=edge_labels,
            font_size=7,  # Smaller font
            font_color="blue",
            ax=ax
        )

    # ================================================================
    ax.set_title("Node and Edge Attributes", fontsize=18)
    ax.margins(0.1)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if ingress is not None and egress is not None:
            filename = f"network_graph_{str(ingress).replace(' ', '')}_{str(egress).replace(' ', '')}.png"
        else:
            filename = "network_graph.png"
        plt.savefig(os.path.join(save_dir, filename), format='png')
        plt.close()
    else:
        plt.show()




def draw_path_in_graph(G, all_paths, ingress, egress, save_dir=None, attribute_names=None):
    """
    Draw each path on its own figure/image instead of overlaying all paths on a single graph.
    - all_paths can be either:
      * a dict mapping color->list_of_paths (legacy), or
      * an iterable/list of paths (each path is a list of nodes), or
      * an iterable/list of (path, objectives) tuples.
    Each path will be drawn in its own figure and saved separately. Filenames include the
    ingress/egress and a hash of the path for uniqueness. If attribute_names is provided,
    the objectives will be shown in a caption below the plot.
    """
    # Normalize input into a list of (path, color, objectives) tuples. Color may be None.
    path_color_obj = []
    if isinstance(all_paths, dict):
        # Legacy format: color -> [paths]
        for color, paths in all_paths.items():
            for path in paths:
                path_color_obj.append((path, color, None))
    else:
        # all_paths may be list of paths or list of (path, objectives)
        normalized = []
        for item in list(all_paths):
            if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], (str, bytes)):
                # treat as (path, objectives)
                normalized.append(item)
            else:
                # treat item as a path with no objectives
                normalized.append((item, None))

        paths = [p for p, _ in normalized]
        objs = [o for _, o in normalized]
        n_paths = len(paths)
        cmap = plt.get_cmap('berlin')
        generated_colors = [mcolors.to_hex(cmap(i / max(1, n_paths - 1))) for i in range(n_paths)] if n_paths > 0 else []
        for idx, (path, obj) in enumerate(normalized):
            color = generated_colors[idx] if idx < len(generated_colors) else '#444444'
            path_color_obj.append((path, color, obj))

    # Layout computed once for consistency across per-path plots
    pos = nx.spectral_layout(G, center=(4, 4), scale=10)
    _node_size = 900

    for idx, (path, color, objectives) in enumerate(path_color_obj):
        plt.figure(figsize=(14, 14))
        # Draw base graph faintly
        nx.draw_networkx_nodes(G, pos, node_size=_node_size, node_color='lightgray')
        nx.draw_networkx_edges(G, pos, width=2.0, edge_color='lightgray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

        # Skip empty or invalid paths
        if not path or len(path) <= 1:
            title = f"Path {idx+1}: empty or single node"
        else:
            # Draw only the edges/nodes for this path
            edges = [(path[i-1], path[i]) for i in range(1, len(path))]
            # Ensure color is a valid matplotlib color (convert RGBA tuples to hex)
            try:
                draw_color = mcolors.to_hex(color) if not isinstance(color, tuple) or len(color) != 4 else mcolors.to_hex(color)
            except Exception:
                draw_color = str(color)
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=3.0, edge_color=[draw_color], alpha=0.95)
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=_node_size//1.7, node_color=[draw_color], alpha=0.7)
            title = f"Path {idx+1} overlay"

        # Highlight ingress & egress clearly on top
        if ingress is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[ingress], node_size=_node_size, node_color="green")
        if egress is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[egress], node_size=_node_size, node_color="red")

        plt.title(f"Graph with Path {idx+1} Overlay ({ingress} → {egress})", fontsize=16)
        plt.axis('off')

        # If objectives provided and attribute_names given, render a small caption
        if objectives is not None and attribute_names is not None:
            try:
                # pair up names and objective values; handle enums with .name or .value
                pairs = []
                for name, val in zip(attribute_names, objectives):
                    if hasattr(val, 'name'):
                        v = val.name
                    elif hasattr(val, 'value'):
                        v = val.value
                    else:
                        v = val
                    pairs.append(f"{name}: {v}")
                caption = " | ".join(pairs)
                plt.gcf().text(0.5, 0.03, caption, ha='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            except Exception:
                pass

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # Hash the path to create a stable unique filename
            try:
                path_str = json.dumps(path, sort_keys=True)
            except Exception:
                path_str = repr(path)
            hash_digest = hashlib.md5(path_str.encode()).hexdigest()[:10]

            # Try to extract HopCount and Integrity from objectives if provided; otherwise compute from G
            hop_val = None
            int_val = None
            try:
                if objectives is not None and attribute_names is not None:
                    # attribute_names expected to be list like EDGE_ATTRIBUTES + NODE_ATTRIBUTES
                    if 'HopCount' in attribute_names:
                        hi = attribute_names.index('HopCount')
                        hop_val = objectives[hi]
                    if 'Integrity' in attribute_names:
                        ii = attribute_names.index('Integrity')
                        int_val = objectives[ii]
                # fallback to computing from graph
                if hop_val is None:
                    hop_val = 0
                    if path and len(path) > 1:
                        for u, v in zip(path[:-1], path[1:]):
                            val = G[u][v].get('HopCount', 1)
                            if hasattr(val, 'value'):
                                val = val.value
                            hop_val += val
                if int_val is None:
                    int_val = 0
                    for n in (path or []):
                        val = G.nodes[n].get('Integrity', 0)
                        if hasattr(val, 'value'):
                            val = val.value
                        int_val += val
            except Exception:
                hop_val = None
                int_val = None

            hop_str = f"hop{str(hop_val).replace(' ', '')}" if hop_val is not None else "hopNA"
            int_str = f"int{str(int_val).replace(' ', '')}" if int_val is not None else "intNA"

            filename = f"graph_with_path_{str(ingress).replace(' ', '')}_{str(egress).replace(' ', '')}_{idx+1}_{hop_str}_{int_str}.png"
            plt.savefig(os.path.join(save_dir, filename), format='png')
            plt.close()
        else:
            plt.show()

def plot_sa_objective_history(history, edge_attributes, node_attributes, save_dir, pair_name=None):
    """Plot accepted SA solutions as a scatter: x=HopCount (minimize), y=Integrity (maximize).
    Highlight an approximate Pareto front computed from the points.
    - history: list of [hopcount, integrity] pairs (integrity positive)
    - edge_attributes/node_attributes kept for compatibility but not used directly here
    """
    os.makedirs(save_dir, exist_ok=True)
    if not history:
        return
    # history may be list of pairs or list of lists; normalize
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c='tab:blue', alpha=0.6, label='accepted solutions')

    # approximate nondominated (Pareto) points: no other point has <= hopcount and >= integrity with at least one strict
    pareto = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        dominated = False
        for j, (x2, y2) in enumerate(zip(xs, ys)):
            if j == i:
                continue
            if x2 <= x and y2 >= y and (x2 < x or y2 > y):
                dominated = True
                break
        if not dominated:
            pareto.append((x, y))

    if pareto:
        # make unique and sort by hopcount asc, integrity desc
        pareto = sorted(set(pareto), key=lambda t: (t[0], -t[1]))
        px, py = zip(*pareto)
        plt.plot(px, py, '-o', color='crimson', label='approx. Pareto front')

    plt.xlabel('HopCount (lower is better)')
    plt.ylabel('Integrity (higher is better)')
    plt.title('SA solutions: HopCount vs Integrity' + (f' ({pair_name})' if pair_name else ''))
    plt.grid(alpha=0.3)
    plt.legend()
    fname = f'sa_objective_scatter{f"_{pair_name}" if pair_name else ""}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

def plot_pareto_and_history(G, pareto_front, solutions, save_dir, pair_name=None):
    """Plot accepted SA solutions (HopCount vs Integrity) and overlay the returned Pareto-front paths.

    - G: networkx graph (used to compute hopcount/integrity for paths in pareto_front)
    - pareto_front: list of (path, objectives) where path is list of nodes
    - solutions: list of [hopcount, integrity] accepted during SA (chronological)
    - save_dir: directory to save the plot
    - pair_name: optional label to include in filename/title
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare solution points
    sol_x = [s[0] for s in solutions] if solutions else []
    sol_y = [s[1] for s in solutions] if solutions else []

    # Prepare pareto front points (compute hopcount & integrity from paths)
    pf_points = []
    for path, _ in (pareto_front or []):
        # compute hopcount from edge attribute
        hop = 0
        for u, v in zip(path[:-1], path[1:]):
            val = G[u][v].get('HopCount', 1)
            if hasattr(val, 'value'):
                val = val.value
            hop += val
        # compute integrity from node attribute (sum)
        integrity = 0
        for n in path:
            val = G.nodes[n].get('Integrity', 0)
            if hasattr(val, 'value'):
                val = val.value
            integrity += val
        pf_points.append((hop, integrity))

    # Sort Pareto points by hop ascending, integrity descending for nicer plotting
    pf_points = sorted(set(pf_points), key=lambda t: (t[0], -t[1]))
    pf_x = [p[0] for p in pf_points]
    pf_y = [p[1] for p in pf_points]

    plt.figure(figsize=(12, 9))

    # plot history points and connect them to show trajectory (light gray)
    if sol_x and sol_y:
        plt.plot(sol_x, sol_y, '-o', color='lightgray', alpha=0.55, linewidth=1, markersize=6, label='SA trajectory')
        # Make accepted solutions more visible by adding a subtle black edge and bringing them above the trajectory
        plt.scatter(sol_x, sol_y, c='tab:blue', alpha=0.95, s=40, label='accepted solutions', edgecolors='black', linewidths=0.6, zorder=4)

    # plot pareto front points emphasized
    if pf_x and pf_y:
        # thick connecting line for front
        plt.plot(pf_x, pf_y, '-o', color='crimson', linewidth=2, markersize=14, alpha=0.95, label='returned Pareto front')
        # large star markers with black edge
        plt.scatter(pf_x, pf_y, c='crimson', marker='*', s=300, edgecolors='black', linewidths=0.8, zorder=5)
        # annotate points with indices and small boxes
        for i, (x, y) in enumerate(zip(pf_x, pf_y)):
            plt.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(8,6), fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.xlabel('HopCount (lower is better)', fontsize=12)
    plt.ylabel('Integrity (higher is better)', fontsize=12)
    plt.title('Pareto front overlay with SA trajectory' + (f' ({pair_name})' if pair_name else ''), fontsize=14)
    plt.grid(alpha=0.35)
    plt.legend(loc='best', fontsize=11)

    fname = f'pareto_with_history{f"_{pair_name}" if pair_name else ""}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=200)
    plt.close()