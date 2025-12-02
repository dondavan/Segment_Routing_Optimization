import networkx as nx
import numpy as np
import os
import hashlib
import json
import matplotlib.pyplot as plt

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




def draw_path_in_graph(G, colored_paths, ingress, egress, save_dir=None):
    """
    Draw the original graph and overlay colored paths on top of it.
    colored_paths: dict of {color: [path]}
    save_dir: directory to save the plot (if provided)
    """
    plt.figure(figsize=(14, 12))  # Larger figure for readability
    pos = nx.spectral_layout(G, center=(4, 4), scale=10)
    _node_size = 900

    # Draw base graph
    nx.draw_networkx_nodes(G, pos, node_size=_node_size, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")  # Smaller font

    # Overlay colored paths
    for color, paths in colored_paths.items():
        for path in paths:
            if len(path) > 1:
                edges = [(path[i-1], path[i]) for i in range(1, len(path))]
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=2.5, edge_color=color, alpha=0.85)
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=_node_size//1.7, node_color=color, alpha=0.5)

    # Highlight ingress & egress
    nx.draw_networkx_nodes(G, pos, nodelist=[ingress], node_size=_node_size, node_color="green")
    nx.draw_networkx_nodes(G, pos, nodelist=[egress], node_size=_node_size, node_color="red")

    plt.title("Graph with Colored Paths Overlay", fontsize=18)
    plt.axis('off')
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if ingress is not None and egress is not None:
            colored_paths_str = json.dumps(colored_paths, sort_keys=True)
            hash_digest = hashlib.md5(colored_paths_str.encode()).hexdigest()
            filename = f"graph_with_colored_paths_{str(ingress).replace(' ', '')}_{str(egress).replace(' ', '')}_{hash_digest}.png"
        else:
            # fallback to hash if no ingress/egress
            colored_paths_str = json.dumps(colored_paths, sort_keys=True)
            hash_digest = hashlib.md5(colored_paths_str.encode()).hexdigest()
            filename = f"graph_with_colored_paths_{hash_digest}.png"
        plt.savefig(os.path.join(save_dir, filename), format='png')
        plt.close()
    else:
        plt.show()

def plot_sa_objective_history(history, edge_attributes, node_attributes, save_dir, pair_name=None):
    os.makedirs(save_dir, exist_ok=True)
    history_arr = list(zip(*history))
    plt.figure(figsize=(10, 6))
    for i, obj_name in enumerate(edge_attributes + node_attributes):
        plt.plot(history_arr[i], label=obj_name)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Simulated Annealing Objective Trajectory' + (f' ({pair_name})' if pair_name else ''))
    plt.legend()
    fname = f'sa_objective_trajectory{f"_{pair_name}" if pair_name else ""}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()