import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import os
from datetime import datetime

from optimizer import backtrack,refine
from optimizer.simulated_annealing import SimulatedAnnealing
from graph import grid,utility


# Graph Hyper Parameters
dim=(15, 15)          # Graph Dimension, example 5*5 2d graph
ingress=(0,0)       # Ingress node, (x,y)
egress=(14,14)        # Egress node, (x,y)

# SA Hyper Parameters
cooling_rate=0.99

# Constrain
colors = ['red','blue','yellow']
T = [0.1,0.3,0.5]
I = [0.1,0.3,0.5]


# Graph attributes
NODE_ATTRIBUTES = ['Integrity', 'Resilience', 'Confidentiality']
EDGE_ATTRIBUTES = ['Latency', 'Bandwidth', 'HopCount']

# Optimization Objectives
ATTRIBUTE_VALUES = Enum('ATTRIBUTE_VALUES', [('VL', 1), ('L', 2), ('M', 3), ('H', 4), ('VH', 5)])

POLICY = {
    'Integrity':[ATTRIBUTE_VALUES.M, ATTRIBUTE_VALUES.H, ATTRIBUTE_VALUES.VH],
    'Bandwidth': [ATTRIBUTE_VALUES.M, ATTRIBUTE_VALUES.H, ATTRIBUTE_VALUES.VH],
    'Confidentiality':[ATTRIBUTE_VALUES.VH],
    'HopCount':[ATTRIBUTE_VALUES.VL, ATTRIBUTE_VALUES.L, ATTRIBUTE_VALUES.M, ATTRIBUTE_VALUES.H, ATTRIBUTE_VALUES.VH],
    'Latency':[ATTRIBUTE_VALUES.L, ATTRIBUTE_VALUES.M, ATTRIBUTE_VALUES.H, ATTRIBUTE_VALUES.VH],
    
    'Resilience':[ATTRIBUTE_VALUES.VL, ATTRIBUTE_VALUES.L, ATTRIBUTE_VALUES.M, ATTRIBUTE_VALUES.H, ATTRIBUTE_VALUES.VH],
}

# Load Graph and get complex ingress/egress pairs
G, ingress_egress_pairs = grid.get_graph(dim=dim)

print(ingress_egress_pairs)

utility.init_edge_weight(G, EDGE_ATTRIBUTES, POLICY)
utility.init_node_weight(G, NODE_ATTRIBUTES, POLICY)

# Create a subdirectory in 'plots' with current timestamp
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot_dir = os.path.join('plots', run_timestamp)
os.makedirs(plot_dir, exist_ok=True)

# Run Simulated Annealing (Multi-objective) for each pair
for idx, (ingress, egress) in enumerate(ingress_egress_pairs):
    print(f"\nPair {idx+1}: Ingress={ingress}, Egress={egress}")
    sa = SimulatedAnnealing(ATTRIBUTE_VALUES)
    pareto_front, is_pareto, solutions = sa.solve_simulated_annealing(G, ingress, egress, NODE_ATTRIBUTES, EDGE_ATTRIBUTES, cooling_rate=cooling_rate, save_dir=None)
    if is_pareto:
        print("Pareto-optimal paths found by Simulated Annealing:")
    else:
        print("No Pareto front found, showing best path found by Simulated Annealing:")
    for pidx, (path, objectives) in enumerate(pareto_front):
        print(f"  Path {pidx+1}: {path}")
        print(f"    Objectives: {dict(zip(EDGE_ATTRIBUTES + NODE_ATTRIBUTES, objectives))}")

    color_map = {}
    colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for pidx, (path, _) in enumerate(pareto_front):
        color = colors[pidx % len(colors)]
        color_map.setdefault(color, []).append(path)
    utility.draw_path_in_graph(G, color_map, ingress, egress, save_dir=plot_dir)
    # Also plot Pareto front overlay with SA accepted-solution history
    try:
        utility.plot_pareto_and_history(G, pareto_front, solutions, plot_dir, pair_name=f"{ingress}_{egress}")
    except Exception:
        pass

for idx, (ingress, egress) in enumerate(ingress_egress_pairs):
    utility.draw_graph(G, NODE_ATTRIBUTES, EDGE_ATTRIBUTES, ingress, egress, save_dir=plot_dir)
