import random
import math
import networkx as nx
import copy

from enum import Enum

class SimulatedAnnealing:
    ATTRIBUTE_VALUES = Enum('ATTRIBUTE_VALUES', [('VL', 1), ('L', 2), ('M', 3), ('H', 4), ('VH', 5)])
    
    def __init__(self, ATTRIBUTE_VALUES):
        self.ATTRIBUTE_VALUES = ATTRIBUTE_VALUES
    
    def attribute_str_to_value(self, attr_str):
        """
        Map a string like 'VL', 'L', 'M', 'H', 'VH' to its corresponding numerical value from ATTRIBUTE_VALUES Enum.
        """
        try:
            return self.ATTRIBUTE_VALUES[attr_str].value
        except (KeyError, AttributeError):
            raise ValueError(f"Unknown attribute string: {attr_str}")

    def path_cost(self, G, path):
        cost = 0
        for u, v in zip(path[:-1], path[1:]):
            val = G[u][v].get('Latency', 1)
            if hasattr(val, 'value'):
                val = val.value
            cost += val
        return cost

    def random_neighbor(self, G, path, egress):
        if len(path) <= 2:
            return path
        idx = random.randint(1, len(path) - 2)
        neighbors = list(G.neighbors(path[idx-1]))
        neighbors = [n for n in neighbors if n != path[idx] and n not in path]
        if not neighbors:
            return path
        new_path = path[:idx] + [random.choice(neighbors)]
        try:
            suffix = nx.shortest_path(G, new_path[-1], egress)
            new_path += suffix[1:]
            return new_path
        except nx.NetworkXNoPath:
            return path

    def path_objectives(self, G, path, node_attributes, edge_attributes):
        edge_obj = {attr: 0 for attr in edge_attributes}
        node_obj = {attr: 0 for attr in node_attributes}
        for u, v in zip(path[:-1], path[1:]):
            for attr in edge_attributes:
                val = G[u][v].get(attr, 0)
                if hasattr(val, 'value'):
                    val = val.value
                edge_obj[attr] += val
        for n in path:
            for attr in node_attributes:
                val = G.nodes[n].get(attr, 0)
                if hasattr(val, 'value'):
                    val = val.value
                node_obj[attr] += val
        return [edge_obj[attr] for attr in edge_attributes] + [node_obj[attr] for attr in node_attributes]

    def dominates(self, a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def update_pareto_front(self, front, candidate):
        non_dominated = []
        candidate_dominated = False
        for p, obj in front:
            if self.dominates(obj, candidate[1]):
                candidate_dominated = True
                break
            elif self.dominates(candidate[1], obj):
                continue
            else:
                non_dominated.append((p, obj))
        if not candidate_dominated:
            non_dominated.append(candidate)
        return non_dominated

    def solve_simulated_annealing(self, G, ingress, egress, node_attributes, edge_attributes, initial_temp=100, cooling_rate=0.99, min_temp=1, max_iter=1000, save_dir=None, pair_name=None):
        if pair_name is None:
            pair_name = f"{str(ingress).replace(' ', '')}_{str(egress).replace(' ', '')}"
        # Start with a random valid path from ingress to egress
        try:
            current_path = nx.algorithms.simple_paths.random_simple_path(G, ingress, egress)
        except Exception:
            # fallback to shortest path if random fails
            current_path = nx.shortest_path(G, ingress, egress)
        current_obj = self.path_objectives(G, current_path, node_attributes, edge_attributes)
        pareto_front = [(current_path, current_obj)]
        temp = initial_temp
        iter_count = 0
        # Record objectives at each iteration
        history = []
        while temp > min_temp and iter_count < max_iter:
            neighbor = self.random_neighbor(G, current_path, egress)
            neighbor_obj = self.path_objectives(G, neighbor, node_attributes, edge_attributes)
            if self.dominates(neighbor_obj, current_obj) or random.random() < math.exp(-sum([n-c for n,c in zip(neighbor_obj, current_obj)]) / temp):
                current_path = neighbor
                current_obj = neighbor_obj
            pareto_front = self.update_pareto_front(pareto_front, (copy.deepcopy(current_path), neighbor_obj))
            temp *= cooling_rate
            iter_count += 1
            history.append(current_obj[:])
        # Remove duplicate paths (by node sequence)
        unique = {}
        for p, o in pareto_front:
            p_tuple = tuple(p)
            if p_tuple not in unique:
                unique[p_tuple] = o
        # Plot the objective history if requested
        if save_dir is not None and len(history) > 0:
            from graph.utility import plot_sa_objective_history
            plot_sa_objective_history(history, edge_attributes, node_attributes, save_dir, pair_name)
        # If no Pareto front found, return best path and a flag
        if not unique:
            # Find best path by minimum sum of objectives in history
            if history:
                best_idx = min(range(len(history)), key=lambda i: sum(history[i]))
                best_obj = history[best_idx]
                return [([current_path], best_obj)], False
            else:
                return [], False
        return [(list(p), o) for p, o in unique.items()], True