import random
import math
import networkx as nx
import copy

from enum import Enum
from graph.utility import plot_sa_objective_history

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

    def random_neighbor(self, G, path, egress, temp=None, initial_temp=None):
        """
        Generate a neighbor path. When temperature is high -> exploration (random simple path);
        when temperature is low -> exploitation (small local modification favoring low HopCount edges).
        """
        # If path too short, nothing to change
        if len(path) <= 2:
            return path

        # If temperature information isn't provided, fall back to previous behavior
        if temp is None or initial_temp is None or initial_temp <= 0:
            # original local-change behavior
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

        # Determine exploration probability (higher temp => more exploration)
        explore_prob = max(0.0, min(1.0, float(temp) / float(initial_temp)))

        if random.random() < explore_prob:
            # Exploration: attempt a random simple path from the same ingress to egress
            try:
                return nx.algorithms.simple_paths.random_simple_path(G, path[0], egress)
            except Exception:
                # fallback to original local move if random path generation fails
                pass

        # Exploitation: perform a small local modification favoring low HopCount edges
        idx = random.randint(1, len(path) - 2)
        u = path[idx-1]
        # candidate neighbors that are not already in the path (to avoid cycles)
        candidates = [n for n in G.neighbors(u) if n != path[idx] and n not in path]
        if not candidates:
            return path
        # score candidates by HopCount (prefer lower hop count)
        def hopcount_of(u, v):
            val = G[u][v].get('HopCount', 1)
            if hasattr(val, 'value'):
                val = val.value
            return val
        candidates.sort(key=lambda n: hopcount_of(u, n))
        # choose from top-K candidates, K scales with temperature (more options when warmer)
        K = max(1, int(1 + (len(candidates)-1) * min(1.0, temp / initial_temp)))
        chosen = random.choice(candidates[:K])
        new_path = path[:idx] + [chosen]
        try:
            suffix = nx.shortest_path(G, new_path[-1], egress)
            new_path += suffix[1:]
            return new_path
        except nx.NetworkXNoPath:
            # if no suffix found, return original path
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

    def path_opt_objectives(self, G, path):
        """Return optimization vector: [HopCount (minimize), -Integrity (minimize because negative maximizes integrity)]."""
        hop_count = 0
        for u, v in zip(path[:-1], path[1:]):
            val = G[u][v].get('HopCount', 1)
            if hasattr(val, 'value'):
                val = val.value
            hop_count += val
        integrity = sum(G.nodes[n].get('Integrity', 0).value if hasattr(G.nodes[n].get('Integrity', 0), 'value') else G.nodes[n].get('Integrity', 0) for n in path)
        return [hop_count, -integrity]

    def dominates(self, a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def update_pareto_front(self, front, candidate):
        print(front)
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

    def meets_constraints(self, G, path, node_attributes, edge_attributes):
        # For every edge: Bandwidth >= M, Latency >= L
        for u, v in zip(path[:-1], path[1:]):
            bandwidth = G[u][v].get('Bandwidth', 0)
            if hasattr(bandwidth, 'value'):
                bandwidth = bandwidth.value
            latency = G[u][v].get('Latency', 0)
            if hasattr(latency, 'value'):
                latency = latency.value
            if bandwidth < 3:  # M = 3
                return False
            if latency < 2:    # L = 2
                return False
        # For every node: Confidentiality >= VH, Integrity >= M
        for n in path:
            confidentiality = G.nodes[n].get('Confidentiality', 0)
            if hasattr(confidentiality, 'value'):
                confidentiality = confidentiality.value
            integrity = G.nodes[n].get('Integrity', 0)
            if hasattr(integrity, 'value'):
                integrity = integrity.value
            if confidentiality < 5:  # VH = 5
                return False
            if integrity < 3:        # M = 3
                return False
        return True

    def sa_objective(self, G, path):
        # Minimize hop count (sum of HopCount edge attribute), maximize integrity (negative for maximization)
        hop_count = 0
        for u, v in zip(path[:-1], path[1:]):
            val = G[u][v].get('HopCount', 1)
            if hasattr(val, 'value'):
                val = val.value
            hop_count += val
        integrity = sum(G.nodes[n].get('Integrity', 0).value if hasattr(G.nodes[n].get('Integrity', 0), 'value') else G.nodes[n].get('Integrity', 0) for n in path)
        return hop_count, -integrity

    def solve_simulated_annealing(self, G, ingress, egress, node_attributes, edge_attributes, initial_temp=100, cooling_rate=0.99, min_temp=1, max_iter=1000, save_dir=None, pair_name=None):
        if pair_name is None:
            pair_name = f"{str(ingress).replace(' ', '')}_{str(egress).replace(' ', '')}"
        # Start with a random valid path from ingress to egress that meets constraints
        found = False
        for _ in range(100):
            try:
                candidate = nx.algorithms.simple_paths.random_simple_path(G, ingress, egress)
                if self.meets_constraints(G, candidate, node_attributes, edge_attributes):
                    current_path = candidate
                    found = True
                    break
            except Exception:
                continue
        if not found:
            # fallback to shortest path if random fails
            current_path = nx.shortest_path(G, ingress, egress)
            if not self.meets_constraints(G, current_path, node_attributes, edge_attributes):
                return [], False

        # Full objectives (for reporting) and opt objectives (for Pareto/acceptance)
        current_full_obj = self.path_objectives(G, current_path, node_attributes, edge_attributes)
        current_opt = self.path_opt_objectives(G, current_path)

        pareto_front = [(copy.deepcopy(current_path), current_opt)]
        # keep a mapping of path tuple -> full objectives for final reporting
        results_full = {tuple(current_path): current_full_obj}

        temp = initial_temp
        iter_count = 0
        history = []  # record opt objectives for SA trajectory
        best_path = current_path
        best_obj = tuple(self.sa_objective(G, current_path))

        while temp > min_temp and iter_count < max_iter:
            neighbor = self.random_neighbor(G, current_path, egress, temp=temp, initial_temp=initial_temp)
            if not self.meets_constraints(G, neighbor, node_attributes, edge_attributes):
                temp *= cooling_rate
                iter_count += 1
                continue
            neighbor_full_obj = self.path_objectives(G, neighbor, node_attributes, edge_attributes)
            neighbor_opt = self.path_opt_objectives(G, neighbor)
            neighbor_score = tuple(self.sa_objective(G, neighbor))
            current_score = tuple(self.sa_objective(G, current_path))
            # acceptance on scalarized score (sum of opt components) using sa_objective sign convention
            delta = (neighbor_score[0] - current_score[0]) + (neighbor_score[1] - current_score[1])
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_path = neighbor
                current_full_obj = neighbor_full_obj
                current_opt = neighbor_opt
                current_score = neighbor_score
                if neighbor_score < best_obj:
                    best_path = neighbor
                    best_obj = neighbor_score
            # update pareto front using opt vectors
            pareto_front = self.update_pareto_front(pareto_front, (copy.deepcopy(current_path), current_opt))
            # store the full objectives for this path
            results_full.setdefault(tuple(current_path), current_full_obj)
            temp *= cooling_rate
            iter_count += 1
            # record hop count (minimize) and integrity (positive, for plotting) while opt vectors keep -integrity
            history.append([current_opt[0], -current_opt[1]])

        # Remove duplicate paths (by node sequence) from pareto list and map to full objectives
        unique = {}
        for p, o in pareto_front:
            p_tuple = tuple(p)
            if p_tuple not in unique and p_tuple in results_full:
                unique[p_tuple] = results_full[p_tuple]

        # Plot the optimization objective history (HopCount, Integrity) if requested
        if save_dir is not None and len(history) > 0:
            # plot_sa_objective_history expects two lists; use HopCount and Integrity names
            plot_sa_objective_history(history, ['HopCount'], ['Integrity'], save_dir, pair_name)

        # If no Pareto front found, return best path and a flag
        if not unique:
            return [([best_path], self.path_objectives(G, best_path, node_attributes, edge_attributes))], False
        return [(list(p), o) for p, o in unique.items()], True