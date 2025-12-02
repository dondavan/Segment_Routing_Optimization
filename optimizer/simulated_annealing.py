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
        # keep current pareto objectives and history of pareto snapshots (list of lists of objective tuples)
        self.pareto_objectives = []
        self.pareto_history = []

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
        """Update Pareto archive.
        - front: list of (path, obj_list)
        - candidate: (path, obj_list)
        Returns new front with dominated entries removed and candidate added if nondominated.
        Also updates self.pareto_objectives (list of objective tuples) and self.pareto_history.
        """
        cand_obj = tuple(candidate[1])
        new_front = []
        # If any existing dominates candidate, candidate is dominated -> skip adding
        for p, o in front:
            o_t = tuple(o)
            if self.dominates(o_t, cand_obj):
                # candidate is dominated by existing front member
                return front
        # Candidate is not dominated by existing members; remove any existing members it dominates
        for p, o in front:
            o_t = tuple(o)
            if self.dominates(cand_obj, o_t):
                # skip dominated existing
                continue
            new_front.append((p, o))
        # add candidate
        new_front.append(candidate)
        # update internal objective list and history
        objs = [tuple(o) for (_p, o) in new_front]
        objs_sorted = sorted(set(objs))
        self.pareto_objectives = objs_sorted
        self.pareto_history.append(objs_sorted.copy())
        return new_front

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
                return [], False, []

        # Full objectives (for reporting) and opt objectives (for Pareto/acceptance)
        current_full_obj = self.path_objectives(G, current_path, node_attributes, edge_attributes)
        current_opt = self.path_opt_objectives(G, current_path)

        pareto_front = []
        # keep a mapping of path tuple -> full objectives for final reporting
        results_full = {tuple(current_path): current_full_obj}
        pareto_front = self.update_pareto_front(pareto_front, (copy.deepcopy(current_path), current_opt))

        temp = initial_temp
        iter_count = 0
        solutions = []  # record accepted solutions as [HopCount, Integrity]
        # record initial accepted solution
        solutions.append([current_opt[0], -current_opt[1]])
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

            # record evaluated neighbor's full objectives so we can report later
            results_full.setdefault(tuple(neighbor), neighbor_full_obj)
            # update Pareto archive with the evaluated neighbor (even if not accepted)
            pareto_front = self.update_pareto_front(pareto_front, (copy.deepcopy(neighbor), neighbor_opt))

            neighbor_score = tuple(self.sa_objective(G, neighbor))
            current_score = tuple(self.sa_objective(G, current_path))

            # acceptance on scalarized score (sum of opt components) using sa_objective sign convention
            delta = (neighbor_score[0] - current_score[0]) + (neighbor_score[1] - current_score[1])
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # accept neighbor as current path
                current_path = neighbor
                current_full_obj = neighbor_full_obj
                current_opt = neighbor_opt
                current_score = neighbor_score

                # record the newly accepted solution (hopcount, integrity positive)
                solutions.append([current_opt[0], -current_opt[1]])

                # update best if improved
                if neighbor_score < best_obj:
                    best_path = neighbor
                    best_obj = neighbor_score

                # also update pareto archive with the accepted current path (may already be present)
                pareto_front = self.update_pareto_front(pareto_front, (copy.deepcopy(current_path), current_opt))

            # store the full objectives for this path (current_path)
            results_full.setdefault(tuple(current_path), current_full_obj)

            temp *= cooling_rate
            iter_count += 1
        
        # Finalize Pareto set: map archive entries to their full objectives and remove duplicates
        unique = {}
        for p, o in pareto_front:
            p_tuple = tuple(p)
            if p_tuple not in unique and p_tuple in results_full:
                unique[p_tuple] = results_full[p_tuple]

        # Return best path found (or Pareto set) and indicate whether a pareto front was found; also return solutions history
        if not unique:
            return [([best_path], self.path_objectives(G, best_path, node_attributes, edge_attributes))], False, solutions
        return [(list(p), o) for p, o in unique.items()], True, solutions