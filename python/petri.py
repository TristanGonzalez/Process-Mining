import pm4py
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer


class Petri:
    def __init__(self, log):
        self.petri = None
        self.im = None
        self.fm = None
        self.log = log

    def create_process_model(self, method):
        if method == 'alpha':
            self.petri, self.im, self.fm = pm4py.discover_petri_net_alpha(log=self.log)
        elif method == 'heuristic':
            self.petri, self.im, self.fm = pm4py.discover_petri_net_heuristics(log=self.log)
        elif method == 'inductive':
            self.petri, self.im, self.fm = pm4py.discover_petri_net_inductive(log=self.log)
        else:
            raise ValueError("Method not of right kind")


    def view_process_model(self):
        pm4py.view_petri_net(self.petri, self.im, self.fm, log=self.log)


    def find_decision_points(self):
        decision_points = {}

        for place in self.petri.places:
            # outgoing transitions from this place
            outgoing_transitions = [arc.target for arc in self.petri.arcs if arc.source == place]
            if len(outgoing_transitions) <= 1:
                continue

            # Precompute reachability (places) for each outgoing transition
            reachable_places_per_branch = []
            for t in outgoing_transitions:
                rp = self.reachable_places_from_transition(t)
                reachable_places_per_branch.append(rp)

            decision_classes = []
            empty_count = 0

            for idx, t in enumerate(outgoing_transitions):
                # When tracing branch idx, provide other branches' reachable sets
                other_reachable = set().union(*[s for j, s in enumerate(reachable_places_per_branch) if j != idx])

                if self.is_visible(t) and not self.is_duplicate(t):
                    current = {t.label}
                else:
                    current = self.trace_decision_class(t, other_reachable, visited=set())
                if current:
                    decision_classes.append(current)
                else:
                    empty_count += 1

            if len(decision_classes) > 0:
                # If some branches had no visible evidence, include explicit tau/do-nothing branch
                if empty_count > 0:
                    decision_classes.append({"tau (do nothing)"})
                decision_points[place.name] = decision_classes

        return decision_points

    def reachable_places_from_transition(self, t_start):
        """
        Breadth-first search collecting places reachable from transition t_start.
        This traversal ignores the join-stopping rule; it's used to detect merges.
        We follow transitions and places, but avoid infinite loops by visited set.
        """
        visited_places = set()
        visited_transitions = set()
        queue = []

        # successors: place arcs from transition
        succ_places = [arc.target for arc in self.petri.arcs if arc.source == t_start]
        for p in succ_places:
            queue.append(p)

        while queue:
            p = queue.pop(0)
            if p in visited_places:
                continue
            visited_places.add(p)

            # transitions outgoing from p
            succ_t = [arc.target for arc in self.petri.arcs if arc.source == p]
            for t in succ_t:
                if t in visited_transitions:
                    continue
                visited_transitions.add(t)
                # add successor places of t
                for a in self.petri.arcs:
                    if a.source == t:
                        if a.target not in visited_places:
                            queue.append(a.target)

        return visited_places

    def trace_decision_class(self, t, other_reachable_places, visited):
        """
        Recursively find visible labels reachable from t, but stop (return empty)
        if we encounter any place that is reachable from other branches (merge).
        visited is per-branch set of transitions/places to avoid infinite recursion.
        """
        decision_labels = set()

        # prevent infinite recursion across transitions
        if t in visited:
            return set()
        visited.add(t)

        succ_places = [arc.target for arc in self.petri.arcs if arc.source == t]
        if not succ_places:
            return set()

        for p in succ_places:
            # If this place appears in other branches' reachable set, it's a true merge: abort.
            if p in other_reachable_places:
                # merging join encountered — cannot attribute downstream activities to this branch
                return set()

            # Otherwise, continue; but avoid looping forever: track visited places too
            if ('place', p) in visited:
                continue
            visited.add(('place', p))

            succ_transitions = [arc.target for arc in self.petri.arcs if arc.source == p]
            for t2 in succ_transitions:
                # visible and unique transition => candidate label
                if self.is_visible(t2) and not self.is_duplicate(t2):
                    decision_labels.add(t2.label)
                else:
                    # recurse with a copy of visited for per-path safety
                    result = self.trace_decision_class(t2, other_reachable_places, visited)
                    # if result is empty due to hitting a merge deeper down, we must return empty per paper
                    if not result:
                        return set()
                    decision_labels |= result

        return decision_labels

    # helpers (unchanged mostly)
    def is_visible(self, t):
        # visible if label is present and non-empty (t.label could be None or "")
        return getattr(t, "label", None)

    def is_duplicate(self, t):
        label = getattr(t, "label", None)
        if not label:
            return False
        same = [tr for tr in self.petri.transitions if getattr(tr, "label", None) == label]
        return len(same) > 1

    def is_join_construct(self, place):
        incoming = [arc.source for arc in self.petri.arcs if arc.target == place]
        return len(incoming) > 1
