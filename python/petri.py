import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

class Petri:
    def __init__(self, log):
        self.petri: PetriNet = None
        self.im: Marking = None
        self.fm: Marking = None
        self.log = log

    def create_process_model(self, method: str, heuristic_params: dict = None, inductive_params: dict = None):
        if method == "alpha":
            self.petri, self.im, self.fm = pm4py.discover_petri_net_alpha(log=self.log)
        elif method == "heuristic":
            self.petri, self.im, self.fm = pm4py.discover_petri_net_heuristics(
                log=self.log,
                dependency_threshold=heuristic_params.get("dependency_threshold", 0.5),
                and_threshold=heuristic_params.get("and_threshold", 0.65),
                loop_two_threshold=heuristic_params.get("loop_two_threshold", 0.5),
            )
        elif method == "inductive":
            print(inductive_params.get("noise_threshold", 0.0))
            self.petri, self.im, self.fm = pm4py.discover_petri_net_inductive(
                log=self.log,
                noise_threshold=inductive_params.get("noise_threshold", 0.0),
                disable_fallthroughs=inductive_params.get("disable_fallthroughs", False),
            )
        else:
            raise ValueError("Unknown process discovery method")


    def view_process_model(self):
        pm4py.view_petri_net(self.petri, self.im, self.fm, log=self.log)

    def find_decision_points(self):
        """
        Algorithm 1: Recursive method for specifying the possible decisions
        at a decision point in terms of sets of log events.
        """
        decision_points = {}

        for place in self.petri.places:
            # collect outgoing transitions from this place
            outgoing_transitions = [arc.target for arc in self.petri.arcs if arc.source == place]

            # skip if not a decision point
            if len(outgoing_transitions) <= 1:
                continue

            decision_classes = []

            # while outgoing edges left (loop over each outgoing transition)
            for t in outgoing_transitions:
                current_class = set()

                # if (t ≠ invisible) ∧ (t ≠ duplicate)
                if self.is_visible(t) and not self.is_duplicate(t):
                    current_class.add(t.label)
                else:
                    # else currentClass ← traceDecisionClass(t)
                    current_class = self.trace_decision_class(t)

                # if currentClass ≠ ∅ then add it
                if current_class:
                    decision_classes.append(current_class)

            # add to result if we found any decision classes
            if len(decision_classes) > 1 :
                decision_points[place.name] = decision_classes

        return decision_points

    def trace_decision_class(self, t):
        """
        Algorithm 1 (continued): traceDecisionClass
        Recursively find visible labels reachable from t,
        but stop if a join construct is encountered.
        """
        decision_class = set()

        # while successor places of passed transition left
        succ_places = [arc.target for arc in self.petri.arcs if arc.source == t]
        for p in succ_places:
            # if p = join construct then return ∅
            if self.is_join_construct(p):
                return set()

            # while successor transitions of p left
            succ_transitions = [arc.target for arc in self.petri.arcs if arc.source == p]
            for t2 in succ_transitions:
                if self.is_visible(t2) and not self.is_duplicate(t2):
                    decision_class.add(t2.label)
                else:
                    result = self.trace_decision_class(t2)
                    if not result:
                        # join found deeper down → return ∅
                        return set()
                    else:
                        decision_class |= result

        return decision_class

    # helper methods (same semantics)
    def is_visible(self, t):
        # visible if transition has a non-empty label
        label = getattr(t, "label", None)
        return bool(label)

    def is_duplicate(self, t):
        label = getattr(t, "label", None)
        if not label:
            return False
        same = [tr for tr in self.petri.transitions if getattr(tr, "label", None) == label]
        return len(same) > 1

    def is_join_construct(self, place):
        # join construct if place has more than one incoming arc
        incoming = [arc.source for arc in self.petri.arcs if arc.target == place]
        return len(incoming) > 1
