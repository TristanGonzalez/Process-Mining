import pm4py
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer


class petri:
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
        for i, place in enumerate(self.petri.places):
            # outgoing transitions with labels
            outgoing_transitions = [arc.target for arc in self.petri.arcs if
                                    arc.source == place and getattr(arc.target, "label", None)]
            if len(outgoing_transitions) > 1:  # only decision points
                decision_points[len(decision_points)] = [t.label for t in outgoing_transitions]

        return decision_points

