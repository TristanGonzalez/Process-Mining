import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import re

class Data:
    def __init__(self, path):
        self.path = path
        if path == "data/running-example.xes":
            self.kind = "running-example"
        elif path == "data/Road_Traffic_Fine_Management_Process.xes":
            self.kind = "Road_Traffic_Fine_Management_Process"
        elif path == "data/BPI Challenge 2017.xes":
            self.kind = "BPI Challenge 2017"
        else:
            raise ValueError("Dataset Unknown")


    def load_xes(self):
        self.log = xes_importer.apply(self.path)
        return self.log

    def logs_to_df(self):
        df = pm4py.convert_to_dataframe(self.log)
        return df


    def dataframe_from_dp(self, dp):
        rows = []

        # Get all distinct activities
        activities = set()
        for trace in self.log:
            for event in trace:
                activities.add(event['concept:name'])

        for trace in self.log:
            case_id = trace.attributes.get('concept:name') or trace.attributes.get('case:concept:name')

            # Initialize activity counts
            activity_counts = {act: 0 for act in activities}

            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] += 1

                # Check decision points
                for dp_id, options in dp.items():
                    if {activity} in options:
                        # Base row
                        row = {
                            "decision_point": dp_id,
                            "activity": activity,
                        }

                        # Add all event attributes dynamically
                        for key, value in event.items():
                            if key not in row:
                                match = re.match(r'^.+?(?=:)', key)
                                if match:

                                    key = match.group(0).lower()
                                    if key != "concept":
                                        row[key] = value
                                else:
                                    key = key.lower()
                                    row[key] = value

                        for item in trace:
                            for key, value in item.items():

                                if key not in row:
                                    match = re.match(r'^.+?(?=:)', key)
                                    if match:
                                        key = match.group(0).lower()
                                    else:
                                        key = key.lower()


                                if key != "concept":
                                    row[key] = value



                        # Add activity counts

                        for key, value in activity_counts.items():
                            row[key] = value


                        rows.append(row)

        df_decision_full = pd.DataFrame(rows)
        return df_decision_full

        # if self.kind == "running-example":
        #     return self.dataframe_from_dp_running_examples(dp)
        # elif self.kind == "Road_Traffic_Fine_Management_Process":
        #     return self.dataframe_from_dp_road_traffic(dp)
        # else:
        #     return self.dataframe_from_dp_BPI_challenge(dp)

    import pandas as pd

    def dataframe_from_dp_running_examples(self, dp):
        """
        Generic function for running-examples logs.
        Tracks activity counts and includes all event & case attributes.
        """
        rows = []

        # Get all distinct activities
        activities = set()
        for trace in self.log:
            for event in trace:
                activities.add(event['concept:name'])

        for trace in self.log:
            case_id = trace.attributes.get('concept:name') or trace.attributes.get('case:concept:name')

            # Initialize activity counts
            activity_counts = {act: 0 for act in activities}

            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] += 1

                # Check decision points
                for dp_id, options in dp.items():
                    if {activity} in options:
                        # Base row
                        row = {
                            "decision_point": dp_id,
                            "activity": activity,
                        }

                        # Add all event attributes dynamically
                        for key, value in event.items():
                            if key not in row:
                                match = re.match(r'^.+?(?=:)', key)
                                if match:
                                    key = match.group(0).lower()
                                    row[key] = value
                                else:
                                    key = key.lower()
                                    row[key] = value

                        # # Add all case attributes dynamically
                        # for key, value in trace.attributes.items():
                        #     if key not in row:
                        #         row[key] = value

                        # Add activity counts
                        row.update(activity_counts)

                        rows.append(row)

        df_decision_full = pd.DataFrame(rows)
        return df_decision_full

    def dataframe_from_dp_road_traffic(self, dp):
        """
        Road Traffic Fine log function.
        Tracks activity counts, decision points, and all attributes.
        """
        rows = []

        # Get all distinct activities
        activities = set()
        for trace in self.log:
            for event in trace:
                activities.add(event['concept:name'])

        for trace in self.log:
            case_id = trace.attributes.get('concept:name') or trace.attributes.get('case:concept:name')

            # Initialize activity counts
            activity_counts = {act: 0 for act in activities}

            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] += 1

                # Check decision points
                for dp_id, options in dp.items():
                    if {activity} in options:
                        # Base row
                        row = {
                            "decision_point": dp_id,
                            "activity": activity,
                        }

                        # Add all event attributes dynamically
                        for key, value in event.items():
                            if key not in row:
                                match = re.match(r'^.+?(?=:)', key)
                                if match:
                                    key = match.group(0).lower()
                                    row[key] = value
                                else:
                                    key = key.lower()
                                    row[key] = value

                        # Add activity counts
                        row.update(activity_counts)

                        rows.append(row)

        df_decision_full = pd.DataFrame(rows)
        return df_decision_full

    def dataframe_from_dp_BPI_challenge(self, dp):
        """
        BPI Challenge 2017 log function.
        Tracks activity counts, decision points, and all attributes.
        """
        rows = []

        # Get all distinct activities
        activities = set()
        for trace in self.log:
            for event in trace:
                activities.add(event['concept:name'])

        for trace in self.log:
            case_id = trace.attributes.get('concept:name') or trace.attributes.get('case:concept:name')

            # Initialize activity counts
            activity_counts = {act: 0 for act in activities}

            for event in trace:
                activity = event['concept:name']
                activity_counts[activity] += 1

                # Check decision points
                for dp_id, options in dp.items():
                    if {activity} in options:
                        # Base row
                        row = {
                            "decision_point": dp_id,
                            "activity": activity,
                        }

                        # Add all event attributes dynamically
                        for key, value in event.items():
                            if key not in row:
                                match = re.match(r'^.+?(?=:)', key)
                                if match:
                                    key = match.group(0).lower()
                                    row[key] = value
                                else:
                                    key = key.lower()
                                    row[key] = value

                        # Add activity counts

                        for key, value in activity_counts:
                            row[key] = value


                        rows.append(row)

        df_decision_full = pd.DataFrame(rows)
        return df_decision_full