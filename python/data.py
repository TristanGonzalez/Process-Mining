import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import pickle
import os
import re

class Data:
    def __init__(self, path):
        self.path = path
        if path == "data/xes/running-example.xes":
            self.kind = "running-example"
        elif path == "data/xes/Road_Traffic_Fine_Management_Process.xes":
            self.kind = "Road_Traffic_Fine_Management_Process"
        elif path == "data/xes/BPI Challenge 2017.xes":
            self.kind = "BPI Challenge 2017"
        else:
            raise ValueError("Dataset Unknown")
        self.cache_file = "data/pkl/" + self.kind + ".pkl"

    def load_xes(self):
        # Check for cached version
        if os.path.exists(self.cache_file):
            print(f"Loading cached log from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                self.log = pickle.load(f)
        else:
            print("Parsing XES log (this may take a while)...")
            self.log = xes_importer.apply(self.path)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.log, f)
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

            for idx, event in enumerate(trace):
                activity = event['concept:name']
                activity_counts[activity] += 1

                # Determine which decision point (if any) this activity belongs to
                dp_id = None
                for place_name, options in dp.items():
                    for opt in options:
                        if activity in opt:
                            dp_id = place_name
                            break
                    if dp_id:
                        break

                # Base row for every event
                row = {
                    "decision_point": dp_id,
                    "activity": activity,
                    "case_id": case_id
                }

                # Add event attributes dynamically
                for key, value in event.items():
                    key_clean = key.split(":")[0].lower() if ":" in key else key.lower()
                    if key_clean != "concept":
                        row[key_clean] = value

                # Add trace attributes
                for attr_key, attr_value in trace.attributes.items():
                    key_clean = attr_key.split(":")[0].lower() if ":" in attr_key else attr_key.lower()
                    if key_clean not in row and key_clean != "concept":
                        row[key_clean] = attr_value

                # Add activity counts
                for act, count in activity_counts.items():
                    row[f"count_{act}"] = count

                rows.append(row)

        df_decision_full = pd.DataFrame(rows)
        return df_decision_full


    def dataframe_from_dp_2(self, dp):
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

