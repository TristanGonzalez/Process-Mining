import yaml
from data import Data
from petri import Petri
from decision_mining_ml import DecisionMiningML
import json

import os
# print(os.getcwd())

# Load Yaml Config
print("1. Opening config.yaml... \n")


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

log_path = config.get("log_path")
process_model_type = config.get("process_model_type", "heuristic")
heuristic_params = config.get("heuristic", {})
inductive_params = config.get("inductive", {})



# Load Log File
print("2. Opening Logs... \n")

data_processor = Data(path=log_path)
log = data_processor.load_xes()



# Creating Petri Net
print("3. Creating Petri Net... \n")

petri_processor = Petri(log)
petri_processor.create_process_model(
    method=process_model_type,
    heuristic_params=heuristic_params,
    inductive_params=inductive_params
)
petri_processor.view_process_model()


# Finding Decision Points
print("4. Finding Decision Points... \n")
decision_points = petri_processor.find_decision_points()


for i, decision in enumerate(decision_points):
    print(f"Decision Point {str(i+1)}: {decision}  -> {str(decision_points[decision])} \n")



# Creating Data Frame
print("5. Creating Data Frame... \n")


df = data_processor.dataframe_from_dp(decision_points)

print("Dataframe contains: ", len(df), " rows")

path = f"data/decision_output/{data_processor.kind}.csv"


# Saving Data Frame
print("6. Saving Data Frame... \n")

output_csv = config.get("output_csv", path)
df.to_csv(output_csv, index=False, sep=";")


# Saving Data Frame
print("7. Decision Mining... \n")

dm = DecisionMiningML(path)  # YAML parameters are loaded automatically

res = dm.train_for_all(test_size=0.3)

out = {}
for k, v in res.items():
    if isinstance(v, dict) and 'error' in v:
        out[k] = {'error': v['error']}
    else:
        out[k] = {
            'n_samples': v.get('n_samples'),
            'eval_default': v.get('eval_default'),
            'eval_pruned': v.get('eval_pruned'),
            'viz_default': v.get('viz_default'),
            'rules_default': v.get('rules_default')
        }

print(json.dumps(out, indent=2))
