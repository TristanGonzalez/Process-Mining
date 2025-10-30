import yaml
from data import Data
from petri import Petri

# ---------------- Load YAML -----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

log_path = config.get("log_path")
process_model_type = config.get("process_model_type", "heuristic")
heuristic_params = config.get("heuristic", {})
inductive_params = config.get("inductive", {})

# ---------------- Load log -----------------
data_processor = Data(path=log_path)
log = data_processor.load_xes()




# ---------------- Create Petri net -----------------
petri_processor = Petri(log)
petri_processor.create_process_model(
    method=process_model_type,
    heuristic_params=heuristic_params,
    inductive_params=inductive_params
)
petri_processor.view_process_model()



decision_points = petri_processor.find_decision_points()



for decision in decision_points:
    print(decision + ": " + str(decision_points[decision]), "\n")



# # ---------------- Create DataFrame -----------------
# df = data_processor.dataframe_from_dp_2(decision_points)
#
# # ---------------- Save CSV -----------------
# output_csv = config.get("output_csv", f"data/decision_output/{data_processor.kind}.csv")
# df.to_csv(output_csv, index=False, sep=";")
# print(f"DataFrame saved to: {output_csv}")
# print(df.head())