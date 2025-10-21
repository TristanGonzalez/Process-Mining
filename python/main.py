import os
from data import Data
import petri



path1 = "data/running-example.xes"
path2 = "data/Road_Traffic_Fine_Management_Process.xes"
path3 = "data/BPI Challenge 2017.xes"

path = path1

data_processor = Data(path=path)

log = data_processor.load_xes()
petri_processor = petri.Petri(log)
petri_processor.create_process_model("heuristic")
petri_processor.view_process_model()
decision_points = petri_processor.find_decision_points()

df = data_processor.dataframe_from_dp(decision_points)

df.to_csv(f"{path}.csv", index=False, sep=";")

print(df)
