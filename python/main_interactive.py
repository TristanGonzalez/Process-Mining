import tempfile

import streamlit as st
from data import Data
from petri import Petri
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from PIL import Image


st.set_page_config(page_title="Interactive Process Mining", layout="wide")
st.title("Interactive Process Mining Pipeline")

# ---------------- Session state init ----------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "log" not in st.session_state:
    st.session_state.log = None
if "data_processor" not in st.session_state:
    st.session_state.data_processor = None
if "petri_processor" not in st.session_state:
    st.session_state.petri_processor = None
if "decision_points" not in st.session_state:
    st.session_state.decision_points = None
if "df" not in st.session_state:
    st.session_state.df = None
if "log_choice" not in st.session_state:
    st.session_state.log_choice = None

# ---------------- Helper to display Petri net ----------------
def display_petri_net(petri, im, fm):
    """
    Generate a PNG of the Petri net and show it in Streamlit.
    """
    gviz = pn_visualizer.apply(petri, im, fm, variant=pn_visualizer.Variants.WO_DECORATION)

    # Use a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
        pn_visualizer.save(gviz, tmp_file.name)
        image = Image.open(tmp_file.name)
        # Use the new parameter
        st.image(image, caption="Petri Net", use_container_width=True)

# ---------------- Step 1: Select predefined XES file ----------------
if st.session_state.step == 1:
    st.header("1️⃣ Select Event Log")
    log_options = {
        "Running Example": "data/running-example.xes",
        "Road Traffic Fine": "data/Road_Traffic_Fine_Management_Process.xes",
        "BPI Challenge 2017": "data/BPI Challenge 2017.xes"
    }
    log_choice = st.selectbox("Choose log:", options=list(log_options.keys()))
    if st.button("Load Event Log"):
        log_path = log_options[log_choice]
        data_processor = Data(path=log_path)
        log = data_processor.load_xes()
        st.session_state.log = log
        st.session_state.data_processor = data_processor
        st.session_state.log_choice = log_choice
        st.success(f"Loaded log `{log_choice}` with {len(log)} traces")
        st.session_state.step = 2

# ---------------- Step 2: Select discovery method ----------------
if st.session_state.step >= 2:
    st.header("2️⃣ Select Process Discovery Method")
    process_model_type = st.radio(
        "Choose method:",
        options=["heuristic", "inductive", "alpha"],
        index=0
    )
    if st.session_state.step == 2 and st.button("Confirm Method"):
        st.session_state.process_model_type = process_model_type
        st.session_state.step = 3

# ---------------- Step 3: Set parameters ----------------
if st.session_state.step >= 3:
    st.header("3️⃣ Set Tunable Parameters")

    if st.session_state.process_model_type == "heuristic":
        heuristic_dep = st.slider("Dependency threshold", 0.0, 1.0, 0.5)
        heuristic_and = st.slider("AND threshold", 0.0, 1.0, 0.65)
        heuristic_loop = st.slider("Loop-two threshold", 0.0, 1.0, 0.5)
        if st.button("Confirm Parameters"):
            st.session_state.heuristic_params = {
                "dependency_threshold": heuristic_dep,
                "and_threshold": heuristic_and,
                "loop_two_threshold": heuristic_loop
            }
            st.session_state.step = 4

    elif st.session_state.process_model_type == "inductive":
        noise_threshold = st.slider("Noise threshold", 0.0, 1.0, 0.0)
        multi_proc = st.checkbox("Enable multiprocess", True)
        disable_fallthroughs = st.checkbox("Disable fallthroughs", False)
        if st.button("Confirm Parameters"):
            st.session_state.inductive_params = {
                "noise_threshold": noise_threshold,
                "multi_processing": multi_proc,
                "disable_fallthroughs": disable_fallthroughs
            }
            st.session_state.step = 4

    else:  # alpha
        st.write("No parameters for alpha method")
        if st.button("Confirm Alpha Method"):
            st.session_state.step = 4

# ---------------- Step 4: Create Petri Net ----------------
if st.session_state.step >= 4:
    st.header("4️⃣ Create Petri Net")
    if st.session_state.petri_processor is None and st.button("Create Petri Net"):
        petri_processor = Petri(st.session_state.log)
        heuristic_params = st.session_state.get("heuristic_params", {})
        inductive_params = st.session_state.get("inductive_params", {})
        petri_processor.create_process_model(
            method=st.session_state.process_model_type,
            heuristic_params=heuristic_params,
            inductive_params=inductive_params
        )
        st.session_state.petri_processor = petri_processor
        st.success("Petri net created")
        st.session_state.step = 5

    if st.session_state.petri_processor is not None and st.button("View Petri Net"):
        display_petri_net(
            st.session_state.petri_processor.petri,
            st.session_state.petri_processor.im,
            st.session_state.petri_processor.fm
        )

# ---------------- Step 5: Find Decision Points ----------------
if st.session_state.step >= 5:
    st.header("5️⃣ Find Decision Points")
    if st.session_state.decision_points is None and st.button("Find Decision Points"):
        decision_points = st.session_state.petri_processor.find_decision_points()
        st.session_state.decision_points = decision_points
        st.success("Decision points found")
        st.session_state.step = 6

    if st.session_state.decision_points is not None:
        st.json(st.session_state.decision_points)

# ---------------- Step 6: Create DataFrame ----------------
if st.session_state.step >= 6:
    st.header("6️⃣ Create DataFrame")
    if st.session_state.df is None and st.button("Generate DataFrame"):
        df = st.session_state.data_processor.dataframe_from_dp(st.session_state.decision_points)
        st.session_state.df = df
        st.success(f"DataFrame created with shape {df.shape}")
        st.session_state.step = 7

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(15))

# ---------------- Step 7: Export CSV ----------------
if st.session_state.step >= 7:
    st.header("7️⃣ Export CSV")
    csv_filename = st.text_input(
        "CSV filename",
        value=f"data/{st.session_state.log_choice.replace(' ', '_')}.csv"
    )
    if st.button("Save CSV") and st.session_state.df is not None:
        st.session_state.df.to_csv(csv_filename, index=False, sep=";")
        st.success(f"DataFrame saved to {csv_filename}")