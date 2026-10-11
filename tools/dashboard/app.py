"""Dynamic 3D Scene Graph Dashboard.

Unified local application connecting:
  A. RealSense D455 Live Stream & Graceful Dataset Recorder
  B. Dataset & Output Artifact Explorer with Paginated Inspection
  C. One-Click Geometric-Semantic Mapping Pipeline Runner
  D. Natural-Language Navigation & Direct /cmd_vel Kinematic Vehicle Simulator
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.dashboard.recording_manager import RecordingManager
from tools.dashboard.dataset_browser import DatasetBrowser
from tools.dashboard.pipeline_runner import PipelineRunner
from tools.dashboard.simulator import VehicleSimulator, SimulationResult


# Initialize session state singletons
if "recording_manager" not in st.session_state:
    st.session_state.recording_manager = RecordingManager(REPO_ROOT)
if "dataset_browser" not in st.session_state:
    st.session_state.dataset_browser = DatasetBrowser(REPO_ROOT)
if "pipeline_runner" not in st.session_state:
    st.session_state.pipeline_runner = PipelineRunner(REPO_ROOT)
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "sim_step_idx" not in st.session_state:
    st.session_state.sim_step_idx = 0


# --- App Page Configuration ---
st.set_page_config(
    page_title="Dynamic 3D Scene Graph",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling for Clean Aesthetics ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 15px;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #0f172a;
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .warning-box {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 10px 14px;
        border-radius: 6px;
        color: #facc15;
        font-size: 0.88rem;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown('<div class="main-header">Dynamic 3D Scene Graph</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Recording · Mapping · Navigation simulation</div>', unsafe_allow_html=True)
st.markdown('<div class="badge-tag">Local application · RealSense D455 · 4-Motor RC Car</div>', unsafe_allow_html=True)


# --- Sidebar Navigation & System Diagnostics ---
rec_mgr: RecordingManager = st.session_state.recording_manager
browser: DatasetBrowser = st.session_state.dataset_browser
pipe_runner: PipelineRunner = st.session_state.pipeline_runner

with st.sidebar:
    st.header("Navigation")
    nav_tab = st.radio(
        "Module Selection",
        [
            "A. Record a new dataset",
            "B. Select and inspect a dataset",
            "C. Process the selected dataset",
            "D. Command and vehicle simulation",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Hardware Diagnostics")
    rs_ok, rs_msg = rec_mgr.check_realsense_available()
    if rs_ok:
        st.success(f"🎥 {rs_msg}")
    else:
        st.warning(f"⚠️ {rs_msg}")

    orb_status = pipe_runner.check_orbslam3_status()
    if orb_status["is_ready"]:
        st.success(f"🗺️ ORB-SLAM3 Ready")
    else:
        st.info(f"ℹ️ SLAM Trajectory Mode (Uses TUM files)")

    st.markdown("---")
    st.caption(f"Perception Workspace: `{REPO_ROOT.name}`")


# ==============================================================================
# SECTION A: RECORD A NEW DATASET
# ==============================================================================
if nav_tab == "A. Record a new dataset":
    st.subheader("A. Record a New RealSense D455 Dataset")
    st.caption("Captures synchronized RGB-D sequence in TUM format with hardware timestamps.")

    st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Groundtruth Calibration Notice:</b> The recorded <code>groundtruth.txt</code> file contains synthetic
        identity placeholder poses for pipeline indexing. It does not represent measured external optical ground truth.
    </div>
    """, unsafe_allow_html=True)

    col_setup, col_preview = st.columns([1, 1.2])

    with col_setup:
        st.markdown("#### Recording Setup")
        rec_status = rec_mgr.get_status()

        default_name = f"d455_rgbd_{time.strftime('%Y%m%d_%H%M%S')}"
        ds_name_input = st.text_input(
            "Dataset Name",
            value=rec_status["dataset_name"] or default_name,
            disabled=rec_status["is_recording"],
            help="Unique name for the directory under data/raw/",
        )
        enable_imu = st.checkbox("Record 6-Axis IMU (Accel + Gyro)", value=False, disabled=rec_status["is_recording"])

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔴 Start Recording", disabled=rec_status["is_recording"], use_container_width=True):
                success = rec_mgr.start_recording(
                    dataset_name=ds_name_input,
                    enable_imu=enable_imu,
                )
                if success:
                    st.success(f"Recording started: {ds_name_input}")
                    st.rerun()
                else:
                    st.error(rec_mgr.error_message or "Failed to start camera.")

        with col_b2:
            if st.button("⏹️ Stop Recording", disabled=not rec_status["is_recording"], use_container_width=True):
                with st.spinner("Gracefully stopping recorder & flushing disk queue..."):
                    rec_mgr.stop_recording()
                st.success("Recording finalized and flushed to disk!")
                st.rerun()

        st.markdown("#### Live Telemetry")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Status", rec_status["status"])
        with m_col2:
            st.metric("Elapsed Time", f"{rec_status['elapsed_seconds']}s")
        with m_col3:
            st.metric("Frames Recorded", rec_status["frame_count"])

        if rec_status["save_location"]:
            st.text_input("Save Location", value=rec_status["save_location"], disabled=True)

        if rec_status["error_message"]:
            st.error(rec_status["error_message"])

    with col_preview:
        st.markdown("#### RGB & Depth Camera Preview")
        preview_bytes = rec_mgr.get_preview_image_bytes()
        if preview_bytes:
            st.image(preview_bytes, caption="Live D455 Color (Left) + Aligned Depth Colormap (Right)", use_column_width=True)
        else:
            # Standby placeholder
            st.info("No active preview stream. When recording is initiated, synchronized frames appear here.")
            placeholder_img = np.zeros((240, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "D455 Stream Standby", (180, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
            st.image(placeholder_img, use_column_width=True)

        if rec_status["is_recording"]:
            time.sleep(0.8)
            st.rerun()


# ==============================================================================
# SECTION B: SELECT AND INSPECT A DATASET
# ==============================================================================
elif nav_tab == "B. Select and inspect a dataset":
    st.subheader("B. Select & Inspect Dataset Artifacts")
    st.caption("Browse all files, check validation metrics, and preview images, maps, and configs.")

    source_type = st.radio("Repository Directory", ["Raw Recordings (data/raw/)", "Mapping Outputs (outputs/)"], horizontal=True)
    is_output = (source_type == "Mapping Outputs (outputs/)")

    available_folders = browser.list_mapping_runs() if is_output else browser.list_raw_datasets()
    if not available_folders:
        st.warning(f"No directories found in {'outputs/' if is_output else 'data/raw/'}.")
    else:
        col_sel, col_info = st.columns([1.5, 1])
        with col_sel:
            selected_folder = st.selectbox("Select Target Folder", available_folders)
        with col_info:
            summary = browser.get_dataset_summary(selected_folder, is_output=is_output)
            st.metric("Total Files", summary.get("total_files", 0))

        if selected_folder:
            target_path = (browser.outputs_root if is_output else browser.raw_root) / selected_folder
            files = browser.get_recursive_file_list(target_path)

            tab_files, tab_meta = st.tabs(["📁 File Tree & Viewer", "📊 Dataset Summary"])

            with tab_meta:
                if summary.get("validation"):
                    val = summary["validation"]
                    st.markdown("##### Recording Validation Report")
                    v1, v2, v3 = st.columns(3)
                    v1.metric("Status", val.get("status", "UNKNOWN"))
                    v2.metric("Frame Pairs", val.get("counts", {}).get("associated_pairs", 0))
                    v3.metric("Valid Depth %", f"{val.get('quality', {}).get('mean_valid_depth_fraction', 0)*100:.1f}%")
                    st.json(val)
                elif summary.get("audit_summary"):
                    st.markdown("##### Mapping Audit Summary")
                    st.json(summary["audit_summary"])
                elif summary.get("meta"):
                    st.json(summary["meta"])
                else:
                    st.info("No structured validation metadata found in this folder.")

            with tab_files:
                col_tbl, col_view = st.columns([1, 1.3])

                with col_tbl:
                    st.markdown("##### File Explorer")
                    # Category filter
                    cats = sorted(list(set(f["category"] for f in files)))
                    cat_filter = st.multiselect("Filter File Type", cats, default=[])
                    filtered_files = [f for f in files if (not cat_filter or f["category"] in cat_filter)]

                    # Pagination
                    page_size = 15
                    total_pages = max(1, math.ceil(len(filtered_files) / page_size))
                    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
                    start_idx = (page - 1) * page_size
                    page_items = filtered_files[start_idx: start_idx + page_size]

                    # Table
                    df_display = pd.DataFrame([{
                        "Path": f["relative_path"],
                        "Type": f["category"],
                        "Size": f["size_str"],
                    } for f in page_items])
                    st.dataframe(df_display, use_container_width=True, height=380)

                    # File selector for viewing
                    file_options = [f["relative_path"] for f in page_items]
                    selected_rel_file = st.selectbox("Inspect File", file_options if file_options else ["None"])

                with col_view:
                    st.markdown("##### File Previewer")
                    if selected_rel_file and selected_rel_file != "None":
                        file_abs_path = target_path / selected_rel_file
                        preview_data = browser.read_file_preview(file_abs_path)

                        st.caption(f"**Path:** `{selected_rel_file}` | **Category:** {preview_data.get('type')}")

                        ptype = preview_data.get("type")
                        if ptype == "image":
                            st.image(preview_data["path"], use_column_width=True)
                        elif ptype == "numpy_image":
                            st.image(preview_data["array"], caption=f"ROS Occupancy Grid ({preview_data['shape']})", use_column_width=True)
                        elif ptype == "video":
                            st.video(preview_data["path"])
                        elif ptype == "json":
                            st.json(preview_data["data"])
                        elif ptype == "jsonl":
                            st.json(preview_data["data"])
                            st.caption(f"Displaying top 50 of {preview_data.get('total_lines')} records.")
                        elif ptype == "yaml":
                            st.json(preview_data["data"])
                        elif ptype == "text":
                            st.code(preview_data["content"], language="text")
                        elif ptype == "ply_info":
                            st.info(f"3D Point Cloud Mesh: `{selected_rel_file}` ({preview_data['size_str']})")
                            with open(file_abs_path, "rb") as bf:
                                st.download_button("⬇️ Download .PLY File", bf.read(), file_name=file_abs_path.name)
                        elif ptype == "binary":
                            st.info(f"Binary file ({preview_data['size_str']})")
                            with open(file_abs_path, "rb") as bf:
                                st.download_button(f"⬇️ Download {file_abs_path.name}", bf.read(), file_name=file_abs_path.name)
                        elif ptype == "error":
                            st.error(preview_data["message"])


# ==============================================================================
# SECTION C: PROCESS THE SELECTED DATASET
# ==============================================================================
elif nav_tab == "C. Process the selected dataset":
    st.subheader("C. Process Recording (Offline Mapping Pipeline)")
    st.caption("Runs tools/mapping/run_mapping_pipeline.py with validated parameters to generate Nav2 costmaps & 3D scene graphs.")

    raw_datasets = browser.list_raw_datasets()
    if not raw_datasets:
        st.warning("No recordings discovered in data/raw/. Please record a sequence in Section A first.")
    else:
        col_cfg, col_logs = st.columns([1, 1.2])

        with col_cfg:
            st.markdown("#### Execution Parameters")
            sel_dataset = st.selectbox("Selected Dataset", raw_datasets)

            default_run_id = f"semantic_mapping_{sel_dataset}"
            run_id_input = st.text_input(
                "Unique Run ID",
                value=default_run_id,
                help="Unique output folder name created in outputs/",
            )

            stride = st.slider("Sampling Stride (Frames)", min_value=1, max_value=30, value=10, step=1,
                               help="Step stride across synchronized frame pairs.")

            # Optional trajectory override
            traj_options = ["Automatic (ORB-SLAM3 Headless Ingest)", "Custom Trajectory File"]
            traj_mode = st.radio("Camera Trajectory Mode", traj_options, index=0)
            custom_traj_path = None
            if traj_mode == "Custom Trajectory File":
                custom_traj_path = st.text_input("Trajectory Path (.txt)", value=f"outputs/orbslam3_{sel_dataset}/CameraTrajectory.txt")

            pipe_status = pipe_runner.get_status()

            if st.button("🚀 Process Recording", disabled=pipe_status["is_running"], use_container_width=True):
                ok = pipe_runner.start_pipeline(
                    dataset_name=sel_dataset,
                    sample_step=stride,
                    run_id=run_id_input,
                    trajectory_file=custom_traj_path if traj_mode == "Custom Trajectory File" else None,
                )
                if ok:
                    st.success(f"Mapping pipeline initiated for {sel_dataset}!")
                    st.rerun()
                else:
                    st.error("Failed to launch mapping pipeline.")

            st.markdown("#### Run Status")
            st.metric("Pipeline State", pipe_status["status"])
            st.metric("Execution Time", f"{pipe_status['elapsed_seconds']}s")
            if pipe_status["output_dir"]:
                st.caption(f"Output: `{pipe_status['output_dir']}`")

        with col_logs:
            st.markdown("#### Real-Time Pipeline Logs")
            recent_logs = pipe_status.get("recent_logs", [])
            log_text = "\n".join(recent_logs) if recent_logs else "No active pipeline logs."
            st.text_area("Stdout / Stderr Stream", value=log_text, height=340, disabled=True)

            if pipe_status["is_running"]:
                time.sleep(1.0)
                st.rerun()

        # Artifacts discovery upon completion
        artifacts = pipe_runner.get_discovered_artifacts()
        if artifacts:
            st.markdown("---")
            st.markdown("#### Generated Pipeline Deliverables")
            art_cols = st.columns(3)

            if "vis_semantic_floor_plan" in artifacts:
                with art_cols[0]:
                    st.image(artifacts["vis_semantic_floor_plan"], caption="Semantic Floor Plan", use_column_width=True)
            if "vis_mapping_visualization" in artifacts:
                with art_cols[1]:
                    st.image(artifacts["vis_mapping_visualization"], caption="3D Multi-View Scene Graph", use_column_width=True)
            if "vis_planned_goal_route" in artifacts:
                with art_cols[2]:
                    st.image(artifacts["vis_planned_goal_route"], caption="Sample A* Nav2 Route", use_column_width=True)


# ==============================================================================
# SECTION D: COMMAND AND VEHICLE SIMULATION
# ==============================================================================
elif nav_tab == "D. Command and vehicle simulation":
    st.subheader("D. Command & Vehicle Kinematic Simulator")
    st.caption("Interactive Plotly top-down simulation of differential pure-pursuit steering or direct /cmd_vel sequences on verified costmaps.")

    st.markdown("""
    <div class="warning-box">
        🛡️ <b>Kinematic Simulation Gate:</b> This simulation integrates vehicle unicycle kinematics against the costmap.
        Physical motors remain completely UNPOWERED (no <code>--enable-motion</code>).
        Real wheel slip and turning resistance are not measured without wheel encoders.
    </div>
    """, unsafe_allow_html=True)

    # 1. Map Run Selection
    output_runs = browser.list_mapping_runs()
    if not output_runs:
        st.warning("No mapping outputs discovered in outputs/. Please execute Section C first.")
    else:
        sel_run = st.selectbox("Select Mapped Environment Run", output_runs, index=0)
        run_dir = browser.outputs_root / sel_run

        try:
            simulator = VehicleSimulator(run_dir)
        except Exception as e:
            st.error(f"Failed to load map data from {sel_run}: {e}")
            simulator = None

        if simulator:
            col_ctrl, col_plot = st.columns([1, 1.5])

            with col_ctrl:
                mode_choice = st.radio("Simulation Command Mode", ["Mode 1: Navigation Goal", "Mode 2: Direct Velocity Sequence (/cmd_vel)"])

                if mode_choice == "Mode 1: Navigation Goal":
                    st.markdown("##### Goal Instruction")

                    # Dropdown of confirmed entities
                    conf_entities = simulator.get_confirmed_entity_names()
                    dropdown_pick = st.selectbox("Target Object Dropdown", ["None"] + conf_entities, index=1 if conf_entities else 0)

                    # Prompt input
                    default_goal = f"Go to {dropdown_pick.split(' ')[0]} and stop 0.5 metres away" if (dropdown_pick != "None") else "Go to chair_001 and stop 0.5 metres away"
                    goal_input = st.text_input("Natural Language Goal", value=default_goal)

                    start_x = st.number_input("Start Pose X (m)", value=0.0, step=0.1)
                    start_y = st.number_input("Start Pose Y (m)", value=0.0, step=0.1)
                    start_yaw = st.number_input("Start Pose Yaw (deg)", value=0.0, step=15.0)

                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        if st.button("🗺️ Plan & Simulate", use_container_width=True):
                            with st.spinner("Resolving goal and simulating pure pursuit..."):
                                res = simulator.simulate_navigation_goal(
                                    goal_instruction=goal_input,
                                    start_pose=(start_x, start_y, math.radians(start_yaw)),
                                )
                                st.session_state.sim_result = res
                                st.session_state.sim_step_idx = len(res.states) - 1 if res.states else 0
                    with col_p2:
                        if st.button("🔄 Reset", use_container_width=True):
                            st.session_state.sim_result = None
                            st.session_state.sim_step_idx = 0
                            st.rerun()

                else:  # Mode 2: Direct Velocity Commands
                    st.markdown("##### Direct Velocity Commands (/cmd_vel)")
                    sample_json = json.dumps([
                        {"linear_x": 0.20, "angular_z": 0.00, "duration_s": 2.0},
                        {"linear_x": 0.15, "angular_z": 0.40, "duration_s": 3.0},
                        {"linear_x": 0.00, "angular_z": 0.00, "duration_s": 0.5},
                    ], indent=2)

                    cmd_json_str = st.text_area("Command Sequence (JSON)", value=sample_json, height=180)
                    start_x = st.number_input("Start Pose X (m)", value=0.0, step=0.1, key="cmd_sx")
                    start_y = st.number_input("Start Pose Y (m)", value=0.0, step=0.1, key="cmd_sy")
                    start_yaw = st.number_input("Start Pose Yaw (deg)", value=0.0, step=15.0, key="cmd_syaw")

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        if st.button("⚡ Simulate Velocity Sequence", use_container_width=True):
                            try:
                                cmd_seq = json.loads(cmd_json_str)
                                with st.spinner("Simulating vehicle kinematics..."):
                                    res = simulator.simulate_velocity_commands(
                                        command_sequence=cmd_seq,
                                        start_pose=(start_x, start_y, math.radians(start_yaw)),
                                    )
                                    st.session_state.sim_result = res
                                    st.session_state.sim_step_idx = len(res.states) - 1 if res.states else 0
                            except Exception as e:
                                st.error(f"Invalid JSON command format: {e}")

                    with col_c2:
                        if st.button("🔄 Reset", use_container_width=True, key="cmd_reset"):
                            st.session_state.sim_result = None
                            st.session_state.sim_step_idx = 0
                            st.rerun()

                # Simulation Result Telemetry
                cur_res: Optional[SimulationResult] = st.session_state.sim_result
                if cur_res:
                    st.markdown("---")
                    st.markdown("##### Simulation Telemetry")
                    if cur_res.has_collision:
                        st.error(f"💥 COLLISION DETECTED at t={cur_res.collision_time_s:.2f}s with costmap obstacle!")
                    elif cur_res.success:
                        st.success(f"✅ {cur_res.message}")
                    else:
                        st.warning(cur_res.message)

                    t_col1, t_col2 = st.columns(2)
                    t_col1.metric("Trajectory Steps", len(cur_res.states))
                    duration_total = cur_res.states[-1].time_s if cur_res.states else 0.0
                    t_col2.metric("Total Duration", f"{duration_total:.2f}s")

                    # Scrubber slider
                    if cur_res.states:
                        st.session_state.sim_step_idx = st.slider(
                            "Scrub Simulation Timeline",
                            min_value=0,
                            max_value=len(cur_res.states) - 1,
                            value=min(st.session_state.sim_step_idx, len(cur_res.states) - 1),
                            step=1,
                            format="Step %d",
                        )
                        s_state = cur_res.states[st.session_state.sim_step_idx]
                        st.caption(f"Time: {s_state.time_s:.2f}s | Pose: ({s_state.x:.2f}m, {s_state.y:.2f}m) | Heading: {math.degrees(s_state.yaw_rad):.1f}° | Speed: {s_state.linear_v:.2f}m/s")

            with col_plot:
                st.markdown("#### Top-Down Occupancy Map & Vehicle State")
                cur_res: Optional[SimulationResult] = st.session_state.sim_result
                step_idx = st.session_state.sim_step_idx if cur_res else None
                fig = simulator.render_topdown_plotly(sim_result=cur_res, current_step_idx=step_idx)
                st.plotly_chart(fig, use_container_width=True)
