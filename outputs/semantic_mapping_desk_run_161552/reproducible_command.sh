#!/bin/bash
python tools/mapping/run_mapping_pipeline.py data/raw/desk_run_20261009_161552 --output-root outputs --run-id semantic_mapping_desk_run_161552 --trajectory-file outputs/orbslam3_desk_run_161552/CameraTrajectory.txt --sample-step 5
