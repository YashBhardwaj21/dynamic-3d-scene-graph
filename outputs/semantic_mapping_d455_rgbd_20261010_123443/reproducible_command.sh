#!/bin/bash
python tools/mapping/run_mapping_pipeline.py data/raw/d455_rgbd_20261010_123443 --output-root outputs --trajectory-file outputs/orbslam3_d455_trajectory/full_run/CameraTrajectory.txt --sample-step 10
