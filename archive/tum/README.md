# Archived TUM RGB-D Dataset Infrastructure

## Overview
This directory contains historical loaders, replay sources, ROS 2 player nodes, and launch files used during early benchmark evaluation against the TUM RGB-D benchmark dataset (specifically `rgbd_dataset_freiburg1_desk`).

## When and Why It Existed
During initial pipeline development, TUM RGB-D sequences provided repeatable visual odometry and ground-truth SE(3) trajectory evaluation before physical camera hardware ingestion was deployed on the Jetson Orin Nano.

## Functionality & Status
- **Loaders**: `tum_loader.py` parsed text file indices (`rgb.txt`, `depth.txt`, `groundtruth.txt`).
- **Replay**: `tum_source.py` generated `FramePacket` instances; `tum_player.py` paced playback on ROS 2 simulated clock `/clock`.
- **Launchers**: `tum_scene_graph.launch.py`, `tum_slam.launch.py`, `tum_yolo.launch.py` executed replay scenarios.

## Why It Is Archived
The project architecture is now centered on the live Intel RealSense D455 sensor connected via USB 3.x directly to the Jetson Orin Nano, running RTAB-Map SLAM in real time. Embedding dataset-specific loaders, topic checks, and playback nodes inside the production runtime violated the principle of separation of concerns.

External benchmark evaluation code is isolated here and does not contaminate the active production runtime.
