# Archived Legacy D455 Windows -> WSL 2 TCP Transport Bridge

## Overview
This directory contains the legacy network bridge used to stream RealSense D455 sensor data across the Windows host and WSL 2 boundary:
- `d455_sender.py`: Windows host script running `pyrealsense2`, capturing RGB-D and IMU frames, and streaming them via TCP (port 5000) using Protocol v2 binary framing.
- `d455_bridge/`: ROS 2 package containing `d455_receiver.py`, which deserialized incoming TCP packets and republished them on `/camera/camera/*` ROS topics.
- Associated loopback and framing tests.

## When and Why It Existed
During initial development on Windows workstations, direct USB device passthrough for Intel RealSense cameras to WSL 2 was often unstable or unsupported without custom USB/IP drivers. The TCP loopback bridge allowed native Windows hardware capture while developing ROS 2 algorithms inside WSL 2.

## Functionality & Status
The implementation was verified and functional, including runtime depth scale propagation, clock mapping with drift compensation, and bounded queue drop policies.

## Why It Is Archived
The authoritative deployment platform is the **Jetson Orin Nano**, where the Intel RealSense D455 is connected directly over USB 3.x. The production pipeline ingests sensor data directly through standard Linux/ROS 2 camera drivers (`realsense2_camera`), making the TCP Windows bridge obsolete for deployment.
