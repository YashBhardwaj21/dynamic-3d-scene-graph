#!/usr/8in/env python3
"""Generates a timestamp synchronization quality report for the TUM dataset (Stage 1 QA)."""

import argparse
from pathlib import Path
import numpy as np

from scene_graph.data.tum_loader import TUMLoader
from scene_graph.data.synchronization import associate


def print_association_stats(primary_ts, secondary_ts, matches, name):
    print(f"\n--- {name} Synchronization ---")
    if not matches:
        print("No matches found.")
        return
        
    print(f"Primary entries:   {len(primary_ts)}")
    print(f"Secondary entries: {len(secondary_ts)}")
    print(f"Total Matches:     {len(matches)} ({(len(matches)/len(primary_ts))*100:.1f}%)")
    
    deltas = []
    for p_idx, s_idx in matches:
        deltas.append(abs(primary_ts[p_idx] - secondary_ts[s_idx]))
        
    deltas = np.array(deltas)
    print(f"Max delta-t:       {np.max(deltas):.6f} sec")
    print(f"Mean delta-t:      {np.mean(deltas):.6f} sec")
    print(f"Median delta-t:    {np.median(deltas):.6f} sec")
    print(f"95th percentile:   {np.percentile(deltas, 95):.6f} sec")


def main():
    parser = argparse.ArgumentParser(description="TUM Synchronization QA")
    parser.add_argument("--sequence_dir", type=str, default="rgbd_dataset_freiburg1_desk")
    parser.add_argument("--max_dt", type=float, default=0.02)
    args = parser.parse_args()
    
    seq_dir = Path(args.sequence_dir)
    if not seq_dir.is_dir():
        print(f"Dataset directory not found: {seq_dir}")
        return
        
    loader = TUMLoader(seq_dir)
    rgb = loader.load_rgb()
    depth = loader.load_depth()
    pose = loader.load_groundtruth()
    
    rgb_ts = [e.timestamp for e in rgb]
    depth_ts = [e.timestamp for e in depth]
    pose_ts = [e.timestamp for e in pose]
    
    print(f"Loaded {len(rgb_ts)} RGB, {len(depth_ts)} Depth, {len(pose_ts)} Pose entries.")
    
    rgb_depth_matches = associate(rgb_ts, depth_ts, max_dt=args.max_dt)
    print_association_stats(rgb_ts, depth_ts, rgb_depth_matches, "RGB <-> Depth")
    
    rgb_pose_matches = associate(rgb_ts, pose_ts, max_dt=args.max_dt)
    print_association_stats(rgb_ts, pose_ts, rgb_pose_matches, "RGB <-> Pose")


if __name__ == "__main__":
    main()
