#!/usr/bin/env python3
"""Run YOLOE detector on all frames and save the observation stream (Substage 2.5)."""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from scene_graph.data.tum_loader import TUMLoader
from scene_graph.data.frame_packet import build_frame_packets
from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.perception.yoloe_detector import YOLOEDetector


def convert_for_json(obj):
    """Helper to convert numpy arrays and special types to JSON serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Prepare fixed observation stream.")
    parser.add_argument("--sequence_dir", type=str, default="rgbd_dataset_freiburg1_desk")
    parser.add_argument("--out_dir", type=str, default="data/processed/tum_fr1_desk")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--model", type=str, default="models/yoloe/yoloe-26m-seg.pt")
    parser.add_argument("--mode", type=str, choices=["controlled", "discovery"], default="controlled")
    args = parser.parse_args()
    
    seq_dir = Path(args.sequence_dir)
    if not seq_dir.is_dir():
        print(f"Dataset directory not found: {seq_dir}")
        return
        
    out_dir = Path(args.out_dir)
    
    # Vocabulary Definitions
    vocabularies = {
        "controlled": {
            "name": "v4_11",
            "classes": {
                "monitor", "computer", "keyboard", "mouse", "telephone",
                "book", "cup", "pen", "paper", "desk", "table"
            }
        },
        "discovery": {
            "name": "expanded",
            "classes": {
                "person", "chair", "table", "desk", "monitor", "laptop",
                "computer", "keyboard", "mouse", "telephone", "phone",
                "book", "notebook", "paper", "pen", "pencil", "cup",
                "bottle", "can", "bag", "backpack", "headphones",
                "remote", "clock", "lamp", "printer", "box", "container"
            }
        }
    }
    
    current_vocab = vocabularies[args.mode]
    obs_dir_name = "observations_controlled" if args.mode == "controlled" else "observations_discovery"
    obs_dir = out_dir / obs_dir_name
    obs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Camera and Loader
    intrinsics = CameraIntrinsics()
    depth_model = DepthModel()
    
    loader = TUMLoader(seq_dir)
    
    # 2. Build packets for the ENTIRE sequence
    print("Building frame packets for the ENTIRE sequence...")
    # Passing None for start_frame/end_frame to process all
    packets = build_frame_packets(seq_dir, start_frame=None, end_frame=None)
    print(f"Generated {len(packets)} packets.")
    
    # 3. Setup Detector
    print(f"Initializing YOLOEDetector with {args.model}...")
    try:
        detector = YOLOEDetector(
            model_path=args.model,
            confidence_threshold=0.40,
            allowed_classes=current_vocab["classes"],
            intrinsics=intrinsics,
            depth_model=depth_model
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install ultralytics: pip install ultralytics")
        return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    # 4. Process frames and save chunks
    print("Running detection and chunking...")
    
    chunk_idx = 0
    current_chunk_frames = []
    chunk_files = []
    
    start_time = time.time()
    
    def save_chunk(frames, c_idx):
        if not frames:
            return
        chunk_name = f"chunk_{c_idx:04d}.json"
        chunk_path = obs_dir / chunk_name
        print(f"Saving {len(frames)} frames to {chunk_path}...")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump({"frames": frames}, f, indent=2)
        chunk_files.append(chunk_name)
    
    for i, packet in enumerate(packets):
        obs_list = detector.detect(packet)
        
        # Serialize observations
        obs_dicts = []
        for obs in obs_list:
            obs_dict = {
                "obs_id": obs.obs_id,
                "frame_index": obs.frame_index,
                "timestamp": obs.timestamp,
                "class_name": obs.class_name,
                "confidence": obs.confidence,
                "bbox_xyxy": convert_for_json(obs.bbox_xyxy),
                "mask_rle": obs.mask_rle,
                "centroid_camera": convert_for_json(obs.centroid_camera),
                "centroid_world": convert_for_json(obs.centroid_world),
                "bbox_min_world": convert_for_json(obs.bbox_min_world),
                "bbox_max_world": convert_for_json(obs.bbox_max_world),
                "valid_point_count": obs.valid_point_count,
                "point_cloud_ref": obs.point_cloud_ref
            }
            obs_dicts.append(obs_dict)
            
        frame_data = {
            "frame_index": packet.frame_index,
            "timestamp": packet.timestamp,
            "observations": obs_dicts
        }
        current_chunk_frames.append(frame_data)
        
        # Save chunk if it reached the size
        if len(current_chunk_frames) >= args.chunk_size:
            save_chunk(current_chunk_frames, chunk_idx)
            chunk_idx += 1
            current_chunk_frames = []
            
        if (i + 1) % 10 == 0 or (i + 1) == len(packets):
            print(f"Processed {i + 1}/{len(packets)} frames.")
            
    # Save any remaining frames in the last chunk
    if current_chunk_frames:
        save_chunk(current_chunk_frames, chunk_idx)
            
    end_time = time.time()
    print(f"Detection completed in {end_time - start_time:.2f} seconds.")
    
    # 5. Save metadata
    metadata_json = out_dir / "metadata.json"
    
    metadata = {
        "dataset": args.sequence_dir,
        "total_frames": len(packets),
        "chunk_size": args.chunk_size,
        "chunks": chunk_files,
        "detector": detector.get_model_info(),
        "generation_time": end_time - start_time
    }
    
    print(f"Saving metadata to {metadata_json}...")
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print("Done!")


if __name__ == "__main__":
    main()
