#!/usr/bin/env python3
"""Run YOLOE detector on all frames and save the observation stream (Substage 2.5)."""

import argparse
import json
import time
from pathlib import Path
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import TUMReplaySource
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
    parser.add_argument("--config", type=str, default="configs/tum_fr1_desk.yaml")
    parser.add_argument("--out_dir", type=str, default="data/processed/tum_fr1_desk")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--mode", type=str, choices=["controlled", "discovery"], default="controlled")
    args = parser.parse_args()
    
    # 1. Load config
    config = SceneGraphConfig.from_files("configs/default.yaml", args.config)
    
    seq_dir = Path(config.dataset.root)
    if not seq_dir.is_dir():
        print(f"Dataset directory not found: {seq_dir}")
        return
        
    out_dir = Path(args.out_dir)
    
    # Vocabulary Definitions
    vocab_config = config.vocabularies.get(config.perception.vocabulary)
    if args.mode == "discovery":
        vocab_config = config.vocabularies.get("expanded")
        
    obs_dir_name = "observations_controlled" if args.mode == "controlled" else "observations_discovery"
    obs_dir = out_dir / obs_dir_name
    obs_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Camera and Loader
    intrinsics = CameraIntrinsics(
        fx=config.camera.fx,
        fy=config.camera.fy,
        cx=config.camera.cx,
        cy=config.camera.cy,
        width=config.camera.width,
        height=config.camera.height
    )
    depth_model = DepthModel(scale=config.depth.scale)
    
    print(f"Initializing TUMReplaySource for {seq_dir}...")
    source = TUMReplaySource(config)
    total_frames = len(source)
    print(f"Source will yield {total_frames} packets.")
    
    # 3. Setup Detector
    model_path = config.perception.model_path
    if args.mode == "discovery":
        model_path = "models/yoloe/yoloe-26m-seg-pf.pt"
        
    print(f"Initializing YOLOEDetector with {model_path}...")
    try:
        detector = YOLOEDetector(
            model_path=model_path,
            confidence_threshold=config.perception.confidence_threshold,
            allowed_classes=set(vocab_config.classes) if vocab_config else set(),
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
    
    for i, packet in enumerate(source):
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
                "point_cloud_ref": obs.point_cloud_ref
            }
            
            geo = obs.object_geometry
            if geo is not None:
                obs_dict.update({
                    "centroid_camera": convert_for_json(geo.centroid_camera),
                    "centroid_world": convert_for_json(geo.centroid_world),
                    "bbox_min_world": convert_for_json(geo.bbox_min_world),
                    "bbox_max_world": convert_for_json(geo.bbox_max_world),
                    "depth_stats": geo.depth_stats,
                    "points_world_sampled": convert_for_json(geo.points_world_sampled),
                    "valid_point_count": geo.valid_point_count
                })
            else:
                obs_dict.update({
                    "centroid_camera": None,
                    "centroid_world": None,
                    "bbox_min_world": None,
                    "bbox_max_world": None,
                    "depth_stats": None,
                    "points_world_sampled": None,
                    "valid_point_count": 0
                })
                
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
            
        if (i + 1) % 10 == 0 or (i + 1) == total_frames:
            print(f"Processed {i + 1}/{total_frames} frames.")
            
    # Save any remaining frames in the last chunk
    if current_chunk_frames:
        save_chunk(current_chunk_frames, chunk_idx)
            
    end_time = time.time()
    print(f"Detection completed in {end_time - start_time:.2f} seconds.")
    
    # 5. Save metadata
    metadata_json = out_dir / "metadata.json"
    
    metadata = {
        "dataset": config.dataset.name,
        "total_frames": total_frames,
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

