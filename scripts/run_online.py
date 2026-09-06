import argparse
import time
import sys

from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.graph.query import QueryEngine


def main():
    parser = argparse.ArgumentParser(description="Run the Dynamic Scene Graph pipeline online.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--realtime", action="store_true", help="Replay matching physical timestamps")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    # Load config with defaults
    config = SceneGraphConfig.from_files("configs/default.yaml", args.config)
    
    # Initialize pipeline
    pipeline = OnlinePipeline(config)
    
    # Initialize data source
    source = TUMReplaySource(config)
    
    print(f"Starting online replay (Total frames: {len(source)})...")
    
    last_timestamp = None
    query_engine = None
    
    for i, packet in enumerate(source):
        if args.max_frames and i >= args.max_frames:
            break
            
        # Real-time replay pacing
        if args.realtime and last_timestamp is not None:
            dt = packet.timestamp - last_timestamp
            if dt > 0:
                time.sleep(dt / args.speed)
                
        last_timestamp = packet.timestamp
        
        # Core pipeline update
        t0 = time.time()
        graph = pipeline.update(packet)
        t1 = time.time()
        
        # Output minimal status
        nodes = graph.get_active_nodes()
        edges = graph.get_active_edges()
        fps = 1.0 / (t1 - t0) if t1 > t0 else 0
        
        sys.stdout.write(f"\rFrame {i:04d} | FPS: {fps:.1f} | Objects: {len(nodes)} | Relations: {len(edges)}")
        sys.stdout.flush()
        
        query_engine = QueryEngine(graph)

    print("\nReplay finished.")
    
    if query_engine:
        print("\n" + "="*50)
        print("FINAL SCENE GRAPH REPORT")
        print("="*50)
        
        active_nodes = query_engine.graph.get_active_nodes()
        active_edges = query_engine.graph.get_active_edges()
        
        print(f"\n1. OBJECTS ({len(active_nodes)} active):")
        node_map = {}
        for node in active_nodes:
            track = node.track
            node_map[track.object_id] = track.class_name
            age = track.last_observed_frame - track.first_observed_frame
            print(f"  - {track.object_id} [{track.class_name.upper()}] (tracked for {age} frames, conf: {track.detection_confidence:.2f})")
            
        print(f"\n2. RELATIONS ({len(active_edges)} active):")
        # Group relations by predicate for cleaner reading
        from collections import defaultdict
        grouped_edges = defaultdict(list)
        for edge in active_edges:
            subj_cls = node_map.get(edge.subject_id, "unknown")
            obj_cls = node_map.get(edge.object_id, "unknown")
            grouped_edges[edge.predicate].append(f"{subj_cls}({edge.subject_id}) -> {obj_cls}({edge.object_id})")
            
        for pred, items in sorted(grouped_edges.items()):
            print(f"  [{pred}]: {len(items)} relationships")
            for item in items[:10]: # Print up to 10 per category to avoid spam
                print(f"      {item}")
            if len(items) > 10:
                print(f"      ... and {len(items) - 10} more")
                
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
