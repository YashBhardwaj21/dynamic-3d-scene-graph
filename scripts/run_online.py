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

    # Load config
    config = SceneGraphConfig.from_files(args.config)
    
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
        print("\nFinal State Summary:")
        for edge in query_engine.graph.get_active_edges():
            print(f"  {edge.subject_id} {edge.predicate} {edge.object_id}")


if __name__ == "__main__":
    main()
