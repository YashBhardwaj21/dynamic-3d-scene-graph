import argparse
import time
import sys

from scene_graph.config import load_config
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.pipeline.online_pipeline import OnlinePipeline
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.query import QueryEngine


def main():
    parser = argparse.ArgumentParser(description="Run the Dynamic Scene Graph pipeline online.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--realtime", action="store_true", help="Replay matching physical timestamps")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    # Load config with defaults
    config = load_config(args.config)
    
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
        print(f"FRAME {query_engine.graph.current_frame_index}")
        print("="*50)
        
        active_nodes = query_engine.graph.get_active_nodes()
        all_edges = query_engine.graph.edges.values()
        
        active_edges = [e for e in all_edges if e.is_active]
        uncertain_edges = [e for e in all_edges if e.participation == GraphParticipationState.ACTIVE and e.state != RelationState.SUPPORTED]
        
        print(f"\nOBJECTS ({len(active_nodes)})")
        node_map = {}
        for node in active_nodes:
            track = node.track
            node_map[track.object_id] = track.class_name
            print(f"  {track.object_id}  {track.class_name.upper()}")
            print(f"      state: {node.state.value.upper()}")
            print(f"      confidence: {track.detection_confidence:.2f}\n")
            
        print(f"ACTIVE RELATIONS ({len(active_edges)})")
        # Deduplicate and format cleanly
        for edge in sorted(active_edges, key=lambda e: (e.predicate, e.subject_id, e.object_id)):
            subj_cls = node_map.get(edge.subject_id, "unknown")
            obj_cls = node_map.get(edge.object_id, "unknown")
            print(f"  {subj_cls} --{edge.predicate}--> {obj_cls}")
            
        print(f"\nTEMPORALLY UNCERTAIN ({len(uncertain_edges)})")
        for edge in sorted(uncertain_edges, key=lambda e: (e.predicate, e.subject_id, e.object_id)):
            subj_cls = node_map.get(edge.subject_id, "unknown")
            obj_cls = node_map.get(edge.object_id, "unknown")
            print(f"  {subj_cls} --{edge.predicate}--> {obj_cls}")
            print(f"      state: {edge.state.value.upper()}")
            print(f"      last_confirmed_frame: {edge.latest_evidence.frame_index if edge.latest_evidence else 'N/A'}\n")
            
        print("GRAPH SUMMARY")
        print(f"  nodes: {len(active_nodes)}")
        print(f"  active edges: {len(active_edges)}")
        print(f"  uncertain edges: {len(uncertain_edges)}")
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
