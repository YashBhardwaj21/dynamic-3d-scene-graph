from typing import Dict, List, Tuple

from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.graph.event import GraphEvent, GraphEventType
from scene_graph.graph.graph_history import GraphHistory


class TemporalSceneGraph:
    """
    Maintains the stateful, dynamic network of objects and relations over time.
    
    Key invariant: The graph represents the CURRENT belief about the physical scene,
    not an accumulation of everything ever observed. Dead edges are removed, not just
    marked as REMOVED.
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        # Key: (subject_id, object_id, predicate) -> GraphEdge
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        
        self.history = GraphHistory()
        
        self.current_frame_index: int = -1
        self.current_timestamp: float = -1.0

    def update_nodes(self, tracks: list, object_states: Dict[str, ObjectState]):
        """Update the set of nodes in the graph based on tracks and states."""
        # Add or update nodes
        for track in tracks:
            state = object_states.get(track.object_id, ObjectState.UNKNOWN)
            
            if track.object_id in self.nodes:
                # Update existing
                node = self.nodes[track.object_id]
                node.track = track
                node.state = state
                # Reactivate if it was removed but is now stable again
                if state == ObjectState.STABLE and node.participation == GraphParticipationState.REMOVED:
                    node.participation = GraphParticipationState.ACTIVE
                elif state == ObjectState.UNKNOWN:
                    node.participation = GraphParticipationState.REMOVED
            elif state != ObjectState.UNKNOWN:
                # Add new
                self.nodes[track.object_id] = GraphNode(
                    object_id=track.object_id,
                    class_name=track.class_name,
                    track=track,
                    state=state
                )
                self.history.add_event(GraphEvent(
                    event_type=GraphEventType.NODE_ADDED,
                    timestamp=self.current_timestamp,
                    frame_index=self.current_frame_index,
                    subject_id=track.object_id
                ))
                
        # Mark UNKNOWN nodes as REMOVED and cascade to edges
        for obj_id, state in list(object_states.items()):
            if state == ObjectState.UNKNOWN and obj_id in self.nodes:
                if self.nodes[obj_id].participation != GraphParticipationState.REMOVED:
                    self.nodes[obj_id].participation = GraphParticipationState.REMOVED
                    self.history.add_event(GraphEvent(
                        event_type=GraphEventType.NODE_REMOVED,
                        timestamp=self.current_timestamp,
                        frame_index=self.current_frame_index,
                        subject_id=obj_id
                    ))
                self._remove_edges_for_node(obj_id)

    def update_edges(self, evidences: list, relation_states: Dict[Tuple[str, str, str], RelationState]):
        """Update edges to reflect only the current relation state machine output.
        
        Edges are added when SUPPORTED, updated when state changes, and DELETED
        when the state machine garbage-collects a key or the state is CONTRADICTED/UNKNOWN.
        """
        
        # Build lookup for latest evidence
        latest_ev = {}
        for ev in evidences:
            key = (ev.subject_id, ev.object_id, ev.predicate)
            latest_ev[key] = ev
        
        # 1. Remove edges whose keys no longer exist in the state machine
        #    (they were garbage-collected)
        stale_keys = [k for k in self.edges if k not in relation_states]
        for key in stale_keys:
            self._remove_edge(key)
            
        # 2. Update or create edges based on current state
        for key, state in relation_states.items():
            subject_id, object_id, predicate = key
            
            # Edges can only exist if both nodes exist and are active
            if subject_id not in self.nodes or object_id not in self.nodes:
                # Clean up orphaned edges
                if key in self.edges:
                    self._remove_edge(key)
                continue
            
            if state == RelationState.SUPPORTED:
                if key in self.edges:
                    # Update existing edge
                    edge = self.edges[key]
                    edge.state = state
                    edge.participation = GraphParticipationState.ACTIVE
                    if key in latest_ev:
                        edge.latest_evidence = latest_ev[key]
                elif key in latest_ev:
                    # Create new edge
                    self.edges[key] = GraphEdge(
                        predicate=predicate,
                        subject_id=subject_id,
                        object_id=object_id,
                        state=state,
                        latest_evidence=latest_ev[key],
                        participation=GraphParticipationState.ACTIVE,
                        start_time=self.current_timestamp,
                        start_frame=self.current_frame_index
                    )
                    self.history.add_event(GraphEvent(
                        event_type=GraphEventType.EDGE_ADDED,
                        timestamp=self.current_timestamp,
                        frame_index=self.current_frame_index,
                        subject_id=subject_id,
                        object_id=object_id,
                        predicate=predicate
                    ))
            elif state in (RelationState.CONTRADICTED, RelationState.UNKNOWN):
                # Remove the edge entirely — the graph is current belief only
                if key in self.edges:
                    self._remove_edge(key)
                # Hypothesized relations are NOT yet in the graph.
                # They exist in the state machine but haven't earned a graph edge.
                # If there was an old edge, remove it (the relation was contradicted
                # and is now re-hypothesized).
                if key in self.edges:
                    self._remove_edge(key)

    def _remove_edges_for_node(self, node_id: str):
        """Remove all edges connected to a given node."""
        keys_to_remove = [
            k for k in self.edges
            if k[0] == node_id or k[1] == node_id
        ]
        for k in keys_to_remove:
            self._remove_edge(k)
            
    def _remove_edge(self, key: Tuple[str, str, str]):
        """Helper to remove edge and log event."""
        if key in self.edges:
            edge = self.edges[key]
            edge.end_time = self.current_timestamp
            edge.end_frame = self.current_frame_index
            self.history.add_event(GraphEvent(
                event_type=GraphEventType.EDGE_REMOVED,
                timestamp=self.current_timestamp,
                frame_index=self.current_frame_index,
                subject_id=key[0],
                object_id=key[1],
                predicate=key[2]
            ))
            del self.edges[key]

    def get_active_nodes(self) -> List[GraphNode]:
        """Return all STABLE nodes."""
        return [node for node in self.nodes.values() if node.is_active]
        
    def get_active_edges(self) -> List[GraphEdge]:
        """Return all CONFIRMED edges connecting STABLE nodes."""
        return [
            edge for edge in self.edges.values() 
            if edge.is_active and 
               edge.subject_id in self.nodes and
               edge.object_id in self.nodes and
               self.nodes[edge.subject_id].is_active and 
               self.nodes[edge.object_id].is_active
        ]
