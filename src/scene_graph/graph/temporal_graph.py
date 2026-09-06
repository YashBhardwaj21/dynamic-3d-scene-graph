from typing import Dict, List, Tuple

from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge
from scene_graph.temporal.object_state import ObjectState
from scene_graph.temporal.relation_state import RelationState


class TemporalSceneGraph:
    """
    Maintains the stateful, dynamic network of objects and relations over time.
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        # Key: (subject_id, object_id, predicate) -> GraphEdge
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        
        self.current_frame_index: int = -1
        self.current_timestamp: float = -1.0

    def update_nodes(self, tracks: list, object_states: Dict[str, ObjectState]):
        """Update the set of nodes in the graph based on tracks and states."""
        # Add or update nodes
        for track in tracks:
            state = object_states.get(track.object_id, ObjectState.REMOVED)
            
            if track.object_id in self.nodes:
                # Update existing
                node = self.nodes[track.object_id]
                node.track = track
                node.state = state
            elif state != ObjectState.REMOVED:
                # Add new
                self.nodes[track.object_id] = GraphNode(
                    object_id=track.object_id,
                    class_name=track.class_name,
                    track=track,
                    state=state
                )
                
        # Remove nodes that are fully REMOVED
        for obj_id, state in list(object_states.items()):
            if state == ObjectState.REMOVED and obj_id in self.nodes:
                del self.nodes[obj_id]
                # Also remove all edges connected to this node
                self._remove_edges_for_node(obj_id)

    def update_edges(self, evidences: list, relation_states: Dict[Tuple[str, str, str], RelationState]):
        """Update the set of edges in the graph based on evidence and states."""
        
        # Build lookup for latest evidence
        latest_ev = {}
        for ev in evidences:
            key = (ev.subject_id, ev.object_id, ev.predicate)
            latest_ev[key] = ev
            
        for key, state in relation_states.items():
            subject_id, object_id, predicate = key
            
            # Edges can only exist if both nodes exist in the graph
            if subject_id not in self.nodes or object_id not in self.nodes:
                continue
                
            if key in self.edges:
                edge = self.edges[key]
                edge.state = state
                if key in latest_ev:
                    edge.latest_evidence = latest_ev[key]
            elif state != RelationState.LOST and key in latest_ev:
                self.edges[key] = GraphEdge(
                    predicate=predicate,
                    subject_id=subject_id,
                    object_id=object_id,
                    state=state,
                    latest_evidence=latest_ev[key]
                )
                
        # Cleanup LOST edges
        for key, state in list(relation_states.items()):
            if state == RelationState.LOST and key in self.edges:
                del self.edges[key]

    def _remove_edges_for_node(self, node_id: str):
        keys_to_remove = [
            k for k in self.edges.keys() 
            if k[0] == node_id or k[1] == node_id
        ]
        for k in keys_to_remove:
            del self.edges[k]

    def get_active_nodes(self) -> List[GraphNode]:
        """Return all STABLE nodes."""
        return [node for node in self.nodes.values() if node.is_active]
        
    def get_active_edges(self) -> List[GraphEdge]:
        """Return all CONFIRMED edges connecting STABLE nodes."""
        return [
            edge for edge in self.edges.values() 
            if edge.is_active and 
               self.nodes[edge.subject_id].is_active and 
               self.nodes[edge.object_id].is_active
        ]
