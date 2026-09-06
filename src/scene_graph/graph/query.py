from typing import List, Optional

from scene_graph.graph.temporal_graph import TemporalSceneGraph
from scene_graph.graph.node import GraphNode
from scene_graph.graph.edge import GraphEdge


class QueryEngine:
    """
    API for querying the temporal scene graph.
    """
    def __init__(self, graph: TemporalSceneGraph):
        self.graph = graph

    def get_objects(self, class_name: Optional[str] = None) -> List[GraphNode]:
        """Return all active nodes, optionally filtered by class."""
        nodes = self.graph.get_active_nodes()
        if class_name:
            nodes = [n for n in nodes if n.class_name == class_name]
        return nodes

    def get_object_by_id(self, object_id: str) -> Optional[GraphNode]:
        """Return a specific active node."""
        node = self.graph.nodes.get(object_id)
        if node and node.is_active:
            return node
        return None

    def what_is(self, predicate: str, object_id: str) -> List[GraphNode]:
        """
        Query: What is [predicate] [object_id]?
        e.g., what_is("ON", "table_1") -> [cup_1, monitor_1]
        """
        edges = self.graph.get_active_edges()
        
        # We want to find edges where:
        # edge.predicate == predicate AND edge.object_id == object_id
        # and return the edge.subject_id
        
        subjects = []
        for edge in edges:
            if edge.predicate == predicate and edge.object_id == object_id:
                subjects.append(self.graph.nodes[edge.subject_id])
                
        return subjects

    def what_does(self, subject_id: str, predicate: str) -> List[GraphNode]:
        """
        Query: What does [subject_id] [predicate]?
        e.g., what_does("cup_1", "ON") -> [table_1]
        """
        edges = self.graph.get_active_edges()
        
        objects = []
        for edge in edges:
            if edge.subject_id == subject_id and edge.predicate == predicate:
                objects.append(self.graph.nodes[edge.object_id])
                
        return objects

    def get_relations_between(self, subject_id: str, object_id: str) -> List[GraphEdge]:
        """Get all active relations between two specific objects."""
        edges = self.graph.get_active_edges()
        return [
            e for e in edges 
            if e.subject_id == subject_id and e.object_id == object_id
        ]
