from typing import List, Dict, Optional
from scene_graph.graph.event import GraphEvent, GraphEventType

class GraphHistory:
    """Maintains a chronological append-only log of all structural changes to the graph."""
    
    def __init__(self):
        self.events: List[GraphEvent] = []
        
    def add_event(self, event: GraphEvent):
        """Record a new event."""
        self.events.append(event)
        
    def get_events_for_object(self, object_id: str) -> List[GraphEvent]:
        """Retrieve all events (node or edge) involving a specific object."""
        return [
            e for e in self.events 
            if e.subject_id == object_id or e.object_id == object_id
        ]
        
    def get_events_in_window(self, start_time: float, end_time: float) -> List[GraphEvent]:
        """Retrieve all events that occurred within a time window."""
        return [
            e for e in self.events 
            if start_time <= e.timestamp <= end_time
        ]
