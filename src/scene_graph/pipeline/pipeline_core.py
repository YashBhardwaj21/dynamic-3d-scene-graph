from scene_graph.tracking.causal_tracker import CausalTracker

class PipelineCore:
    """Shared update logic for both online and offline execution.
    
    Receives a single FramePacket and updates the entire causal scene graph state.
    """
    
    def __init__(self, config):
        self.config = config
        self.tracker = CausalTracker(self.config)
        # TODO: Initialize detector, geometry cacher, relation registry, graph
        
    def update(self, frame_packet, observations=None):
        """Called once per input frame. Causal execution."""
        # 1. Perception
        # If running online, we run the detector here. If offline, observations are passed in.
        # if observations is None:
        #     observations = self.detector.detect(frame_packet)
        
        # 2. Geometry
        # geometry = self._cache_geometry(observations, frame_packet)
        
        # 3. Tracking
        if observations is not None:
            tracks = self.tracker.update(observations, frame_packet.frame_index)
        else:
            tracks = []
        
        # 4. Relations
        # context = FrameContext(frame_packet, geometry, ...)
        # evidences = self.relation_registry.compute_all(tracks, context)
        
        # 5. Graph & Temporal State
        # state = self.temporal_graph.update(frame_packet.frame_index, tracks, evidences)
        
        return {"tracks": tracks}

