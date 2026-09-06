from scene_graph.pipeline.pipeline_core import PipelineCore


class OfflinePipeline:
    """Offline pipeline for replaying stored observation streams.
    
    This is used for the fixed-perception validation experiments, ensuring
    all configurations consume the exact same observation inputs.
    """
    
    def __init__(self, config):
        self.config = config
        self.core = PipelineCore(config)
        # In offline mode, the core's detector should be the StoredObservationLoader
        
    def process_sequence(self, source):
        """Process an entire stream of frames/observations."""
        states = []
        for frame_packet in source:
            state = self.core.update(frame_packet)
            states.append(state)
        return states
