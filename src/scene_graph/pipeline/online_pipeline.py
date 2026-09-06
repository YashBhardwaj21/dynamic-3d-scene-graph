from scene_graph.pipeline.pipeline_core import PipelineCore


class OnlinePipeline:
    """Stateful, causal pipeline for streaming or live execution.
    
    Starts from an empty state and processes FramePackets sequentially.
    """
    
    def __init__(self, config):
        self.config = config
        self.core = PipelineCore(config)
        
    def process_frame(self, frame_packet):
        """Process a single frame packet. Returns the updated state."""
        return self.core.update(frame_packet)
