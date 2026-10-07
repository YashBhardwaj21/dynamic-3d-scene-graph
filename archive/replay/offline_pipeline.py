from scene_graph.config import SceneGraphConfig
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline


class OfflinePipeline:
    """Offline pipeline for replaying stored observation streams.
    
    This is used for the fixed-perception validation experiments, ensuring
    all configurations consume the exact same observation inputs.
    """
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.core = SceneGraphPipeline(config)
        
    def process_sequence(self, source):
        """Process an entire stream of frames/observations.
        
        Assumes source produces (FramePacket, List[Observation]) tuples.
        """
        states = []
        for frame_packet, observations in source:
            state = self.core.update(frame_packet, observations)
            states.append(state)
        return states
