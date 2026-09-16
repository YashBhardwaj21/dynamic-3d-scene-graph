from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector, PerceptionMode
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline


class OnlinePipeline:
    """Online pipeline combining open-vocabulary YOLOE perception with scene-graph processing."""

    def __init__(self, config: SceneGraphConfig):
        self.config = config

        mode_str = getattr(config.perception, "mode", None)
        prompts = getattr(config.perception, "prompts", None) or getattr(config.perception, "classes", None)
        if mode_str is None:
            mode = PerceptionMode.TEXT_PROMPT if prompts else PerceptionMode.PROMPT_FREE
        else:
            mode = PerceptionMode.from_str(mode_str)

        model_path = getattr(config.perception, "model_path", None) or getattr(config.perception, "model", "models/yoloe-26m-seg.pt")
        image_size = getattr(config.perception, "image_size", 480)

        self.detector = YOLOEDetector(
            model_path=model_path,
            confidence_threshold=config.perception.confidence_threshold,
            mode=mode,
            text_prompts=list(prompts) if prompts else None,
            device=getattr(config.perception, "device", "auto"),
            image_size=image_size,
        )

        self.core = SceneGraphPipeline(config)
        self.last_observations = None

    def update(self, packet: FramePacket):
        """Process one frame and update the scene graph."""
        observations = self.detector.detect(packet)
        self.last_observations = observations
        return self.core.update(packet, observations)