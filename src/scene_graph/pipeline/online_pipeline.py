from scene_graph.config import SceneGraphConfig
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline


class OnlinePipeline:
    """Online pipeline combining YOLOE perception with scene-graph processing."""

    def __init__(self, config: SceneGraphConfig):
        self.config = config

        allowed_classes = None

        if config.perception.classes:
            allowed_classes = tuple(config.perception.classes)
        elif (
            config.perception.vocabulary
            and config.vocabularies
            and config.perception.vocabulary in config.vocabularies
        ):
            allowed_classes = tuple(
                config.vocabularies[config.perception.vocabulary].classes
            )

        if not allowed_classes:
            raise ValueError(
                "No classes provided for detector. "
                "Set perception.classes or a valid perception.vocabulary."
            )

        self.detector = YOLOEDetector(
            model_path=config.perception.model_path,
            confidence_threshold=config.perception.confidence_threshold,
            allowed_classes=allowed_classes,
        )

        self.core = SceneGraphPipeline(config)

    def update(self, packet: FramePacket):
        """Process one frame and update the scene graph."""
        observations = self.detector.detect(packet)
        return self.core.update(packet, observations)