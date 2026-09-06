from scene_graph.config import SceneGraphConfig
from scene_graph.data.tum_source import FramePacket
from scene_graph.perception.yoloe_detector import YOLOEDetector
from scene_graph.pipeline.pipeline_core import SceneGraphPipeline


class OnlinePipeline:
    """Online pipeline for real-time observation generation.
    
    This wraps the core SceneGraphPipeline with a live detector, ensuring
    observations are generated dynamically for each frame.
    """
    
    def __init__(self, config: SceneGraphConfig):
        self.config = config
        
        # 1. Perception
        model_path = config.perception.model_path
        
        intrinsics = None
        depth_model = None
        if config.camera:
            from scene_graph.geometry.camera import CameraIntrinsics
            intrinsics = CameraIntrinsics(
                fx=config.camera.fx, fy=config.camera.fy,
                cx=config.camera.cx, cy=config.camera.cy,
                width=config.camera.width, height=config.camera.height
            )
        if config.depth:
            from scene_graph.geometry.camera import DepthModel
            depth_model = DepthModel(scale=config.depth.scale)
            
        allowed_classes = None
        if config.perception.classes:
            allowed_classes = set(config.perception.classes)
        elif config.perception.vocabulary and config.vocabularies and config.perception.vocabulary in config.vocabularies:
            allowed_classes = set(config.vocabularies[config.perception.vocabulary].classes)
            
        if not allowed_classes:
            raise ValueError("No classes provided for open-vocabulary detector. Set perception.classes or a valid perception.vocabulary.")
            
        self.detector = YOLOEDetector(
            model_path=model_path,
            confidence_threshold=config.perception.confidence_threshold,
            allowed_classes=allowed_classes,
            intrinsics=intrinsics,
            depth_model=depth_model
        )
        
        # 2. Core Logic
        self.core = SceneGraphPipeline(config)
        
    def update(self, packet: FramePacket):
        """Process a single frame and update the scene graph."""
        observations = self.detector.detect(packet)
        return self.core.update(packet, observations)
