"""Detector Interface (Substage 2.3)."""

from abc import ABC, abstractmethod
from typing import List
from scene_graph.data.frame_packet import FramePacket
from scene_graph.perception.observation import Observation


class DetectorInterface(ABC):
    """Abstract base class for object detectors.
    
    Detectors take a FramePacket (containing RGB, optional Depth and Pose)
    and return a list of Observations.
    """
    
    @abstractmethod
    def detect(self, packet: FramePacket) -> List[Observation]:
        """Detect objects in the frame packet.
        
        Args:
            packet: A synchronized frame packet.
            
        Returns:
            List of Observations detected in the frame.
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model version and config information for reproducibility."""
        pass
