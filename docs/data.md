# Data Architecture

## FrameSource
FrameSource serves as the interface boundary between disk/ROS and the core pipeline algorithms.

`python
class FrameSource(ABC):
    def __iter__(self) -> Iterator[FramePacket]: ...
`

## FramePacket
The FramePacket avoids dataset-specific fields (e.g., TUM timestamps). It acts as a standardized data container carrying:
- rame_index
- 	imestamp
- gb
- depth
- world_T_camera
- camera_model
- metadata

## Configuration
Configuration is strongly typed using Pydantic schemas. This ensures validation at startup and provides helpful errors for missing or invalid parameters.
