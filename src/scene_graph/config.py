import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field(..., description="Dataset type identifier, e.g. 'tum_rgbd'")
    name: Optional[str] = None
    root: Path = Field(..., description="Root directory of the dataset")
    
    @field_validator("root")
    @classmethod
    def validate_root(cls, v: Path) -> Path:
        if not v.exists() or not v.is_dir():
            pass
        return v

class SequenceConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    start_frame: int = Field(0, ge=0)
    end_frame: Optional[int] = Field(None, ge=0)
    
    @model_validator(mode='after')
    def validate_frames(self) -> 'SequenceConfig':
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) cannot be less than start_frame ({self.start_frame})")
        return self

class CameraConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    fx: float = Field(..., gt=0)
    fy: float = Field(..., gt=0)
    cx: float = Field(..., gt=0)
    cy: float = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

class DepthConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scale: float = Field(..., gt=0, description="Depth scale factor (e.g. 5000.0 for TUM)")

class PoseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str = Field("tum_groundtruth")

class SyncConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    rgb_depth_max_dt: float = Field(0.02, gt=0)
    rgb_pose_max_dt: float = Field(0.02, gt=0)

class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_valid_points: int = Field(30, gt=0)
    depth_outlier_band_m: float = Field(0.10, gt=0)

class VocabularyConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    classes: Tuple[str, ...]

class PerceptionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field("yoloe")
    model_path: str = Field("models/yoloe-26m-seg.pt")
    confidence_threshold: float = Field(0.40, ge=0.0, le=1.0)
    image_size: int = Field(640, gt=0)
    device: str = Field("auto")
    mode: str = Field("controlled")
    vocabulary: str = Field("v4_11")
    classes: Tuple[str, ...] = Field(default_factory=tuple)

class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    association_method: str = Field("hungarian")
    association_threshold_m: float = Field(0.25, gt=0)
    max_missing_frames: int = Field(5, ge=0)
    min_hits_to_confirm: int = Field(3, ge=1)
    velocity_history_min: int = Field(3, ge=1)

class ObjectTemporalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_hits_to_confirm: int = Field(3, ge=1)
    max_missing_frames: int = Field(5, ge=0)

class RelationTemporalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    confirm_frames: int = Field(3, ge=1)
    max_missing_frames: int = Field(2, ge=0)

class TemporalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    object: ObjectTemporalConfig = Field(default_factory=ObjectTemporalConfig)
    relation: RelationTemporalConfig = Field(default_factory=RelationTemporalConfig)

class DistanceRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    near_threshold: float = Field(0.50, gt=0)
    far_threshold: float = Field(1.50, gt=0)

class SupportRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plane_residual_m: float = Field(0.05, gt=0)
    min_support_overlap: float = Field(0.05, gt=0)

class DirectionalRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    margin_x: float = Field(0.1, gt=0)
    margin_y: float = Field(0.1, gt=0)

class ContainmentRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_containment_ratio: float = Field(0.5, gt=0, le=1.0)

class DepthOrderRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    depth_margin: float = Field(0.1, gt=0)

class OcclusionRelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    min_mask_overlap_ratio: float = Field(0.1, gt=0, le=1.0)
    min_depth_order_ratio: float = Field(0.7, gt=0, le=1.0)
    min_valid_depth_samples: int = Field(5, gt=0)
    depth_margin: float = Field(0.05, gt=0)

class RelationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    distance: DistanceRelationConfig = Field(default_factory=DistanceRelationConfig)
    support: SupportRelationConfig = Field(default_factory=SupportRelationConfig)
    directional: DirectionalRelationConfig = Field(default_factory=DirectionalRelationConfig)
    containment: ContainmentRelationConfig = Field(default_factory=ContainmentRelationConfig)
    depth_order: DepthOrderRelationConfig = Field(default_factory=DepthOrderRelationConfig)
    occlusion: OcclusionRelationConfig = Field(default_factory=OcclusionRelationConfig)

class RolesConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    support_surface: Tuple[str, ...] = Field(default_factory=lambda: ("desk", "table"))
    container: Tuple[str, ...] = Field(default_factory=tuple)
    ordinary_object: Tuple[str, ...] = Field(default_factory=lambda: ("monitor", "computer", "keyboard", "mouse", "telephone", "book", "cup", "pen", "paper", "bottle"))

class SceneGraphConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: Optional[DatasetConfig] = None
    sequence: Optional[SequenceConfig] = None
    camera: Optional[CameraConfig] = None
    depth: Optional[DepthConfig] = None
    pose_source: Optional[str] = None
    pose: Optional[PoseConfig] = None
    sync: Optional[SyncConfig] = Field(default_factory=SyncConfig)
    geometry: Optional[GeometryConfig] = Field(default_factory=GeometryConfig)
    perception: Optional[PerceptionConfig] = Field(default_factory=PerceptionConfig)
    tracking: Optional[TrackingConfig] = Field(default_factory=TrackingConfig)
    temporal: Optional[TemporalConfig] = Field(default_factory=TemporalConfig)
    relations: Optional[RelationConfig] = Field(default_factory=RelationConfig)
    roles: Optional[RolesConfig] = Field(default_factory=RolesConfig)
    vocabularies: Optional[Dict[str, VocabularyConfig]] = None
    evaluation_window: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_files(cls, base_path: str | Path, override_path: str | Path | None = None) -> "SceneGraphConfig":
        def load_yaml(path: str | Path) -> dict:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}

        def merge_dicts(base: dict, override: dict) -> dict:
            merged = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = merge_dicts(merged[key], value)
                else:
                    merged[key] = value
            return merged
            
        base_dict = load_yaml(base_path)
        if override_path:
            override_dict = load_yaml(override_path)
            base_dict = merge_dicts(base_dict, override_dict)
            
        # Ensure we don't blow up if some old field exists or if pose_source is present
        # Pydantic by default allows extra fields (unless model_config is set)
        return cls.model_validate(base_dict)
