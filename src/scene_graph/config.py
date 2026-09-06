import yaml
from pathlib import Path
from typing import Optional, Dict, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Dataset type identifier")
    name: Optional[str] = None
    root: Path = Field(..., description="Root directory of the dataset")

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: Path) -> Path:
        if not value.exists() or not value.is_dir():
            raise ValueError(f"Dataset root does not exist or is not a directory: {value}")
        return value


class SequenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(0, ge=0)
    end_frame: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def validate_frames(self) -> "SequenceConfig":
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) cannot be less than start_frame ({self.start_frame})"
            )
        return self


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: float = Field(..., gt=0)
    fy: float = Field(..., gt=0)
    cx: float = Field(..., gt=0)
    cy: float = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class DepthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = Field(..., gt=0)


class PoseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field("tum_groundtruth")


class SyncConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rgb_depth_max_dt: float = Field(0.02, gt=0)
    rgb_pose_max_dt: float = Field(0.02, gt=0)


class RobustDepthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field("mad")
    k: float = Field(3.5, gt=0)


class DownsamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voxel_size_m: float = Field(0.005, gt=0)


class PlaneRansacConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(100, ge=1)
    min_inliers: int = Field(3, ge=3)
    random_seed: int = Field(0, ge=0)


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_valid_points: int = Field(30, gt=0)
    robust_depth: RobustDepthConfig = Field(default_factory=RobustDepthConfig)
    downsampling: DownsamplingConfig = Field(default_factory=DownsamplingConfig)
    measurement_noise_std_m: float = Field(0.01, gt=0)
    plane_ransac: PlaneRansacConfig = Field(default_factory=PlaneRansacConfig)


class VocabularyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    classes: Tuple[str, ...]


class PerceptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field("yoloe")
    model_path: str = Field("models/yoloe-26m-seg.pt")
    confidence_threshold: float = Field(0.40, ge=0.0, le=1.0)
    image_size: int = Field(640, gt=0)
    device: str = Field("auto")
    mode: str = Field("controlled")
    vocabulary: str = Field("v4_11")
    classes: Tuple[str, ...] = Field(default_factory=tuple)


class TrackingConfirmationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_hits: int = Field(3, ge=1)


class TrackingOcclusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_missing_seconds: float = Field(1.0, ge=0.0)


class TrackingProcessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acceleration_std_mps2: float = Field(1.0, gt=0)


class TrackingMeasurementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_std_m: float = Field(0.02, gt=0)


class TrackingInitializationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_variance_m2: float = Field(0.01, gt=0)
    velocity_variance_m2s2: float = Field(1.0, gt=0)


class TrackingGatingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chi2_probability: float = Field(0.999, gt=0, lt=1)


class TrackingAssociationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_distance_m: float = Field(0.50, gt=0)
    size_weight: float = Field(0.25, ge=0)


class TrackingHistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_buffer_size: int = Field(10, ge=1)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: TrackingConfirmationConfig = Field(default_factory=TrackingConfirmationConfig)
    occlusion: TrackingOcclusionConfig = Field(default_factory=TrackingOcclusionConfig)
    process: TrackingProcessConfig = Field(default_factory=TrackingProcessConfig)
    measurement: TrackingMeasurementConfig = Field(default_factory=TrackingMeasurementConfig)
    initialization: TrackingInitializationConfig = Field(default_factory=TrackingInitializationConfig)
    gating: TrackingGatingConfig = Field(default_factory=TrackingGatingConfig)
    association: TrackingAssociationConfig = Field(default_factory=TrackingAssociationConfig)
    history: TrackingHistoryConfig = Field(default_factory=TrackingHistoryConfig)


class RelationTemporalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_threshold: float = Field(0.8, gt=0)
    contradiction_threshold: float = Field(-0.8, lt=0)
    decay_per_second: float = Field(0.1, ge=0)
    unknown_after_seconds: float = Field(2.0, ge=0)
    lost_after_seconds: float = Field(5.0, ge=0)


class TemporalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation: RelationTemporalConfig = Field(default_factory=RelationTemporalConfig)


class DistanceRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    near_threshold: float = Field(0.50, gt=0)
    far_threshold: float = Field(1.50, gt=0)


class SupportRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plane_residual_m: float = Field(0.05, gt=0)
    min_support_overlap: float = Field(0.05, gt=0, le=1.0)
    contact_tolerance_m: float = Field(0.02, gt=0)
    min_contact_density: float = Field(0.05, gt=0, le=1.0)
    min_plane_points: int = Field(10, gt=2)
    min_plane_alignment_cosine: float = Field(0.8, gt=0, le=1.0)


class DirectionalRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    margin_x: float = Field(0.1, gt=0)
    margin_y: float = Field(0.1, gt=0)
    uncertainty_sigma: float = Field(2.0, gt=0)


class ContainmentRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_containment_ratio: float = Field(0.5, gt=0, le=1.0)


class DepthOrderRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depth_margin: float = Field(0.1, gt=0)
    uncertainty_sigma: float = Field(2.0, gt=0)


class OcclusionRelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_mask_overlap_ratio: float = Field(0.1, gt=0, le=1.0)
    min_depth_order_ratio: float = Field(0.7, gt=0, le=1.0)
    min_valid_depth_samples: int = Field(5, gt=0)
    depth_margin: float = Field(0.05, gt=0)


class RelationAdmissibilityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    object: str


class RelationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance: DistanceRelationConfig = Field(default_factory=DistanceRelationConfig)
    support: SupportRelationConfig = Field(default_factory=SupportRelationConfig)
    directional: DirectionalRelationConfig = Field(default_factory=DirectionalRelationConfig)
    containment: ContainmentRelationConfig = Field(default_factory=ContainmentRelationConfig)
    depth_order: DepthOrderRelationConfig = Field(default_factory=DepthOrderRelationConfig)
    occlusion: OcclusionRelationConfig = Field(default_factory=OcclusionRelationConfig)
    admissibility: Dict[str, RelationAdmissibilityRule] = Field(default_factory=dict)


class RolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support_surface: Tuple[str, ...] = Field(default_factory=lambda: ("desk", "table"))
    container: Tuple[str, ...] = Field(default_factory=tuple)
    ordinary_object: Tuple[str, ...] = Field(
        default_factory=lambda: (
            "monitor",
            "computer",
            "keyboard",
            "mouse",
            "telephone",
            "book",
            "cup",
            "pen",
            "paper",
            "bottle",
        )
    )


class SceneGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
            with open(path, "r") as file:
                return yaml.safe_load(file) or {}

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

        return cls.model_validate(base_dict)


def load_config(path: str | Path) -> SceneGraphConfig:
    return SceneGraphConfig.from_files(
        base_path="configs/default.yaml",
        override_path=path,
    )